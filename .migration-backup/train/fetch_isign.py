"""
Stream the iSign (ISLTranslate) pose corpus into the shared feature contract.

    .venv-tf/bin/python train/fetch_isign.py --limit 2000
    .venv-tf/bin/python train/fetch_isign.py            # everything

Reads  huggingface.co/datasets/Exploration-Lab/iSign, poses parts aa..ad
Writes data/isign/shard_XXX.npz   X (n,32,65,3) float32, uid (n,)
       data/isign/_done_shards     what has been written, for resume

Why it streams instead of downloading
-------------------------------------
The four parts are 158 GB of a split ZIP. Converted to the 65-point contract the
same corpus is about 3 GB, because a 32-frame resample of 65 points throws away
almost everything a 25 fps 576-point recording holds. So the archive is inflated
in flight and only the features are kept: nothing near 158 GB ever lands on disk.

The format, established by reading it rather than assuming
----------------------------------------------------------
pose-format v0.1, data (frames, people, 576, 3):

    POSE_LANDMARKS         33   indices   0..32
    FACE_LANDMARKS        468            33..500
    LEFT_HAND_LANDMARKS    21           501..521
    RIGHT_HAND_LANDMARKS   21           522..542
    POSE_WORLD_LANDMARKS   33           543..575

Pose 0..22 plus the two hands is exactly our 65-point layout, so no remapping.
Coordinates are PIXELS on a 300x300 frame, hence /300 and aspect 1.0. The aspect
is stated in the file rather than guessed, which is the failure that has cost
this project three times: see train/clip_io.py.

Dims are 3, not the 4 the header's "XYZC" format string implies. That was worth
an hour: hand-decoding put the frame stride at 576*3 and the size arithmetic at
576*4, and only the library settles it. Hence pose_format is a dependency here
rather than a reimplementation.
"""
import argparse
import io
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import features  # noqa: E402

OUT = ROOT / "data" / "isign"
PARTS = [f"iSign-poses_v1.1_part_a{c}" for c in "abcd"]
URL = "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/{}"
SHARD = 2000          # clips per shard
POSE, LH, RH = (0, 33), (501, 522), (522, 543)


def token() -> str:
    p = Path.home() / ".cache" / "huggingface" / "token"
    if not p.exists():
        raise SystemExit("no huggingface token at ~/.cache/huggingface/token")
    return p.read_text().strip()


def part_sizes(tok: str) -> list[int]:
    """Byte length of each part, so a global offset maps to (part, offset)."""
    import urllib.request
    out = []
    for name in PARTS:
        req = urllib.request.Request(URL.format(name), method="HEAD",
                                     headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out.append(int(r.headers["Content-Length"]))
    return out


def stream_parts(tok: str, start: int = 0, sizes: list[int] | None = None):
    """The four parts, concatenated, as one byte stream, resuming on failure.

    158 GB on one connection for two hours does not survive: the first attempt
    died on an SSL read timeout with nothing written. Each part is therefore
    resumed with a Range request from the byte it stopped at, so a dropped
    connection costs seconds rather than the whole run.
    """
    import time
    import urllib.error
    import urllib.request
    sizes = sizes or [None] * len(PARTS)
    for pi, name in enumerate(PARTS):
        size = sizes[pi]
        # skip whole parts that lie entirely before the resume point
        if size is not None and start >= size:
            start -= size
            continue
        got = start
        start = 0
        while True:
            hdr = {"Authorization": f"Bearer {tok}"}
            if got:
                hdr["Range"] = f"bytes={got}-"
            try:
                req = urllib.request.Request(URL.format(name), headers=hdr)
                with urllib.request.urlopen(req, timeout=120) as r:
                    # A Range request does not survive Hugging Face's redirect to
                    # its CDN reliably. If the server answers 200 rather than 206
                    # it is sending the WHOLE file again, and treating those bytes
                    # as continuation silently desynchronises the ZIP parser: the
                    # first run stopped at 20.9 GB of a 45 GB part having read a
                    # duplicated prefix as if it were new data. So when a resume
                    # is not honoured, skip forward to where we actually were.
                    partial = r.status == 206
                    if size is None:
                        cl = r.headers.get("Content-Length")
                        size = int(cl) if cl else None
                    skip = 0 if (partial or not got) else got
                    if skip:
                        print(f"    {name}: range ignored (HTTP {r.status}), "
                              f"discarding {skip/1e9:.2f} GB to realign", flush=True)
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        if skip:
                            drop = min(skip, len(chunk))
                            chunk = chunk[drop:]
                            skip -= drop
                            if not chunk:
                                continue
                        got += len(chunk)
                        yield chunk
                break                       # part finished cleanly
            except Exception as exc:        # noqa: BLE001
                if size is not None and got >= size:
                    break
                print(f"    {name}: {type(exc).__name__} at {got/1e9:.2f} GB, "
                      f"resuming", flush=True)
                time.sleep(5)


class Reader:
    """Buffered forward-only reader over the concatenated stream."""

    def __init__(self, gen, base=0):
        self.gen, self.buf, self.eof, self.pos = gen, b"", False, base

    def _fill(self, n):
        while len(self.buf) < n and not self.eof:
            try:
                self.buf += next(self.gen)
            except StopIteration:
                self.eof = True

    def read(self, n):
        self._fill(n)
        out, self.buf = self.buf[:n], self.buf[n:]
        self.pos += len(out)
        return out


def to_contract(data: np.ndarray) -> np.ndarray | None:
    """(T,1,576,3) pixels on 300x300 -> the shared 32x65x3 features."""
    pose = data[:, 0, POSE[0]:POSE[0] + features.POSE_KEEP, :3]
    lh = data[:, 0, LH[0]:LH[1], :3]
    rh = data[:, 0, RH[0]:RH[1], :3]
    pts = np.concatenate([pose, lh, rh], axis=1).astype(np.float64)
    pts[..., 0] /= 300.0
    pts[..., 1] /= 300.0
    pts[..., 2] /= 300.0
    present = (np.abs(lh).sum(axis=(1, 2)) > 0) | (np.abs(rh).sum(axis=(1, 2)) > 0)
    idx = np.flatnonzero(present)
    if idx.size < 8:
        return None
    lo = max(int(idx[0]) - 2, 0)
    hi = min(int(idx[-1]) + 3, len(present))
    pts = pts[lo:hi]
    if pts.shape[0] < 8:
        return None
    v = features.extract(pts, 1.0)          # 300x300, stated in the file
    return v if np.all(np.isfinite(v)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from pose_format import Pose

    OUT.mkdir(parents=True, exist_ok=True)
    done = OUT / "_done_shards"
    # Resume by CLIP, not by shard. The ZIP has to be re-streamed from the start
    # either way (it is parsed sequentially), but a clip already converted is not
    # converted again, so an interrupted run costs bandwidth and not work.
    seen = set()
    existing = sorted(OUT.glob("shard_*.npz"))
    for f in existing:
        try:
            seen.update(str(u) for u in np.load(f, allow_pickle=True)["uid"])
        except Exception:
            pass
    shard0 = len(existing)
    print(f"{len(existing)} shards on disk, {len(seen):,} clips already converted")

    tok = token()
    off_file = OUT / "_stream_offset"
    sizes = part_sizes(tok)
    start = int(off_file.read_text()) if off_file.exists() else 0
    if start:
        print(f"resuming the stream at {start/1e9:.2f} GB of {sum(sizes)/1e9:.1f} GB")
    r = Reader(stream_parts(tok, start, sizes), base=start)
    xs, uids = [], []
    kept = skipped = 0
    shard = shard0
    reused = 0
    while True:
        sig = r.read(4)
        if len(sig) < 4 or sig != b"PK\x03\x04":
            break
        head = r.read(26)
        (_, flag, meth, _, _, _, csize, usize, nlen, elen) = struct.unpack("<HHHHHIIIHH", head)
        name = r.read(nlen).decode("utf-8", "replace")
        r.read(elen)
        blob = r.read(csize)
        if not name.endswith(".pose") or meth != 8 or not blob:
            continue
        off_file.write_text(str(r.pos))
        if Path(name).stem in seen:
            reused += 1
            continue
        try:
            raw = zlib.decompress(blob, -15)
            p = Pose.read(raw)
            v = to_contract(np.ma.filled(p.body.data, 0.0))
        except Exception:
            v = None
        if v is None:
            skipped += 1
            continue
        xs.append(v)
        uids.append(Path(name).stem)
        kept += 1
        if len(xs) >= SHARD:
            sp = OUT / f"shard_{shard:03d}.npz"
            np.savez_compressed(sp, X=np.stack(xs).reshape(-1, features.SEQ_LEN,
                                features.N_POINTS, features.N_DIMS).astype(np.float32),
                                uid=np.array(uids))
            with done.open("a") as fh:
                fh.write(sp.name + "\n")
            print(f"  wrote {sp.name}  {kept} kept  {skipped} skipped  "
                  f"{r.pos/1e9:.1f} GB read", flush=True)
            xs, uids = [], []
            shard += 1
        if args.limit and kept >= args.limit:
            break
    if xs:
        sp = OUT / f"shard_{shard:03d}.npz"
        np.savez_compressed(sp, X=np.stack(xs).reshape(-1, features.SEQ_LEN,
                            features.N_POINTS, features.N_DIMS).astype(np.float32),
                            uid=np.array(uids))
        print(f"  wrote {sp.name} (final)")
    print(f"\n{kept} new clips kept, {reused} already had, {skipped} skipped, "
          f"{r.pos/1e9:.1f} GB streamed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
