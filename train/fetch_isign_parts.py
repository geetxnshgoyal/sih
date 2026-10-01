"""
Download the iSign pose archive one part at a time, parse locally, delete.

    .venv-tf/bin/python train/fetch_isign_parts.py

Reads  huggingface.co/datasets/Exploration-Lab/iSign, parts aa..ad (158 GB)
Writes data/isign/shard_XXX.npz   X (n,32,65,3) float32, uid (n,)
       data/isign/_parts_done     which parts are finished, for resume

Why not stream it
-----------------
The streaming version failed four times in different ways: a read timeout, a
Range header the CDN silently ignored (so a duplicated prefix was parsed as
continuation and the ZIP desynchronised), a restart that threw away 20 GB of
work, and finally an immediate timeout. Each fix was correct and the pattern
still said the approach was wrong.

Holding one HTTP connection open across 158 GB through a redirecting CDN means
reimplementing resumable download inside a ZIP parser, which entangles two hard
problems and makes every bug look like silent desynchronisation rather than an
error. So the two jobs are separated: curl handles the network, this handles the
archive, and the network is not in the parsing loop at all.

Entries span part boundaries
----------------------------
The parts are a byte-level split, not four independent zips, so an entry can
straddle a boundary. Whatever is left unconsumed at the end of a part is carried
forward and prepended to the next, which is why parts are processed in order and
the carry is written to disk between them: an interrupted run resumes without
re-downloading a finished part.
"""
import json
import struct
import subprocess
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
SHARD = 2000
CHUNK = 1 << 22


def token() -> str:
    return (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()


def part_size(name: str, tok: str) -> int | None:
    """Content-Length for one part, so "finished" is a fact and not a guess."""
    import urllib.request
    try:
        req = urllib.request.Request(URL.format(name), method="HEAD",
                                     headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        return None


def download(name: str, dest: Path, tok: str, expected: int | None = None) -> bool:
    """Fetch one part, resuming from whatever is already on disk.

    The retry loop is HERE and not inside curl, and that distinction is the
    whole bug that lost 24 hours. `-C -` computes its resume offset ONCE, when
    curl starts, from the file's size at that moment. `--retry` then re-runs the
    transfer within the same invocation and restarts from that original offset,
    discarding everything fetched since. Observed directly: the file reached
    9.05 GB overnight and was back to 0.43 GB by morning, because each of the
    100 network errors threw away all progress made since the process began.

    Calling curl afresh per attempt makes `-C -` recompute the offset against
    the file as it actually is, so a dropped connection costs one attempt rather
    than everything. Hugging Face honours Range on this URL (206 on three of
    three probes), so resumption genuinely works once curl is asked correctly.
    """
    import time
    attempt = 0
    stalled = 0
    while True:
        attempt += 1
        before = dest.stat().st_size if dest.exists() else 0
        if expected and before >= expected:
            return True
        rc = subprocess.run(
            ["curl", "-sSL", "-C", "-",
             # no --retry: one attempt per invocation, so the next one re-reads
             # the file size instead of rewinding to a stale offset
             "--connect-timeout", "30",
             "--speed-limit", "10000", "--speed-time", "120",
             "-H", f"Authorization: Bearer {tok}",
             "-o", str(dest), URL.format(name)]).returncode
        after = dest.stat().st_size if dest.exists() else 0
        if rc == 0 and (not expected or after >= expected):
            return True
        gained = after - before
        if gained <= 0:
            stalled += 1
            if stalled >= 12:
                print(f"    {name}: 12 attempts with no progress, giving up "
                      f"at {after/1e9:.2f} GB", flush=True)
                return False
        else:
            stalled = 0
        pct = f" ({after/expected*100:.1f}%)" if expected else ""
        print(f"    {name}: attempt {attempt} exit {rc}, +{gained/1e6:.0f} MB, "
              f"now {after/1e9:.2f} GB{pct}", flush=True)
        time.sleep(5)

def to_contract(data: np.ndarray) -> np.ndarray | None:
    pose = data[:, 0, 0:features.POSE_KEEP, :3]
    lh = data[:, 0, 501:522, :3]
    rh = data[:, 0, 522:543, :3]
    pts = np.concatenate([pose, lh, rh], axis=1).astype(np.float64) / 300.0
    present = (np.abs(lh).sum(axis=(1, 2)) > 0) | (np.abs(rh).sum(axis=(1, 2)) > 0)
    idx = np.flatnonzero(present)
    if idx.size < 8:
        return None
    lo, hi = max(int(idx[0]) - 2, 0), min(int(idx[-1]) + 3, len(present))
    pts = pts[lo:hi]
    if pts.shape[0] < 8:
        return None
    v = features.extract(pts, 1.0)          # 300x300, stated in the file
    return v if np.all(np.isfinite(v)) else None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abcd",
                    help="which parts to do this run, e.g. 'a' for just part_aa. "
                         "At the measured 1.15 MB/s a part is about 11 hours, so "
                         "doing one at a time is the sane unit of commitment.")
    args = ap.parse_args()
    from pose_format import Pose
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()

    seen = set()
    existing = sorted(OUT.glob("shard_*.npz"))
    for f in existing:
        try:
            seen.update(str(u) for u in np.load(f, allow_pickle=True)["uid"])
        except Exception:
            pass
    shard = len(existing)
    print(f"{shard} shards on disk, {len(seen):,} clips already converted")

    done_f = OUT / "_parts_done"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    carry_f = OUT / "_carry.bin"
    xs, uids, kept, skipped, reused = [], [], 0, 0, 0

    def flush():
        nonlocal xs, uids, shard
        if not xs:
            return
        sp = OUT / f"shard_{shard:03d}.npz"
        np.savez_compressed(sp, X=np.stack(xs).reshape(-1, features.SEQ_LEN,
                            features.N_POINTS, features.N_DIMS).astype(np.float32),
                            uid=np.array(uids))
        print(f"    wrote {sp.name}  {kept} kept  {reused} reused  {skipped} skipped",
              flush=True)
        xs, uids, shard = [], [], shard + 1

    for c in args.parts:
        name = f"iSign-poses_v1.1_part_a{c}"
        if name in done:
            print(f"  {name}: already parsed")
            continue
        pf = OUT / name
        if not download(name, pf, tok, part_size(name, tok)):
            print(f"  {name}: download failed, stopping (rerun to resume)")
            return 1

        buf = carry_f.read_bytes() if carry_f.exists() else b""
        with pf.open("rb") as fh:
            eof = False

            def top_up(target):
                """Read until buf holds `target` bytes, or the file runs out."""
                nonlocal buf, eof
                while len(buf) < target and not eof:
                    more = fh.read(CHUNK)
                    if not more:
                        eof = True
                    else:
                        buf += more

            while True:
                top_up(30)
                if len(buf) < 30:
                    break                       # end of this part
                if buf[:4] != b"PK\x03\x04":
                    break                       # central directory or misalign
                (_, _, meth, _, _, _, csize, _, nlen, elen) = \
                    struct.unpack("<HHHHHIIIHH", buf[4:30])
                need = 30 + nlen + elen + csize
                top_up(need)
                if len(buf) < need:
                    break                       # entry straddles into the next part
                nm = buf[30:30 + nlen].decode("utf-8", "replace")
                blob = buf[30 + nlen + elen:need]
                buf = buf[need:]
                if not nm.endswith(".pose") or meth != 8 or not blob:
                    continue
                stem = Path(nm).stem
                if stem in seen:
                    reused += 1
                    continue
                try:
                    v = to_contract(np.ma.filled(Pose.read(zlib.decompress(blob, -15)).body.data, 0.0))
                except Exception:
                    v = None
                if v is None:
                    skipped += 1
                    continue
                xs.append(v); uids.append(stem); seen.add(stem); kept += 1
                if len(xs) >= SHARD:
                    flush()
        carry_f.write_bytes(buf)
        pf.unlink(missing_ok=True)
        with done_f.open("a") as fh:
            fh.write(name + "\n")
        print(f"  {name}: parsed, {kept} kept so far, carry {len(buf)/1e6:.1f} MB",
              flush=True)

    flush()
    carry_f.unlink(missing_ok=True)
    print(f"\n{kept} new clips, {reused} reused, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
