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


def download(name: str, dest: Path, tok: str) -> bool:
    """curl -C -, because resumption is its job and it is not my code."""
    cmd = ["curl", "-sSL", "-C", "-", "--retry", "8", "--retry-delay", "5",
           "--retry-all-errors", "--connect-timeout", "30",
           "-H", f"Authorization: Bearer {tok}", "-o", str(dest), URL.format(name)]
    print(f"  downloading {name} ...", flush=True)
    return subprocess.run(cmd).returncode == 0


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

    for name in PARTS:
        if name in done:
            print(f"  {name}: already parsed")
            continue
        pf = OUT / name
        if not download(name, pf, tok):
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
