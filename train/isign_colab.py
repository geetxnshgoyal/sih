"""
Convert the iSign pose corpus on Colab or Kaggle, where the network works.

Run this in a Colab or Kaggle notebook. It downloads the 158 GB archive there,
converts it to Setu's 65-point contract, and leaves ~3 GB of features to bring
home. Nothing of the archive comes back.

    !pip -q install pose-format
    !wget -q https://raw.githubusercontent.com/geetxnshgoyal/sih/main/train/isign_colab.py
    !python isign_colab.py --token hf_xxx --out /content/isign

Why not run it here
-------------------
Five consecutive download failures from this machine, the last of them curl
itself failing with Broken pipe after 0.1 GB through all eight retries. Four
were mine and were fixed; that one is a well-tested tool failing at the network
layer, which is a different kind of problem and not one more code can solve.
Colab and Kaggle sit next to Hugging Face's CDN and both have free tiers.

Parity is the whole point
-------------------------
Features computed here must be bit-identical to features computed by
train/features.py, or the model is trained on one thing and runs on another.
So features.py is FETCHED FROM THE REPO rather than reimplemented, and the
script refuses to run if it cannot get it. Do not paste a copy inline.
"""
import argparse
import struct
import subprocess
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np

FEATURES_URL = ("https://raw.githubusercontent.com/geetxnshgoyal/sih/"
                "main/train/features.py")
FACE_URL = ("https://raw.githubusercontent.com/geetxnshgoyal/sih/"
            "main/train/face.py")
PARTS = [f"iSign-poses_v1.1_part_a{c}" for c in "abcd"]
URL = "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/{}"
SHARD = 2000
CHUNK = 1 << 22


def load_features():
    """The repo's own feature code. Never a reimplementation."""
    here = Path(".")
    for name, url in (("face.py", FACE_URL), ("features.py", FEATURES_URL)):
        if not (here / name).exists():
            urllib.request.urlretrieve(url, name)
    sys.path.insert(0, str(here.resolve()))
    import features
    print(f"features.py: SEQ_LEN {features.SEQ_LEN}, N_POINTS {features.N_POINTS}, "
          f"contract {features.FEATURE_SIZE}")
    return features


def to_contract(data, features):
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
    v = features.extract(pts, 1.0)      # 300x300, stated in the file header
    return v if np.all(np.isfinite(v)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="huggingface token")
    ap.add_argument("--out", default="isign")
    ap.add_argument("--parts", default="abcd", help="which parts to do this run")
    args = ap.parse_args()

    features = load_features()
    from pose_format import Pose

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    done_f = out / "_parts_done"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    carry_f = out / "_carry.bin"

    seen = set()
    existing = sorted(out.glob("shard_*.npz"))
    for f in existing:
        try:
            seen.update(str(u) for u in np.load(f, allow_pickle=True)["uid"])
        except Exception:
            pass
    shard = len(existing)
    print(f"{shard} shards, {len(seen):,} clips already converted")

    xs, uids, kept, skipped, reused = [], [], 0, 0, 0

    def flush():
        nonlocal xs, uids, shard
        if not xs:
            return
        sp = out / f"shard_{shard:03d}.npz"
        np.savez_compressed(sp, X=np.stack(xs).reshape(-1, features.SEQ_LEN,
                            features.N_POINTS, features.N_DIMS).astype(np.float32),
                            uid=np.array(uids))
        print(f"    {sp.name}  kept {kept}  reused {reused}  skipped {skipped}",
              flush=True)
        xs, uids, shard = [], [], shard + 1

    for c in args.parts:
        name = f"iSign-poses_v1.1_part_a{c}"
        if name in done:
            print(f"  {name}: already parsed"); continue
        pf = out / name
        print(f"  downloading {name} ...", flush=True)
        rc = subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "20",
                             "--retry-delay", "5", "--retry-all-errors",
                             "-H", f"Authorization: Bearer {args.token}",
                             "-o", str(pf), URL.format(name)]).returncode
        if rc != 0:
            print(f"  {name}: download failed, rerun to resume"); return 1

        buf = carry_f.read_bytes() if carry_f.exists() else b""
        eof = False
        with pf.open("rb") as fh:
            def top_up(target):
                nonlocal buf, eof
                while len(buf) < target and not eof:
                    more = fh.read(CHUNK)
                    if not more:
                        eof = True
                    else:
                        buf += more
            while True:
                top_up(30)
                if len(buf) < 30 or buf[:4] != b"PK\x03\x04":
                    break
                (_, _, meth, _, _, _, csize, _, nlen, elen) = \
                    struct.unpack("<HHHHHIIIHH", buf[4:30])
                need = 30 + nlen + elen + csize
                top_up(need)
                if len(buf) < need:
                    break                       # straddles into the next part
                nm = buf[30:30 + nlen].decode("utf-8", "replace")
                blob = buf[30 + nlen + elen:need]
                buf = buf[need:]
                if not nm.endswith(".pose") or meth != 8 or not blob:
                    continue
                stem = Path(nm).stem
                if stem in seen:
                    reused += 1; continue
                try:
                    p = Pose.read(zlib.decompress(blob, -15))
                    v = to_contract(np.ma.filled(p.body.data, 0.0), features)
                except Exception:
                    v = None
                if v is None:
                    skipped += 1; continue
                xs.append(v); uids.append(stem); seen.add(stem); kept += 1
                if len(xs) >= SHARD:
                    flush()
        carry_f.write_bytes(buf)
        pf.unlink(missing_ok=True)
        with done_f.open("a") as fh:
            fh.write(name + "\n")
        print(f"  {name}: parsed. kept {kept}, carry {len(buf)/1e6:.1f} MB", flush=True)

    flush()
    total = sum(np.load(f)["X"].shape[0] for f in sorted(out.glob("shard_*.npz")))
    print(f"\n{kept} new, {reused} reused, {skipped} skipped. {total:,} clips total.")
    print(f"Bring home: {out}/shard_*.npz  "
          f"({sum(f.stat().st_size for f in out.glob('shard_*.npz'))/1e9:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
