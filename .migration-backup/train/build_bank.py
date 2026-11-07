"""
Embed every landmark clip once, so retrieval experiments cost seconds not hours.

    .venv-tf/bin/python train/build_bank.py

Reads  data/{islgov,ncert,shiksha,cislr}_landmarks/<word>/*.npz
Writes run/bank.npz    emb (N,256) float32, word (N,), source (N,), clip (N,)

Why a cache
-----------
There are about 15,000 clips across four corpora. Turning each into the shared
32x65x3 feature contract and pushing it through the encoder takes tens of
minutes; every retrieval question asked afterwards takes under a second against
the cached vectors. Since the whole point of tonight is to ask many questions
(which bank size, how many references per word, seen versus unseen, which
threshold), paying that cost once is the difference between four experiments
and forty.

The embedding is the layer feeding the softmax of the shipped 83-class model:
everything it learned about the shape of a sign, minus its commitment to 83
specific answers.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import clip_io  # noqa: E402
import features  # noqa: E402

SOURCES = {
    "islgov":  ROOT / "data" / "islgov_landmarks",
    "ncert":   ROOT / "data" / "ncert_landmarks",
    "shiksha": ROOT / "data" / "shiksha_landmarks",
    "cislr":   ROOT / "data" / "cislr_landmarks",
}
OUT = ROOT / "run" / "bank.npz"
MIN_ACTIVE, PAD = 8, 2


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()


def clip_features(path: Path) -> np.ndarray | None:
    """One landmark npz -> the shared feature contract.

    The aspect comes from clip_io, which raises rather than defaulting to 1.0.
    That default was a bug in four files at once: the 13,662 Government
    dictionary clips carry no aspect field and are 16:9, so every one of them
    was processed as though a widescreen body were square.
    """
    pts, aspect = clip_io.load_points(path, features.POSE_KEEP, MIN_ACTIVE, PAD)
    if pts is None:
        return None
    v = features.extract(pts, aspect)
    return v if np.all(np.isfinite(v)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gloss_classifier.keras",
                    help="classifier whose penultimate layer is the embedding")
    ap.add_argument("--out", default="run/bank.npz")
    ap.add_argument("--encoder", action="store_true",
                    help="--model is already an encoder, not a classifier")
    args = ap.parse_args()

    import tensorflow as tf
    mp = ROOT / args.model
    full = tf.keras.models.load_model(mp, compile=False, safe_mode=False)
    enc = full if args.encoder else tf.keras.Model(full.input, full.layers[-2].output)
    probe = enc(np.zeros((1, features.SEQ_LEN,
                          features.N_POINTS * features.N_DIMS), np.float32))
    print(f"encoder {mp.name}, embedding dim {int(probe.shape[-1])}\n")

    words, srcs, clips, feats = [], [], [], []
    for sname, d in SOURCES.items():
        if not d.is_dir():
            print(f"  {sname:8s} missing, skipped")
            continue
        n = 0
        for wd in sorted(d.iterdir()):
            if not wd.is_dir():
                continue
            w = norm(wd.name)
            if not w:
                continue
            for f in sorted(wd.glob("*.npz")):
                v = clip_features(f)
                if v is None:
                    continue
                feats.append(v); words.append(w); srcs.append(sname); clips.append(f.stem)
                n += 1
        print(f"  {sname:8s} {n:6d} clips usable")

    if not feats:
        print("nothing to embed")
        return 1
    X = np.stack(feats).reshape(len(feats), features.SEQ_LEN,
                                features.N_POINTS * features.N_DIMS)
    print(f"\nembedding {len(X)} clips ...")
    E = enc.predict(X, batch_size=512, verbose=0).astype(np.float64)
    E[~np.isfinite(E)] = 0.0
    n = np.linalg.norm(E, axis=1, keepdims=True)
    E = np.divide(E, n, out=np.zeros_like(E), where=n > 1e-9)
    dead = int((n <= 1e-9).sum())

    assert np.all(np.isfinite(E)), "non-finite embedding survived normalisation"
    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, emb=E.astype(np.float32),
                        word=np.array(words), source=np.array(srcs),
                        clip=np.array(clips))
    print(f"  {len(E)} vectors, {len(set(words))} distinct words, {dead} dead")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
