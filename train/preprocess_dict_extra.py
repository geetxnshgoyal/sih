"""
Add dictionary clips to the training set as extra SIGNERS of existing classes.

    .venv-tf/bin/python train/preprocess_dict_extra.py

Reads  data/{ncert,shiksha,islgov}_landmarks/<word>/*.npz
       data/dataset_merged.npz            for the label list and feature shape
Writes data/dict_extra.npz                X, y, signer, corpus, labels

Why this and not more classes
-----------------------------
The measured history of this project is blunt about what moves accuracy. Five
training-side changes did nothing between them. Cutting the vocabulary from 264
to 83 gave +8.4. Adding CISLR, a second corpus of DIFFERENT PEOPLE signing words
already in the set, gave +3.4. Signer diversity is the lever; class count is the
thing to spend sparingly.

So this adds no classes at all. It finds dictionary clips whose word is already
one of the shipped 83 and adds them as extra examples: 95 clips over 65 of the
83 classes, each one a person the model has never seen, filmed by a different
organisation. That is the CISLR lever again, from data already on disk.

They land as their own signer group so leave-one-group-out still means what it
says. Held out, they measure cross-corpus generalisation; held in, they are
extra signers for the classes they cover.

The honest risk, which the evaluation is there to settle: 95 clips is +5.7% of
1,668, they cover only 65 of 83 classes, and they come from a studio domain
quite unlike INCLUDE. That could add diversity or it could add skew. The
leave-one-group-out number decides it, not this docstring.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import clip_io  # noqa: E402
import features  # noqa: E402

BASE = ROOT / "data" / "dataset_merged.npz"
OUT = ROOT / "data" / "dict_extra.npz"
SOURCES = {"ncert": 3, "shiksha": 4, "islgov": 5}   # corpus ids, 0-2 already used
GROUP = 8                                            # signer group, 0-7 already used
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
    base = np.load(BASE, allow_pickle=True)
    labels = [str(x) for x in base["labels"]]
    by_norm = {norm(l): i for i, l in enumerate(labels)}
    print(f"base: {base['X'].shape[0]} clips, {len(labels)} classes")

    Xs, ys, cps = [], [], []
    kept = {}
    for sname, cid in SOURCES.items():
        d = ROOT / "data" / f"{sname}_landmarks"
        if not d.is_dir():
            continue
        n = 0
        for wd in sorted(d.iterdir()):
            if not wd.is_dir():
                continue
            li = by_norm.get(norm(wd.name))
            if li is None:
                continue                      # not a class we already train on
            for f in sorted(wd.glob("*.npz")):
                v = clip_features(f)
                if v is None:
                    continue
                Xs.append(v); ys.append(li); cps.append(cid); n += 1
                kept[labels[li]] = kept.get(labels[li], 0) + 1
        print(f"  {sname:8s} {n:4d} clips")

    if not Xs:
        print("nothing to add")
        return 1
    X = np.stack(Xs).reshape(len(Xs), features.SEQ_LEN, features.N_POINTS,
                             features.N_DIMS).astype(np.float32)
    y = np.array(ys, np.int32)
    signer = np.full(len(y), GROUP, np.int32)
    corpus = np.array(cps, np.int32)
    assert np.all(np.isfinite(X)), "non-finite feature"
    assert X.shape[1:] == base["X"].shape[1:], \
        f"shape {X.shape[1:]} != base {base['X'].shape[1:]}"

    features.check_isotropy(X[:200].reshape(-1, features.N_POINTS,
                                            features.N_DIMS), "dict_extra")
    np.savez_compressed(OUT, X=X, y=y, signer=signer, corpus=corpus,
                        labels=np.array(labels))
    print(f"\n{len(X)} clips over {len(set(ys))} existing classes, "
          f"signer group {GROUP}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
