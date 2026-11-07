"""
Profile every clip's body proportions, cluster into signers, and validate.

    python train/profile_signers.py

Writes data/signer_index.json:  {clip_path: {"class":..., "signer": int}}
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features
import signers

ROOT = Path(__file__).resolve().parent.parent
POSE_DIR = ROOT / "data" / "Pose_Signs"
OUT = ROOT / "data" / "signer_index.json"
CACHE = ROOT / "data" / "signer_profiles.npz"

# INCLUDE states 7 signers; we sweep around that and pick by elbow + validation.
K_CANDIDATES = [3, 4, 5, 6, 7]

# Only ratios that stay stable while the person is signing. Arm-length ratios
# were tried and rejected: 2D foreshortening as the arms move toward the camera
# swamps the between-person difference (forearm/shoulder ranged 0.47-1.68
# within this dataset), so they encode the sign, not the signer.
STABLE = [0, 4, 5]   # head/shoulder, neck/shoulder, facedepth/head


class SafeUnpickler(pickle.Unpickler):
    """The .pkl files are third-party. Permit numpy array reconstruction only."""
    ALLOW = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, mod, name):
        if (mod, name) not in self.ALLOW:
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        return super().find_class(mod, name)


def load(path: Path):
    with open(path, "rb") as fh:
        return SafeUnpickler(fh).load()


def main() -> int:
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        X, meta = z["X"], list(z["meta"])
        print(f"loaded {len(X)} cached profiles from {CACHE.name}")
        return cluster(X, meta)

    paths = sorted(POSE_DIR.rglob("*.pkl"))
    print(f"profiling {len(paths)} clips...")

    X, meta = [], []
    bad = 0
    for i, p in enumerate(paths):
        if i % 500 == 0 and i:
            print(f"  {i}/{len(paths)}")
        try:
            o = load(p)
            unit = features.to_unit(o["keypoints"], o["vid_shape"])
            v = signers.proportions(unit)
            if not np.all(np.isfinite(v)) or not v.any():
                bad += 1
                continue
            X.append(v)
            meta.append({"path": str(p.relative_to(ROOT)), "cls": p.parts[-2]})
        except Exception:
            bad += 1

    X = np.asarray(X)
    np.savez(CACHE, X=X, meta=np.array(meta, dtype=object))
    print(f"\n{len(X)} profiled, {bad} unusable  (cached to {CACHE.name})")
    return cluster(X, meta)


def cluster(X, meta) -> int:
    print("\nfeature spread:")
    for i, nm in enumerate(signers.FEATURE_NAMES):
        print(f"  {nm:18s} std {X[:, i].std():.3f}   "
              f"range [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")

    # z-score so no single ratio dominates the distance metric
    S = X[:, STABLE]
    Z = (S - S.mean(0)) / (S.std(0) + 1e-9)

    n_classes = len({m["cls"] for m in meta})
    print(f"\nclustering ({n_classes} sign classes present)")
    print(f"{'k':>3} {'inertia':>10} {'smallest':>9} {'classes/cluster':>16}")

    results = {}
    for k in K_CANDIDATES:
        labels, C = signers.kmeans(Z, k, seed=7)
        inert = signers.inertia(Z, labels, C)
        sizes = np.bincount(labels, minlength=k)
        cover = [len({meta[i]["cls"] for i in np.where(labels == j)[0]}) for j in range(k)]
        results[k] = (labels, inert, sizes, cover)
        print(f"{k:3d} {inert:10.0f} {sizes.min():9d} {min(cover):5d}-{max(cover):<4d} of {n_classes}")

    # A cluster that is a real person should cover most of the vocabulary.
    # A cluster covering few classes is a class artefact, not a signer.
    print("\nvalidation: a real signer signed most of the vocabulary,")
    print("so every cluster should cover a large share of classes.")
    print("(These are body-type groups, not identified individuals. Holding one")
    print(" out is a conservative proxy for a signer-disjoint split.)")
    best = None
    for k in K_CANDIDATES:
        labels, inert, sizes, cover = results[k]
        frac = min(cover) / n_classes
        ok = frac > 0.5 and sizes.min() >= 50
        print(f"  k={k}: min coverage {frac*100:5.1f}%  min size {sizes.min():4d}  "
              f"{'usable' if ok else 'rejected'}")
        # Prefer the k whose WEAKEST group still covers the most classes.
        # More groups means a smaller held-out fold, but a group that never
        # signed half the vocabulary cannot test those classes at all.
        if ok and (best is None or frac > best[1]):
            best = (k, frac)

    best = best[0] if best else None
    if best is None:
        print("\nNo k produced signer-like clusters. Body proportions did not")
        print("separate the recording sessions. Falling back to a single group;")
        print("report accuracy as clip-level and say so explicitly.")
        labels = np.zeros(len(X), dtype=int)
        best = 1
    else:
        labels = results[best][0]
        print(f"\nusing k={best}")

    index = {m["path"]: {"class": m["cls"], "signer": int(l)}
             for m, l in zip(meta, labels)}
    OUT.write_text(json.dumps(index, indent=0))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(index)} clips, {best} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
