"""
Turn 4,284 pose clips into model-ready arrays.

    python train/preprocess.py

Reads  data/Pose_Signs/**/*.pkl  +  data/signer_index.json
Writes data/dataset.npz          X, y, signer, labels

Every clip goes through features.extract_from_raw, which is the same transform
the browser runs at inference time (guaranteed by train/test_parity.py).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "signer_index.json"
OUT = ROOT / "data" / "dataset.npz"


class SafeUnpickler(pickle.Unpickler):
    """Third-party pickles: permit numpy array reconstruction and nothing else."""
    ALLOW = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, mod, name):
        if (mod, name) not in self.ALLOW:
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        return super().find_class(mod, name)


def clean_label(raw: str) -> str:
    """'48. Hello' -> 'Hello'"""
    return raw.split(".", 1)[-1].strip() if "." in raw else raw.strip()


def main() -> int:
    index = json.loads(INDEX.read_text())
    print(f"{len(index)} clips indexed")

    labels = sorted({clean_label(v["class"]) for v in index.values()})
    label_id = {name: i for i, name in enumerate(labels)}
    print(f"{len(labels)} classes")

    # Anchored + resampled but NOT standardised: augmentation is geometric and
    # must happen in coordinate space, before the whole-vector normalisation.
    # train.py standardises after augmenting, exactly as features.extract does.
    X = np.zeros((len(index), features.SEQ_LEN, features.N_POINTS, features.N_DIMS),
                 dtype=np.float32)
    y = np.zeros(len(index), dtype=np.int32)
    signer = np.zeros(len(index), dtype=np.int32)

    kept = 0
    failed = []
    for i, (rel, meta) in enumerate(sorted(index.items())):
        if i % 500 == 0 and i:
            print(f"  {i}/{len(index)}")
        try:
            with open(ROOT / rel, "rb") as fh:
                obj = SafeUnpickler(fh).load()
            unit = features.to_unit(obj["keypoints"], obj["vid_shape"])
            # Undo MediaPipe's aspect-dependent normalisation before anchoring.
            # INCLUDE is uniformly 1920x1080, so without this every skeleton
            # here is stretched 1.78x vertically relative to a square-format
            # corpus or a differently-shaped webcam. See features.isotropic.
            unit = features.isotropic(unit, features.aspect_of(obj["vid_shape"]))
            seq = features.resample(features.anchor(unit))
            if not np.all(np.isfinite(seq)):
                failed.append(rel)
                continue
            X[kept] = seq
            y[kept] = label_id[clean_label(meta["class"])]
            signer[kept] = meta["signer"]
            kept += 1
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{rel}: {exc}")

    X, y, signer = X[:kept], y[:kept], signer[:kept]
    print(f"\nkept {kept}, failed {len(failed)}")
    if failed[:3]:
        for f in failed[:3]:
            print(f"  ! {f}")

    counts = np.bincount(y, minlength=len(labels))
    print(f"\nsamples per class: min {counts.min()}  median {int(np.median(counts))}  max {counts.max()}")
    print("group sizes:", np.bincount(signer).tolist())
    print(f"X {X.shape} {X.dtype}  (anchored, pre-standardisation)")

    np.savez_compressed(OUT, X=X, y=y, signer=signer, labels=np.array(labels))
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
