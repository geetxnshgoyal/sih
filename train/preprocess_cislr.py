"""
Turn CISLR landmarks into model-ready arrays and merge them with INCLUDE.

    .venv/bin/python train/preprocess_cislr.py

Reads  data/cislr_landmarks/<gloss>/<uid>.npz  +  data/dataset.npz
Writes data/dataset_merged.npz    X, y, signer, labels, corpus

Why this exists
---------------
Held-out-signer accuracy is 40.4% while held-out-clip accuracy is near 100%.
That gap is the whole problem, and it is a data problem: INCLUDE is seven deaf
students in one room in Chennai. CISLR adds 612 clips of 220 of the same 264
glosses, recorded by different people in different rooms off different cameras.

Two things this script has to get right, or the merge makes the numbers worse
while looking like it made them better:

1. TRIMMING. CISLR clips are framed tighter than INCLUDE and run longer, so
   roughly half of every clip is the signer at rest with their hands below the
   bottom edge. features.resample strides across the WHOLE clip, so an untrimmed
   CISLR clip spends half its 32 frames on stillness while an INCLUDE clip
   spends almost none. The model would learn "long still lead-in" as a corpus
   cue rather than learning the sign. We cut to the span where a hand is
   actually visible.

2. SIGNER IDs. CISLR groups are numbered AFTER INCLUDE's 0..2, never merged
   into them. Held-out-group evaluation is only honest if a group is one set of
   people; letting a CISLR clip land in INCLUDE group 0 would put the same
   corpus on both sides of the split and inflate the result, the same class of
   mistake as the 99.8% calibration run (ARCHITECTURE.md §12).

A `corpus` array rides along so evaluation can ask the question that actually
matters: train on one corpus, test on the other.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features
import signers as sg

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "cislr_landmarks"
BASE = ROOT / "data" / "dataset.npz"
ASPECT = ROOT / "data" / "cislr" / "_aspect.json"
OUT = ROOT / "data" / "dataset_merged.npz"

POSE_KEEP = features.POSE_KEEP          # 23
N_POINTS = POSE_KEEP + 21 + 21          # 65, head-only, matches dataset.npz

PAD = 3             # frames of lead-in kept either side of the active span
MIN_ACTIVE = 8      # a clip with fewer visible-hand frames is not a usable sign
CISLR_GROUPS = 4    # recovered signer clusters; see profile_signers.py


def build_points(npz) -> np.ndarray:
    """One clip -> (T, 65, 3) unit coordinates, same layout as dataset.npz."""
    pose = npz["pose"][:, :POSE_KEEP, :3]
    return np.concatenate([pose, npz["lh"][:, :, :3], npz["rh"][:, :, :3]],
                          axis=1).astype(np.float64)


def active_span(npz) -> tuple[int, int] | None:
    """First and last frame with a hand detected, padded.

    Hand presence rather than motion energy, because the thing being cut here
    is not stillness in general, it is the signer standing with their hands
    out of frame before and after the sign. Motion energy would also cut the
    hold at the end of a sign, which carries meaning.
    """
    present = (np.abs(npz["lh"]).sum(axis=(1, 2)) > 0) | \
              (np.abs(npz["rh"]).sum(axis=(1, 2)) > 0)
    idx = np.flatnonzero(present)
    if idx.size < MIN_ACTIVE:
        return None
    return max(int(idx[0]) - PAD, 0), min(int(idx[-1]) + PAD + 1, len(present))


def main() -> int:
    if not BASE.exists():
        print(f"missing {BASE.relative_to(ROOT)}, run train/preprocess.py first")
        return 1
    base = np.load(BASE, allow_pickle=True)
    labels = [str(s) for s in base["labels"]]
    label_id = {name: i for i, name in enumerate(labels)}
    n_include_groups = int(base["signer"].max()) + 1
    print(f"INCLUDE: {len(base['X'])} clips, {len(labels)} classes, "
          f"{n_include_groups} signer groups")

    files = sorted(SRC.rglob("*.npz"))
    if not files:
        print(f"no clips under {SRC.relative_to(ROOT)}, run extract_cislr.py first")
        return 1
    print(f"CISLR:   {len(files)} clips on disk")

    if not ASPECT.exists():
        print(f"missing {ASPECT.relative_to(ROOT)}, frame shapes are not optional,"
              f" see features.isotropic")
        return 1
    aspects = json.loads(ASPECT.read_text())
    missing = [f.stem for f in files if f.stem not in aspects]
    if missing:
        print(f"  ! {len(missing)} clips have no measured frame shape, e.g. {missing[:3]}")
        return 1

    seqs, ys, props = [], [], []
    short = unknown = bad = 0
    for path in files:
        gloss = path.parent.name
        if gloss not in label_id:
            unknown += 1
            continue
        try:
            with np.load(path) as npz:
                span = active_span(npz)
                if span is None:
                    short += 1
                    continue
                pts = build_points(npz)[span[0]:span[1]]
            # Every CISLR clip measured is 300x300, so this is 1.0 and the step
            # is a no-op today: which is exactly why it must be explicit. The
            # square format is the geometrically correct one; INCLUDE is the
            # corpus that needs correcting. See features.isotropic.
            anchored = features.anchor(features.isotropic(pts, aspects[path.stem]))
            seq = features.resample(anchored)
            if not np.all(np.isfinite(seq)):
                bad += 1
                continue
            seqs.append(seq.astype(np.float32))
            ys.append(label_id[gloss])
            # proportions on the anchored sequence, exactly as profile_signers
            # does for INCLUDE, so the clustering sees the same feature space
            props.append(sg.proportions(anchored))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {path.name}: {exc}")
            bad += 1

    if not seqs:
        print("no usable CISLR clips")
        return 1
    Xc = np.stack(seqs)
    yc = np.asarray(ys, dtype=np.int32)
    print(f"  kept {len(Xc)}  (dropped: {short} too short, {unknown} off-vocabulary, "
          f"{bad} unusable)")

    # Recover signer groups from body proportions, then shift past INCLUDE's.
    P = np.asarray(props)
    P = (P - P.mean(0)) / np.maximum(P.std(0), 1e-6)
    groups, _ = sg.kmeans(P, CISLR_GROUPS, seed=0)
    sc = (groups + n_include_groups).astype(np.int32)
    print(f"  CISLR signer groups {n_include_groups}..{n_include_groups + CISLR_GROUPS - 1}"
          f" sizes {np.bincount(groups).tolist()}")

    X = np.concatenate([base["X"], Xc])
    y = np.concatenate([base["y"], yc])
    signer = np.concatenate([base["signer"], sc])
    corpus = np.concatenate([np.zeros(len(base["X"]), np.int32),
                             np.ones(len(Xc), np.int32)])

    features.check_isotropy(Xc[:400], "CISLR")
    features.check_isotropy(base["X"][:400], "INCLUDE")

    counts = np.bincount(y, minlength=len(labels))
    gained = np.bincount(yc, minlength=len(labels))
    print(f"\nmerged {len(X)} clips, {len(labels)} classes")
    print(f"  classes gaining CISLR clips: {int((gained > 0).sum())}")
    print(f"  per class: min {counts.min()} median {int(np.median(counts))} "
          f"max {counts.max()}")

    np.savez_compressed(OUT, X=X, y=y, signer=signer, corpus=corpus,
                        labels=np.array(labels))
    print(f"\nwrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
