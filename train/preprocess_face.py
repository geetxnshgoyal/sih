"""
Build the HEAD_ONLY vs FULL_FACE ablation set from the re-extracted face mesh.

    .venv/bin/python train/preprocess_face.py

Reads  data/video_landmarks/**/*.npz   (pose 33x4, face 468x3, lh 21x3, rh 21x3)
Writes data/dataset_face.npz           X_head, X_full, y, signer, labels

Why this exists
---------------
`FACE_MODE = "FULL_FACE"` is the change HANDOFF.md queues up, and the honest way
to justify it is a controlled comparison, not a hope. This script emits BOTH
representations over the SAME clips, the same signers, and the same classes, so
`train_ablation.py` can attribute any accuracy difference to the face block and
nothing else.

That directness matters here specifically. HANDOFF.md §6 records a case where a
correct fix was reverted because it was measured on a test set structurally
incapable of detecting it. The face mesh carries eyebrow, eye and lip motion;
the only metric that can see it is one where the two arms differ solely in
whether those points are present.

It also unblocks the work now. The full re-extraction is 46 Zenodo parts, of
which 8 are done; this subset — 687 clips, 51 classes — is already on disk.

Layout note
-----------
Face points are appended AFTER the body block:

    [pose 0..22][left hand 21][right hand 21][face 48]  = 113

so landmarks 11 and 12 remain the shoulders and `features.anchor` works
unchanged on both arms. This matches N_POINTS in features.py.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features
from face import FACE_SUBSET

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "video_landmarks"
INDEX = ROOT / "data" / "signer_index.json"
OUT = ROOT / "data" / "dataset_face.npz"

POSE_KEEP = features.POSE_KEEP          # 23
N_HEAD = POSE_KEEP + 21 + 21            # 65
INCLUDE_ASPECT = 1920 / 1080            # every INCLUDE clip; see features.isotropic
N_FULL = N_HEAD + len(FACE_SUBSET)      # 113


def clean_label(raw: str) -> str:
    """'48. Hello' -> 'Hello'"""
    return raw.split(".", 1)[-1].strip() if "." in raw else raw.strip()


def signer_lookup() -> dict:
    """Recover signer groups by matching (category, class, clip stem).

    The face-mesh clips are re-extractions of the same source videos as the
    pose release, so the existing group assignment carries over exactly.
    """
    index = json.loads(INDEX.read_text())
    table = {}
    for rel, meta in index.items():
        parts = rel.split("/")            # data/Pose_Signs/<cat>/<class>/<file>.pkl
        if len(parts) != 5:
            continue
        table[(parts[2], parts[3], os.path.splitext(parts[4])[0])] = meta["signer"]
    return table


def build_points(npz) -> tuple[np.ndarray, np.ndarray]:
    """One clip -> (head-only (T,65,3), full-face (T,113,3)) in unit coordinates.

    MediaPipe already returns normalised x,y here, so there is no to_unit step —
    unlike the pose release, which stores pixels.
    """
    pose = npz["pose"][:, :POSE_KEEP, :3]   # drop the visibility channel and the legs
    lh = npz["lh"][:, :, :3]
    rh = npz["rh"][:, :, :3]
    head = np.concatenate([pose, lh, rh], axis=1).astype(np.float64)

    face = npz["face"][:, FACE_SUBSET, :3].astype(np.float64)
    full = np.concatenate([head, face], axis=1)
    return head, full


def main() -> int:
    files = sorted(SRC.rglob("*.npz"))
    if not files:
        print(f"no clips under {SRC.relative_to(ROOT)} — run the extract loop first")
        return 1
    print(f"{len(files)} clips with face mesh")

    signers = signer_lookup()
    labels = sorted({clean_label(f.parent.name) for f in files})
    label_id = {name: i for i, name in enumerate(labels)}
    print(f"{len(labels)} classes")

    n = len(files)
    X_head = np.zeros((n, features.SEQ_LEN, N_HEAD, 3), dtype=np.float32)
    X_full = np.zeros((n, features.SEQ_LEN, N_FULL, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    signer = np.zeros(n, dtype=np.int32)

    kept = 0
    failed: list[str] = []
    unmapped = 0
    for i, path in enumerate(files):
        if i % 200 == 0 and i:
            print(f"  {i}/{n}")
        try:
            rel = path.relative_to(ROOT).parts   # data/video_landmarks/<cat>/<class>/<file>
            key = (rel[2], rel[3], path.stem)
            if key not in signers:
                unmapped += 1
                continue

            with np.load(path) as npz:
                head, full = build_points(npz)

            # Same transform as features.extract, minus the final standardisation:
            # train.py standardises after augmenting, in coordinate space.
            # These were re-extracted from INCLUDE's own video, which is
            # uniformly 1920x1080 (verified: every vid_shape in the pose
            # release is (1080, 1920)). See features.isotropic.
            head = features.isotropic(head, INCLUDE_ASPECT)
            full = features.isotropic(full, INCLUDE_ASPECT)
            seq_head = features.resample(features.anchor(head))
            seq_full = features.resample(features.anchor(full))
            if not (np.all(np.isfinite(seq_head)) and np.all(np.isfinite(seq_full))):
                failed.append(str(path.name))
                continue

            X_head[kept] = seq_head
            X_full[kept] = seq_full
            y[kept] = label_id[clean_label(path.parent.name)]
            signer[kept] = signers[key]
            kept += 1
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{path.name}: {exc}")

    X_head, X_full = X_head[:kept], X_full[:kept]
    y, signer = y[:kept], signer[:kept]

    print(f"\nkept {kept}, failed {len(failed)}, unmapped {unmapped}")
    for f in failed[:3]:
        print(f"  ! {f}")

    # A class can end up empty if every one of its clips was unmapped or failed.
    # Keeping it would add a softmax unit that can never be correct and would
    # quietly depress top-1 for both arms of the ablation.
    present = np.bincount(y, minlength=len(labels)) > 0
    if not present.all():
        dropped = [labels[i] for i in np.flatnonzero(~present)]
        print(f"dropping {len(dropped)} class(es) with no surviving clips: {', '.join(dropped)}")
        remap = -np.ones(len(labels), dtype=np.int32)
        remap[present] = np.arange(present.sum(), dtype=np.int32)
        y = remap[y]
        labels = [labels[i] for i in np.flatnonzero(present)]

    counts = np.bincount(y, minlength=len(labels))
    print(f"samples per class: min {counts.min()} median {int(np.median(counts))} max {counts.max()}")
    print("group sizes:", np.bincount(signer).tolist())
    print(f"X_head {X_head.shape}   X_full {X_full.shape}")

    np.savez_compressed(
        OUT, X_head=X_head, X_full=X_full, y=y, signer=signer, labels=np.array(labels)
    )
    print(f"\nwrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
