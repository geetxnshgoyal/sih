#!/usr/bin/env python3
"""Build body + optional face tensors from every extracted video source."""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import features  # noqa: E402

OUT = ROOT / "data/dataset_face_motion.npz"
SOURCES = [
    ROOT / "data/ncert_landmarks",
    ROOT / "data/islgov_landmarks",
    ROOT / "data/shiksha_landmarks",
    ROOT / "data/cislr_landmarks",
]
FACE_POINTS = 48


def load_clip(path: Path):
    with np.load(path, allow_pickle=False) as item:
        pose, left, right = item["pose"], item["lh"], item["rh"]
        if len(pose) < 8 or left.shape != (len(pose), 21, 3) or right.shape != (len(pose), 21, 3):
            return None
        body = np.concatenate([pose[:, :features.POSE_KEEP, :3], left, right], axis=1)
        aspect = float(item["aspect"]) if "aspect" in item.files else 1.0
        if not .2 < aspect < 5:
            return None
        face = item["face"] if "face" in item.files else np.zeros((len(body), FACE_POINTS, 3))
        if face.shape[1] == 468:
            from face import FACE_SUBSET
            face = face[:, FACE_SUBSET]
        if face.shape != (len(body), FACE_POINTS, 3):
            face = np.zeros((len(body), FACE_POINTS, 3))
        mask = np.any(face != 0, axis=(1, 2)).astype(np.float32)
    combined = np.concatenate([body, face], axis=1)
    combined = features.resample(features.anchor(features.isotropic(combined, aspect)))
    body_out = combined[:, :65].astype(np.float32)
    face_out = combined[:, 65:].astype(np.float32)
    indices = np.round(np.linspace(0, len(mask) - 1, features.SEQ_LEN)).astype(int)
    sampled_mask = mask[indices]
    face_out *= sampled_mask[:, None, None]
    return body_out, face_out, sampled_mask


def main() -> int:
    files = [(source.name.replace("_landmarks", ""), path)
             for source in SOURCES if source.exists()
             for path in sorted(source.rglob("*.npz"))]
    rows = []
    for source, path in files:
        loaded = load_clip(path)
        if loaded is None:
            continue
        # Source is the conservative grouping when verified signer identity is
        # unavailable. It never pretends clip IDs are independent people.
        rows.append((path.parent.name.strip(), source, path, *loaded))
    if not rows:
        print("No valid extracted landmark clips. Run train/pipeline.py sync first.")
        return 1
    labels = sorted({row[0] for row in rows}, key=str.casefold)
    groups = sorted({row[1] for row in rows})
    label_id, group_id = {v: i for i, v in enumerate(labels)}, {v: i for i, v in enumerate(groups)}
    X_body = np.stack([row[3] for row in rows])
    X_face = np.stack([row[4] for row in rows])
    face_mask = np.stack([row[5] for row in rows])
    y = np.array([label_id[row[0]] for row in rows], dtype=np.int32)
    signer = np.array([group_id[row[1]] for row in rows], dtype=np.int32)
    clip_ids = np.array([hashlib.sha256(str(row[2]).encode()).hexdigest()[:16] for row in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X_body=X_body, X_face=X_face, face_mask=face_mask,
                        y=y, signer=signer, labels=np.array(labels),
                        groups=np.array(groups), clip_ids=clip_ids)
    tmp.replace(OUT)
    print(f"{len(rows)} clips, {len(labels)} labels, {len(groups)} conservative groups -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
