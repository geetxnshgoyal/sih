"""
Extract landmarks from CISLR video for the glosses we already train on.

    .venv-mp/bin/python train/extract_cislr.py

Why this exists
---------------
INCLUDE was recorded by seven deaf students at one school in Chennai. Every
measurement we have taken says the ceiling is signer diversity, not model
capacity: held-out-signer accuracy is 40.4% while held-out-clip accuracy is
near 100%, and neither the face mesh nor SL-GCN moved it. More signers is the
only lever the data supports.

CISLR is a different corpus, different signers, different rooms and cameras.
Its median gloss has ONE clip, so it cannot train its own 4,765-class model , 
but 612 of its clips carry 220 of our 264 labels. Those are worth having: they
add unseen people to classes that already have enough examples to learn.

We deliberately do NOT import CISLR's 4,545 other glosses. A class with one
example does not become a supported word by appearing in labels.json; it
becomes a word the model claims to know and cannot recognise, which is worse
than an honest gap. See ARCHITECTURE.md §13.

Output mirrors extract_video.py exactly, pose 33x4, face 468x3, lh/rh 21x3 , 
so preprocess_face.py can read both corpora with no special case.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "cislr" / "CISLR_v1.5-a_videos" / "CISLR_v1.5-a_videos.zip"
TODO = ROOT / "data" / "cislr" / "_todo.json"
WORK = ROOT / "data" / "_work_cislr"
OUT = ROOT / "data" / "cislr_landmarks"

N_POSE, N_FACE, N_HAND = 33, 468, 21


def to_array(landmarks, n: int, dims: int) -> np.ndarray:
    """A missing hand or face becomes zeros, matching extract_video.py."""
    out = np.zeros((n, dims), dtype=np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark[:n]):
        out[i, 0], out[i, 1], out[i, 2] = lm.x, lm.y, lm.z
        if dims == 4:
            out[i, 3] = getattr(lm, "visibility", 0.0)
    return out


def extract_clip(path: Path, holistic) -> dict[str, np.ndarray] | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    pose, face, lh, rh = [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose.append(to_array(res.pose_landmarks, N_POSE, 4))
        face.append(to_array(res.face_landmarks, N_FACE, 3))
        lh.append(to_array(res.left_hand_landmarks, N_HAND, 3))
        rh.append(to_array(res.right_hand_landmarks, N_HAND, 3))
    cap.release()
    if not pose:
        return None
    return {"pose": np.stack(pose), "face": np.stack(face),
            "lh": np.stack(lh), "rh": np.stack(rh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (smoke test)")
    args = ap.parse_args()

    if not ZIP.exists():
        print(f"missing {ZIP.relative_to(ROOT)}, run data/fetch_cislr.sh with FETCH_ALL=1")
        return 1
    todo = json.loads(TODO.read_text())
    if args.limit:
        todo = todo[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.complexity,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    done = skipped = failed = 0
    try:
        with zipfile.ZipFile(ZIP) as z:
            for i, item in enumerate(todo, 1):
                dest = OUT / item["gloss"] / f"{item['uid']}.npz"
                if dest.exists():
                    skipped += 1
                    continue
                # Stream one clip out of the archive at a time. Unpacking all
                # 7,051 would cost ~3 GB of disk to read 612 of them.
                tmp_vid = WORK / Path(item["path"]).name
                with z.open(item["path"]) as src, open(tmp_vid, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                data = extract_clip(tmp_vid, holistic)
                tmp_vid.unlink(missing_ok=True)
                if data is None:
                    print(f"  ! unreadable: {item['uid']}")
                    failed += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write: savez_compressed builds the zip incrementally,
                # so a kill mid-write leaves a truncated file that still
                # exists and would be skipped as done. The temp name must
                # already end in .npz or savez appends a second suffix.
                tmp = dest.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **data)
                tmp.replace(dest)
                done += 1
                if done % 25 == 0:
                    print(f"  {i}/{len(todo)}  ({done} new, {skipped} had)", flush=True)
    finally:
        holistic.close()
        shutil.rmtree(WORK, ignore_errors=True)

    print(f"\n{done} new, {skipped} already had, {failed} failed"
          f" -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
