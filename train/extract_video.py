"""
Extract landmarks from INCLUDE raw video — including the 468-point face mesh.

    .venv-mp/bin/python train/extract_video.py            # all downloaded parts
    .venv-mp/bin/python train/extract_video.py --stream   # download, extract, delete

Why this exists
---------------
The OpenHands pose release carries 33 pose + 42 hand points and NO face mesh.
Its face coverage is 11 coarse pose landmarks (nose, eyes, ears, mouth
corners), which is enough for head tilt and nod but nothing else. ISL marks
grammar on the face: raised eyebrows for a yes/no question, furrowed for a
wh-question, mouth morphemes distinguishing otherwise identical handshapes.
None of that is recoverable without the mesh, so a hands-only model renders a
question as a statement.

Runs MediaPipe 0.10's legacy Holistic — the same tool AI4Bharat used to build
the pose release, so re-extracted landmarks stay consistent with what is
already trained. (MediaPipe 1.x's HolisticLandmarker is present in the Python
package but non-functional: "Check failed: service_ Service is unavailable".)

Output: data/video_landmarks/<Category>/<sign>/<clip>.npz with
    pose  (T, 33, 4)   x, y, z, visibility
    face  (T, 468, 3)
    lh    (T, 21, 3)
    rh    (T, 21, 3)
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "video"
WORK = ROOT / "data" / "_work"
OUT = ROOT / "data" / "video_landmarks"
ZENODO = "https://zenodo.org/record/4010759/files"

N_POSE, N_FACE, N_HAND = 33, 468, 21


def empty(n: int, dims: int) -> np.ndarray:
    return np.zeros((n, dims), dtype=np.float32)


def to_array(landmarks, n: int, dims: int) -> np.ndarray:
    """A missing hand or face becomes zeros, matching how the pose release
    stores undetected parts — so downstream code needs no special case."""
    if landmarks is None:
        return empty(n, dims)
    out = np.zeros((n, dims), dtype=np.float32)
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
    return {
        "pose": np.stack(pose), "face": np.stack(face),
        "lh": np.stack(lh), "rh": np.stack(rh),
    }


def process_zip(zip_path: Path, holistic, keep_video: bool) -> int:
    stage = WORK / zip_path.stem
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(stage)

    videos = [p for p in stage.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi"}]
    done = 0
    for v in videos:
        # mirror the archive's <Category>/<sign>/ layout in the output
        rel = v.relative_to(stage)
        dest = OUT / rel.parent / f"{v.stem}.npz"
        if dest.exists():
            done += 1
            continue
        data = extract_clip(v, holistic)
        if data is None:
            print(f"  ! unreadable: {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename, rather than writing dest in place.
        #
        # np.savez_compressed builds a zip incrementally, so a process killed
        # mid-write leaves a truncated .npz that still EXISTS — which means the
        # `dest.exists()` skip above treats it as done, and the corruption only
        # surfaces hours later as "Bad CRC-32" in preprocess_face.py. That is
        # exactly what happened on 31 Aug: 29 Jobs clips were lost this way when
        # the pipeline was killed by a /tmp wipe, and the source archive had
        # already been deleted by then.
        #
        # rename() is atomic on the same filesystem, so dest either does not
        # exist or is complete. A kill now leaves a .tmp that is simply ignored
        # and redone.
        # The temp name must itself end in .npz: savez_compressed APPENDS .npz
        # when the path does not already have it, so a ".npz.tmp" name silently
        # becomes ".npz.tmp.npz" and the rename below would target a file that
        # was never written.
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(dest)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(videos)}")

    shutil.rmtree(stage, ignore_errors=True)
    if not keep_video:
        # The landmarks are the deliverable; the video is not. 104 clips become
        # 39 MB of .npz from a 1.3 GB archive, so keeping the archives would
        # cost 57 GB for nothing. A `.extracted` marker stops the downloader
        # re-fetching a part we have already processed.
        zip_path.unlink(missing_ok=True)
        Path(f"{zip_path}.done").unlink(missing_ok=True)
        Path(f"{zip_path}.extracted").touch()
    return done


def zenodo_parts() -> list[str]:
    import json
    import urllib.request
    with urllib.request.urlopen("https://zenodo.org/api/records/4010759") as fh:
        rec = json.load(fh)
    return sorted(f["key"] for f in rec["files"] if f["key"].endswith(".zip"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", action="store_true",
                    help="download each part, extract, then delete it (low disk)")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--keep-video", action="store_true",
                    help="do not delete a part after extracting it")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.complexity,
        refine_face_landmarks=True,     # the whole point: 468-point mesh
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Only touch parts the downloader has marked complete. Without this the
    # extractor picks up a zip that is still being written and dies on
    # BadZipFile, taking the whole run with it.
    parts = zenodo_parts() if args.stream else sorted(
        p.name for p in VIDEO_DIR.glob("*.zip")
        if (VIDEO_DIR / f"{p.name}.done").exists()
        and not (VIDEO_DIR / f"{p.name}.extracted").exists()
    )
    if not parts and not args.stream:
        have = len(list(VIDEO_DIR.glob("*.zip")))
        print(f"no completed parts yet ({have} still downloading).")
        print("Downloads mark completion with a .done file; re-run when some land.")
        return 0
    print(f"{len(parts)} part(s) to process\n")

    total = 0
    try:
        for i, key in enumerate(parts, 1):
            zip_path = VIDEO_DIR / key
            if args.stream and not zip_path.exists():
                print(f"[{i}/{len(parts)}] downloading {key}")
                VIDEO_DIR.mkdir(parents=True, exist_ok=True)
                rc = subprocess.run(
                    ["curl", "-sL", "-C", "-", "--retry", "8", "--retry-delay", "5",
                     "-o", str(zip_path), f"{ZENODO}/{key}?download=1"]
                ).returncode
                if rc != 0:
                    print(f"  download failed, skipping {key}")
                    continue
            if not zip_path.exists():
                continue
            print(f"[{i}/{len(parts)}] extracting {key}")
            try:
                total += process_zip(zip_path, holistic,
                                     keep_video=args.keep_video and not args.stream)
            except zipfile.BadZipFile:
                # a truncated or corrupt part: drop the marker so the
                # downloader fetches it again, and carry on with the rest
                print(f"  corrupt archive, will re-download: {key}")
                Path(f"{zip_path}.done").unlink(missing_ok=True)
                zip_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  failed on {key}: {exc}")
    finally:
        holistic.close()

    print(f"\n{total} clips -> {OUT.relative_to(ROOT)}")
    print("Set FACE_MODE = 'FULL_FACE' in train/features.py to use the mesh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
