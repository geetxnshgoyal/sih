"""
Stream the whole Government-of-India ISL Dictionary through MediaPipe.

    .venv-mp/bin/python train/extract_islgov.py --workers 4

Reads  data/meta/_islgov_files.json   (13,665 paths on the HF repo)
Writes data/islgov_landmarks/<word>/<clip>.npz   pose (T,33,4), lh/rh (T,21,3)

Why stream rather than download first
-------------------------------------
The dataset is ~75 GB of H.265 and this machine has 108 GB free. The deliverable
is the landmarks -- roughly 350 MB -- so each clip is fetched, extracted and
deleted in turn. Disk never holds more than the files currently in flight, and
the run is resumable: an existing .npz is skipped, so a kill costs only the clips
mid-flight.

What this data is and is not
----------------------------
13,665 clips, 12,103 unique words, MEDIAN ONE CLIP PER WORD. It is a dictionary,
so it cannot train a 12,103-class model any more than CISLR could -- see
ARCHITECTURE.md 2.2 for why importing single-example classes manufactures words
the model claims to know and cannot recognise.

It is worth extracting anyway, for two things it IS good for:

  - SELF-SUPERVISED PRETRAINING. 13,665 clips of Indian Sign Language needs no
    labels at all. AI4Bharat's raw_dpc showed cross-language self-supervision
    works; this is the ISL-specific version, and it is the largest ISL corpus
    that exists under a permissive licence.
  - VOCABULARY. 1,183 words have 2+ clips, which is the honest ceiling for
    expanding past the current 264 classes.

Face mesh is not extracted. FACE_MODE is HEAD_ONLY, the FULL_FACE ablation
measured a regression (ARCHITECTURE.md 9), and refine_face_landmarks is a large
fraction of per-frame cost. Turning it off roughly halves the run.

Licence: MIT. Unlike CISLR this is redistributable, so the fetch is reproducible
by anyone with no account and no licence acceptance.
"""
import argparse
import json
import multiprocessing as mp_proc
import os
import signal
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "data" / "meta" / "_islgov_files.json"
OUT = ROOT / "data" / "islgov_landmarks"
REPO = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
N_POSE, N_HAND = 33, 21
CHUNK = 25          # clips per task; small enough that workers get recycled


def word_of(path: str) -> str:
    """Directory name for a clip: the dictionary entry it belongs to."""
    stem = Path(path).stem
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in stem)[:120]


def to_array(landmarks, n: int, dims: int) -> np.ndarray:
    out = np.zeros((n, dims), dtype=np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark[:n]):
        out[i, 0], out[i, 1], out[i, 2] = lm.x, lm.y, lm.z
        if dims == 4:
            out[i, 3] = getattr(lm, "visibility", 0.0)
    return out


def worker(args) -> tuple[int, int, int]:
    shard, complexity, target_fps = args
    import cv2
    import mediapipe as mp

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=complexity,
        refine_face_landmarks=False,          # HEAD_ONLY; see the docstring
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    done = skipped = failed = 0
    try:
        for path in shard:
            dest = OUT / word_of(path) / (Path(path).stem + ".npz")
            if dest.exists():
                skipped += 1
                continue
            url = (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
                   + urllib.parse.quote(path))
            tmp_vid = None
            try:
                fd, tmp_vid = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                urllib.request.urlretrieve(url, tmp_vid)
                cap = cv2.VideoCapture(tmp_vid)
                # Sample at a fixed rate rather than taking every frame. Two
                # reasons, and the second matters more than the speed:
                #   - these entries average ~900 frames and are resampled to 32
                #     later anyway, so most of the decode is thrown away
                #   - source clips are 25 AND 30 fps, so a fixed frame stride
                #     would give different real-time windows for different
                #     entries. Deriving the stride from each clip's own fps
                #     makes a 32-frame window mean the same duration everywhere.
                src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
                stride = max(1, int(round(src_fps / target_fps)))
                pose, lh, rh = [], [], []
                fi = -1
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    fi += 1
                    if fi % stride:
                        continue
                    res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pose.append(to_array(res.pose_landmarks, N_POSE, 4))
                    lh.append(to_array(res.left_hand_landmarks, N_HAND, 3))
                    rh.append(to_array(res.right_hand_landmarks, N_HAND, 3))
                cap.release()
                if not pose:
                    failed += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write. savez_compressed builds its zip incrementally, so
                # a kill mid-write leaves a truncated file that still EXISTS and
                # would be skipped as done on the next run. The temp name must
                # already end in .npz or savez appends a second suffix.
                tmp = dest.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, pose=np.stack(pose),
                                    lh=np.stack(lh), rh=np.stack(rh))
                tmp.replace(dest)
                done += 1
            except Exception:  # noqa: BLE001
                failed += 1
            finally:
                if tmp_vid and os.path.exists(tmp_vid):
                    os.unlink(tmp_vid)          # the video is not the deliverable
    finally:
        holistic.close()
    return done, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--target-fps", type=float, default=15.0,
                    help="resample each clip to this rate before inference")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = json.loads(LIST.read_text())
    if args.limit:
        paths = paths[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    todo = [p for p in paths if not (OUT / word_of(p) / (Path(p).stem + ".npz")).exists()]
    print(f"{len(paths)} clips listed, {len(paths) - len(todo)} already extracted, "
          f"{len(todo)} to do, {args.workers} workers", flush=True)
    if not todo:
        return 0

    # Chunk the work rather than handing each worker one giant shard, so the
    # pool can RECYCLE workers. A MediaPipe process grows steadily -- measured
    # here, throughput fell 12 -> 5 clips/min over a few hours as the workers
    # bloated into swap on a 16 GB machine, and recovered the moment they were
    # restarted. maxtasksperchild caps that: a worker is retired after a few
    # chunks and its memory returns to the system.
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    with mp_proc.Pool(args.workers, maxtasksperchild=2) as pool:
        # Kill the pool if this process is asked to stop.
        #
        # Without this, the default SIGTERM handling kills the parent outright
        # and every worker is reparented to init and keeps running. Measured
        # here: restarting the extractor a few times over one day left 21
        # orphaned MediaPipe workers alive for 12+ hours, which drove the load
        # average to 104 and free memory to 81 MB on a 16 GB machine. The
        # extraction it was competing with was its own.
        def _stop(signum, _frame):
            pool.terminate()
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _stop)

        results = []
        for k, r in enumerate(pool.imap_unordered(
                worker, [(c, args.complexity, args.target_fps) for c in chunks]), 1):
            results.append(r)
            if k % 10 == 0:
                have = sum(x[0] + x[1] for x in results)
                print(f"  {have}/{len(todo)}", flush=True)

    done = sum(r[0] for r in results)
    skipped = sum(r[1] for r in results)
    failed = sum(r[2] for r in results)
    print(f"\n{done} new, {skipped} already had, {failed} failed "
          f"-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
