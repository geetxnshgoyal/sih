"""
Ingest Deaf-run ISL news as unlabelled pretraining data.

    .venv-mp/bin/python train/fetch_ishnews.py --list       # what is available
    .venv-mp/bin/python train/fetch_ishnews.py --limit 20   # fetch + extract

Runs in .venv-mp, not .venv: this needs MediaPipe's legacy `solutions` API, which
exists in the pinned 0.10.14 and was removed in 1.x. Running it under .venv gets
`module 'mediapipe' has no attribute 'solutions'` after the download has already
happened.

Reads  a YouTube channel of Indian Sign Language news
Writes data/ishnews_landmarks/<video_id>/<start>.npz   pose, lh, rh

Why news, and why unlabelled
----------------------------
Everything the model has trained on is CITATION FORM: a dictionary entry or a
studio recording of one sign in isolation. Nobody signs like that. Connected
signing has transitions, coarticulation, rhythm and a signer who is talking
rather than demonstrating, and none of it appears in INCLUDE, CISLR or the
dictionary.

ISH News is Deaf-run, daily, native ISL. As SUPERVISED data it is unusable:
continuous signing with no word boundaries, no glosses, and few presenters. As
SELF-SUPERVISED data none of that matters, and the naturalness is the whole
point. The masked-reconstruction and contrastive objectives in pretrain_ssl.py
need no labels at all.

Standing of the data
--------------------
The videos belong to their creators. This extracts LANDMARKS: 65 points per
frame, from which the video cannot be reconstructed and no face is recoverable.
That is a much more defensible artifact to hold than video, and it is what the
model needs. It is still someone else's work: ISH News is a Deaf-run
organisation with an accessibility mission, and asking them is both the right
thing and likely to be answered yes. Do that before publishing anything derived
from it.

Aspect ratio is MEASURED per video, never assumed. Assuming it is how the same
bug landed twice already, most recently on the ISL dictionary, which was taken
for square and is 16:9 (ARCHITECTURE.md 5.1, features.check_isotropy).
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ishnews_landmarks"
META = ROOT / "data" / "meta" / "ishnews.json"

# ISH News: India Signing Hands, a Deaf-run ISL news channel.
CHANNEL = "https://www.youtube.com/@ISHNews/videos"

WINDOW_S = 20          # seconds of video per stored chunk
TARGET_FPS = 15.0      # matches extract_islgov.py so windows mean the same thing
MAX_MINUTES = 12       # skip anything longer; live streams are not bulletins
N_POSE, N_HAND = 33, 21


def list_videos(limit: int) -> list[dict]:
    """Channel listing via yt-dlp, metadata only."""
    import yt_dlp
    opts = {"quiet": True, "extract_flat": "in_playlist", "playlistend": limit,
            "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(CHANNEL, download=False)
    out = []
    for e in (info or {}).get("entries", []):
        if not e or not e.get("id"):
            continue
        dur = e.get("duration") or 0
        out.append({"id": e["id"], "title": (e.get("title") or "")[:90],
                    "duration": dur})
    return out


def download(vid: str, dest: Path) -> Path | None:
    """Fetch one video at modest resolution: we need landmarks, not pixels."""
    import yt_dlp
    opts = {
        "quiet": True, "no_warnings": True,
        # 480p is plenty for MediaPipe and a fraction of the bytes
        "format": "bestvideo[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl": str(dest / f"{vid}.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {vid}: {exc}")
        return None
    files = list(dest.glob(f"{vid}.*"))
    return files[0] if files else None


def to_array(landmarks, n: int, dims: int) -> np.ndarray:
    out = np.zeros((n, dims), dtype=np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark[:n]):
        out[i, 0], out[i, 1], out[i, 2] = lm.x, lm.y, lm.z
        if dims == 4:
            out[i, 3] = getattr(lm, "visibility", 0.0)
    return out


def extract(path: Path, vid: str, complexity: int) -> tuple[int, float]:
    """Whole video -> WINDOW_S chunks of landmarks. Returns (chunks, aspect)."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0.0
    w, h = int(cap.get(3)), int(cap.get(4))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    aspect = (w / h) if h else 16 / 9
    stride = max(1, int(round(src_fps / TARGET_FPS)))
    per_window = int(WINDOW_S * TARGET_FPS)

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=complexity,
        refine_face_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    dest = OUT / vid
    dest.mkdir(parents=True, exist_ok=True)
    pose, lh, rh = [], [], []
    fi, chunks = -1, 0

    def flush(idx: int) -> int:
        if len(pose) < per_window // 2:
            return 0
        # Aspect travels WITH the landmarks. Anything that guesses it later gets
        # the geometry wrong, which is silent and expensive.
        f = dest / f"{idx:05d}.npz"
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pose=np.stack(pose), lh=np.stack(lh),
                            rh=np.stack(rh), aspect=np.float32(aspect))
        tmp.replace(f)
        return 1

    try:
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
            if len(pose) >= per_window:
                chunks += flush(chunks)
                pose, lh, rh = [], [], []
        chunks += flush(chunks)
    finally:
        cap.release()
        holistic.close()
    return chunks, aspect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()

    vids = list_videos(max(args.limit, 30))
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(vids, indent=1))
    usable = [v for v in vids if 0 < v["duration"] <= MAX_MINUTES * 60]
    print(f"{len(vids)} listed, {len(usable)} within {MAX_MINUTES} min")
    if args.list:
        for v in usable[:20]:
            print(f"  {v['id']}  {v['duration'] // 60:>3}m  {v['title']}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ish-"))
    done = total = 0
    try:
        for v in usable[: args.limit]:
            if (OUT / v["id"]).exists():
                continue
            print(f"[{done + 1}/{args.limit}] {v['id']}  {v['title']}", flush=True)
            f = download(v["id"], work)
            if not f:
                continue
            n, aspect = extract(f, v["id"], args.complexity)
            f.unlink(missing_ok=True)      # the video is not the deliverable
            print(f"    {n} windows, aspect {aspect:.3f}", flush=True)
            total += n
            done += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{done} videos, {total} windows -> {OUT.relative_to(ROOT)}")
    print(f"  {total * WINDOW_S / 60:.0f} minutes of native connected ISL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
