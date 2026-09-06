"""
Ingest NCERT's "<word> | ISL" clips: the medical vocabulary nothing else has.

    .venv-mp/bin/python train/fetch_ncert.py --list
    .venv-mp/bin/python train/fetch_ncert.py

Reads  the NCERT OFFICIAL channel, playlist "ISL Words Program"
Writes data/ncert_landmarks/<word>/<video_id>.npz   pose, lh, rh, aspect
       data/meta/ncert.json                          the catalogue

Why this source
---------------
NCERT is the Government of India's school curriculum body, and this series is
one short clip per word, signed to camera. It supplies the words that were the
standing gap in this project: body pain, please, help, water, blood, bandage,
injection, sorry, stop, food and nurse. Every one of those was checked for and
found absent in INCLUDE, CISLR, the Government ISL dictionary, MS-ASL, WLASL,
AUTSL, ISH News and ISH Shiksha. They are the words a patient needs most, which
is why the phrase board carried them instead of the recogniser.

Being NCERT also settles provenance. It is a public education release from a
government body, which is a cleaner footing than scraping a private channel.

The same one-clip-per-word limit applies
----------------------------------------
199 words, one clip each. A class with a single example cannot be both taught
and examined, so these do not become new classifier classes on their own. Two
honest uses:

  1. For words already in the 83, this is a NEW SIGNER on an unseen corpus,
     which is the lever that made CISLR worth +3.4 points.
  2. For words outside the 83, one clip is a reference embedding, usable for
     one-shot matching, and a seed that recording can extend.

Geometry
--------
These are landscape 1920x1080, aspect 1.778. ISH Shiksha is vertical 0.562.
Mixing the two without correction is exactly the bug that cost 2.1 points
across corpora, so aspect is measured per video and stored in the npz, and
features.isotropic divides it back out.
"""
import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ncert_landmarks"
META = ROOT / "data" / "meta" / "ncert.json"
CHANNEL = "https://www.youtube.com/channel/UCT0s92hGjqLX6p7qY9BBrSA"
PLAYLISTS = (
    "PLUgLcpnv1Yid-bfsAhvh36VwXRtxORopM",   # ISL Words Program      "<word> | ISL"
    "PLUgLcpnv1Yick8uBjTV9J6YCsNVrMsa-e",   # Basic ISL Course       "ISL <word>"
)
TARGET_FPS = 15.0
MAX_SECONDS = 60
TRIM_PAD = 2
MIN_ACTIVE = 8
N_POSE, N_HAND = 33, 21

# Two title formats, both one sign per clip:
#   "Please | ISL"  -> please        (ISL Words Program)
#   "ISL Injury"    -> injury        (Basic ISL Course)
SUFFIX = re.compile(r"^\s*([A-Za-z'&/,. -]{2,40}?)\s*\|\s*ISL\s*$", re.I)
PREFIX = re.compile(r"^\s*ISL\s+([A-Za-z'&/,. -]{2,40})\s*$", re.I)

# Lesson recordings live on the same channel and run 20 to 60 minutes. They are
# whole school periods, not single signs, so a title that looks like one is
# rejected outright rather than relying on the duration cap alone.
LESSON = re.compile(r"NCERT|Grade\s*:|Subject\s*:|Chapter|Live\s*Session|Part\s*[-–]|CLASS-", re.I)


def word_of(title: str) -> str | None:
    t = (title or "").strip()
    if LESSON.search(t):
        return None
    for rx in (SUFFIX, PREFIX):
        m = rx.match(t)
        if m:
            w = re.sub(r"\s+", " ", m.group(1)).strip().lower().strip(".,")
            return w or None
    return None


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name)[:80]


def catalogue(limit: int) -> list[dict]:
    import yt_dlp
    opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True,
            "socket_timeout": 60, "playlistend": 20000}
    items: dict[str, dict] = {}
    sources = [f"https://www.youtube.com/playlist?list={p}" for p in PLAYLISTS]
    sources.append(f"{CHANNEL}/videos")
    for src in sources:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(src, download=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {src.split('/')[-1][:40]}: {exc}")
            continue
        for e in (info or {}).get("entries", []):
            if not e or not e.get("id"):
                continue
            w = word_of(e.get("title") or "")
            if not w:
                continue
            d = e.get("duration") or 0
            if d and d > MAX_SECONDS:
                continue
            items[e["id"]] = {"id": e["id"], "title": e.get("title"),
                              "word": w, "duration": d}
    out = sorted(items.values(), key=lambda i: i["word"])
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(out, indent=1))
    return out[:limit] if limit else out


def to_array(landmarks, n: int, dims: int) -> np.ndarray:
    out = np.zeros((n, dims), dtype=np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark[:n]):
        out[i, 0], out[i, 1], out[i, 2] = lm.x, lm.y, lm.z
        if dims == 4:
            out[i, 3] = getattr(lm, "visibility", 0.0)
    return out


def process(item: dict, work: Path, holistic) -> bool:
    import cv2
    import yt_dlp

    dest = OUT / safe(item["word"]) / f"{item['id']}.npz"
    if dest.exists():
        return False
    opts = {"quiet": True, "no_warnings": True, "socket_timeout": 60,
            "format": "bestvideo[height<=720][ext=mp4]/best[height<=720]/best",
            "outtmpl": str(work / f"{item['id']}.%(ext)s")}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={item['id']}"])
    except Exception:
        return False
    files = list(work.glob(f"{item['id']}.*"))
    if not files:
        return False
    path = files[0]
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        w, h = int(cap.get(3)), int(cap.get(4))
        aspect = (w / h) if h else 16 / 9
        src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        stride = max(1, int(round(src_fps / TARGET_FPS)))
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
        if len(pose) < 8:
            return False

        LH, RH = np.stack(lh), np.stack(rh)
        present = (np.abs(LH).sum(axis=(1, 2)) > 0) | (np.abs(RH).sum(axis=(1, 2)) > 0)
        idx = np.flatnonzero(present)
        if idx.size < MIN_ACTIVE:
            return False
        lo = max(int(idx[0]) - TRIM_PAD, 0)
        hi = min(int(idx[-1]) + TRIM_PAD + 1, len(present))

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pose=np.stack(pose)[lo:hi], lh=LH[lo:hi],
                            rh=RH[lo:hi], aspect=np.float32(aspect))
        tmp.replace(dest)
        return True
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    items = catalogue(args.limit)
    words = sorted({i["word"] for i in items})
    print(f"{len(items)} clips, {len(words)} distinct words")
    if args.list:
        for w in words:
            print(f"  {w}")
        return 0

    import mediapipe as mp
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ncert-"))
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=args.complexity,
        refine_face_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    done = skipped = 0
    try:
        for i, item in enumerate(items, 1):
            if process(item, work, holistic):
                done += 1
            else:
                skipped += 1
            if i % 25 == 0:
                print(f"  {i}/{len(items)}  kept {done}  skipped {skipped}", flush=True)
    finally:
        holistic.close()
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{done} clips written to {OUT.relative_to(ROOT)}  ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
