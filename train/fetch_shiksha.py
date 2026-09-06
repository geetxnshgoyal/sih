"""
Ingest ISH Shiksha's word-of-the-day clips: the vocabulary nothing else has.

    .venv-mp/bin/python train/fetch_shiksha.py --list
    .venv-mp/bin/python train/fetch_shiksha.py --limit 500

Reads  the ISH Shiksha channel
Writes data/shiksha_landmarks/<word>/<video_id>.npz   pose, lh, rh, aspect
       data/meta/shiksha.json                          the catalogue

Why this channel specifically
----------------------------
The shipping model cannot sign pain, water, help, yes or no. Those are absent
from INCLUDE, from CISLR and from the Government of India dictionary, which is
why the phrase board carries them instead. ISH Shiksha is the education arm of
the same Deaf-run organisation as ISH News, and it publishes one short clip per
word. It supplies water, doctor, hospital, sick, fever, cold, thank you, sorry
and name, which is roughly half the gap.

It does NOT supply pain, yes, no or please. Those still need recording, and no
public source we have found has them.

Better than the government dictionary in two ways
-------------------------------------------------
FOCUS.   These are ~4 seconds: one sign, demonstrated, nothing else. The
         government clips average around a minute and are mostly title cards and
         repetition, which is why they needed aggressive trimming.

A NEW SIGNER.  For the ~415 words that overlap corpora we already have, each
         clip is a different person signing them. That is the same lever that
         made CISLR worth +3.4 points, and it is worth more than the new words.

Still one clip per word, so these cannot form new classes on their own: a class
with one example cannot be both taught and examined. They add signers to
existing classes, and they seed a vocabulary that recording can extend.

These are SHORTS: 720x1280, aspect 0.562, vertical. That is the opposite of
every other corpus here and would have been silently destructive before
features.isotropic existed. Aspect is measured per video and stored in the npz.
"""
import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "shiksha_landmarks"
META = ROOT / "data" / "meta" / "shiksha.json"
CHANNEL = "https://www.youtube.com/channel/UCx6PDGJYiE3PXuExsjmZudA"
TARGET_FPS = 15.0
MAX_SECONDS = 90
TRIM_PAD = 2        # frames kept either side of the signing
MIN_ACTIVE = 8      # fewer visible-hand frames than this is not a usable sign
N_POSE, N_HAND = 33, 21

# "Water - Indian Sign Language | ISH Shiksha" -> water
TITLE = re.compile(r"^\s*([A-Za-z'&/ ]{2,40}?)\s*[-|–—]\s*Indian Sign Language", re.I)
ALT = re.compile(r"^\s*([A-Za-z'&/ ]{2,40}?)\s*\|\s*Indian Sign Language", re.I)


def word_of(title: str) -> str | None:
    for rx in (TITLE, ALT):
        m = rx.match(title)
        if m:
            w = re.sub(r"\s+", " ", m.group(1)).strip().lower()
            # a few titles teach several signs at once; those are kept but named
            # for the whole group rather than split, since the clip is not
            # segmented and guessing boundaries would invent data
            return w or None
    return None


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name)[:80]


def catalogue(limit: int) -> list[dict]:
    import yt_dlp
    opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True}
    items: dict[str, dict] = {}
    for feed in ("shorts", "videos"):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"{CHANNEL}/{feed}", download=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {feed}: {exc}")
            continue
        for e in (info or {}).get("entries", []):
            if not e or not e.get("id"):
                continue
            t = e.get("title") or ""
            w = word_of(t)
            if not w:
                continue
            d = e.get("duration") or 0
            if d and d > MAX_SECONDS:
                continue
            items[e["id"]] = {"id": e["id"], "title": t, "word": w, "duration": d}
    out = list(items.values())
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
    opts = {"quiet": True, "no_warnings": True,
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
        aspect = (w / h) if h else 9 / 16
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

        # Trim to the span where a hand is visible.
        #
        # Every clip on this channel ends with a branding card and no person in
        # it, so 65% of the raw frames contain nothing to learn from: measured
        # hand presence is 0.32 untrimmed and 0.79 trimmed, which is exactly
        # what CISLR did. Storing the outro would mean a 32-frame window landing
        # mostly on a logo, and the model learning the logo.
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
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()

    items = catalogue(args.limit)
    words = sorted({i["word"] for i in items})
    print(f"{len(items)} clips, {len(words)} distinct words")
    if args.list:
        for w in words[:60]:
            print(f"  {w}")
        print(f"  ... ({len(words)} total, catalogued in {META.relative_to(ROOT)})")
        return 0

    import mediapipe as mp
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="shiksha-"))
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
                print(f"  {i}/{len(items)}  ({done} new, {skipped} skipped)", flush=True)
    finally:
        holistic.close()
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{done} clips extracted, {skipped} skipped -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
