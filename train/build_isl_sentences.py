"""
Build a sentence-aligned Indian Sign Language corpus from Deaf-run news.

    .venv-mp/bin/python train/build_isl_sentences.py --limit 50 --workers 3

Reads  the ISH News channel
Writes data/isl_sentences/<video_id>/<cue>.npz   landmarks + the English sentence
       data/isl_sentences/index.jsonl            one row per pair

Why this changes what is possible
---------------------------------
Everything built so far recognises ISOLATED SIGNS: one sign per 32-frame window,
83 of them. It cannot read a sentence, and no amount of retraining would let it,
because sentence translation is a different task needing different data.

That data did not appear to exist for ISL. It does. ISH News publishes daily
bulletins signed by Deaf presenters, and the videos carry MANUAL English
subtitles -- human written, not auto-generated -- with precise timestamps. Each
subtitle cue is therefore a (signing segment, English sentence) pair, which is
exactly the supervision continuous sign language translation needs.

Measured on the channel listing: 7,935 videos, 7,615 of them under 15 minutes,
576 hours of footage, ~48 cues per 7-minute bulletin. That projects to roughly
237,000 sentence pairs, against 8,257 in PHOENIX14T and 20,654 in CSL-Daily,
the two benchmarks this field is built on.

What this does NOT do
---------------------
It does not make the current model read sentences. A classifier over 83 labels
cannot emit text; that needs an encoder-decoder trained on these pairs, which is
a separate build. What this script does is create the dataset that build would
need, which is the part that did not exist.

Standing of the data
--------------------
The videos and the subtitles belong to ISH News, a Deaf-run organisation. This
keeps landmarks and text, never video, and no face or identity can be
reconstructed from 65 points. Ask them before publishing anything derived from
it: their mission is access, they are the obvious partner, and doing it behind
their back would be both wrong and a worse story.

Aspect ratio is measured per video and stored beside the landmarks, never
assumed. That mistake has been made twice already.
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
OUT = ROOT / "data" / "isl_sentences"
INDEX = OUT / "index.jsonl"
CHANNEL = "https://www.youtube.com/@ISHNews/videos"
TARGET_FPS = 15.0
MAX_MINUTES = 15
MIN_CUE_S, MAX_CUE_S = 1.0, 20.0
N_POSE, N_HAND = 33, 21

CUE = re.compile(r"(\d\d:\d\d:\d\d\.\d+) --> (\d\d:\d\d:\d\d\.\d+)[^\n]*\n(.*?)(?=\n\n|\Z)", re.S)
TAGS = re.compile(r"<[^>]+>")


def secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(text: str) -> list[tuple[float, float, str]]:
    out = []
    for a, b, body in CUE.findall(text):
        line = TAGS.sub("", body).replace("\n", " ").strip()
        line = re.sub(r"\s+", " ", line)
        if not line:
            continue
        t0, t1 = secs(a), secs(b)
        if MIN_CUE_S <= t1 - t0 <= MAX_CUE_S:
            out.append((t0, t1, line))
    return out


def to_array(landmarks, n: int, dims: int) -> np.ndarray:
    out = np.zeros((n, dims), dtype=np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark[:n]):
        out[i, 0], out[i, 1], out[i, 2] = lm.x, lm.y, lm.z
        if dims == 4:
            out[i, 3] = getattr(lm, "visibility", 0.0)
    return out


def process(vid: str, title: str, work: Path, complexity: int) -> int:
    """One bulletin -> one .npz per subtitle cue. Returns pairs written."""
    import cv2
    import mediapipe as mp
    import yt_dlp

    dest = OUT / vid
    if (dest / "_done").exists():
        return 0
    dest.mkdir(parents=True, exist_ok=True)

    # Subtitles first: no subtitles, no pairs, so do not spend a video download.
    sub_opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "writesubtitles": True, "subtitleslangs": ["en-IN", "en"],
                "subtitlesformat": "vtt", "outtmpl": str(work / f"{vid}.%(ext)s")}
    try:
        with yt_dlp.YoutubeDL(sub_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception:
        return 0
    vtts = sorted(work.glob(f"{vid}*.vtt"))
    if not vtts:
        shutil.rmtree(dest, ignore_errors=True)
        return 0
    cues = parse_vtt(vtts[0].read_text(encoding="utf8", errors="ignore"))
    for f in vtts:
        f.unlink(missing_ok=True)
    if not cues:
        shutil.rmtree(dest, ignore_errors=True)
        return 0

    v_opts = {"quiet": True, "no_warnings": True,
              "format": "bestvideo[height<=480][ext=mp4]/best[height<=480]/best",
              "outtmpl": str(work / f"{vid}.%(ext)s")}
    try:
        with yt_dlp.YoutubeDL(v_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception:
        return 0
    files = [f for f in work.glob(f"{vid}.*") if f.suffix != ".vtt"]
    if not files:
        return 0
    path = files[0]

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=complexity,
        refine_face_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    written = 0
    rows = []
    try:
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        w, h = int(cap.get(3)), int(cap.get(4))
        aspect = (w / h) if h else 16 / 9
        stride = max(1, int(round(src_fps / TARGET_FPS)))
        # One pass over the video, bucketing frames into whichever cue spans
        # them. Seeking per cue would re-decode the file dozens of times.
        buckets: dict[int, list] = {i: [] for i in range(len(cues))}
        fi = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
            if fi % stride:
                continue
            t = fi / src_fps
            hit = next((i for i, (a, b, _) in enumerate(cues) if a <= t <= b), None)
            if hit is None:
                continue
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buckets[hit].append((to_array(res.pose_landmarks, N_POSE, 4),
                                 to_array(res.left_hand_landmarks, N_HAND, 3),
                                 to_array(res.right_hand_landmarks, N_HAND, 3)))
        cap.release()

        for i, (a, b, text) in enumerate(cues):
            frames = buckets[i]
            if len(frames) < 8:
                continue
            f = dest / f"{i:04d}.npz"
            tmp = f.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp,
                pose=np.stack([x[0] for x in frames]),
                lh=np.stack([x[1] for x in frames]),
                rh=np.stack([x[2] for x in frames]),
                aspect=np.float32(aspect))
            tmp.replace(f)
            rows.append({"video": vid, "cue": i, "start": a, "end": b,
                         "text": text, "frames": len(frames),
                         "aspect": round(aspect, 4), "title": title})
            written += 1
    finally:
        holistic.close()
        path.unlink(missing_ok=True)

    if rows:
        with open(INDEX, "a", encoding="utf8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (dest / "_done").touch()
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()

    import yt_dlp
    OUT.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": "in_playlist",
                           "skip_download": True,
                           "playlistend": args.limit * 3}) as ydl:
        info = ydl.extract_info(CHANNEL, download=False)
    vids = [e for e in (info or {}).get("entries", [])
            if e and e.get("id") and 0 < (e.get("duration") or 0) <= MAX_MINUTES * 60]
    print(f"{len(vids)} bulletins listed, taking {args.limit}\n")

    work = Path(tempfile.mkdtemp(prefix="isl-"))
    total = done = 0
    try:
        for v in vids[: args.limit]:
            n = process(v["id"], (v.get("title") or "")[:110], work, args.complexity)
            if n:
                done += 1
                total += n
                print(f"  [{done}] {v['id']}  {n:>3} pairs  {v.get('title','')[:52]}",
                      flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{total} sentence pairs from {done} bulletins -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
