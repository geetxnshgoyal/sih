"""
Expand the sign playback library from 96 words to as many as we have clips for.

    .venv-tf/bin/python train/export_signs.py --limit 400

Reads  data/{ncert,shiksha,islgov}_landmarks/<word>/*.npz
Writes app/public/model/_signs.json   word -> 24 frames x 65 points x [x, y]

The gap this closes
-------------------
The clinician types "water" and the patient view says "No matching sign
recording for this message". Not because the sign is unknown, but because
_signs.json held 96 words and water was not one of them. The reverse direction
was limited by a hand-built library while 14,500 clips of the same coordinates
sat in data/ unused.

Playback needs exactly what recognition needs
---------------------------------------------
SignPlayer draws a shoulder-anchored, shoulder-scaled 65-point skeleton, and
auto-fits it to the canvas. That is the output of features.anchor(isotropic(.))
which every corpus here already passes through. So a playback entry is the same
pipeline as a training example, stopped one step earlier.

The renderer reads only x and y, never z (verified in SignPlayer.tsx), so z is
written as 0. The shape stays 65x3 to match the SignFrame type, and a column of
zeros costs almost nothing once gzipped.

Source order is deliberate
--------------------------
NCERT and ISH Shiksha first: those are four-second clips of one sign, framed
for teaching. The Government dictionary averages a minute and is largely title
cards and repetition, so its clips are trimmed to the hand-visible span and used
only where nothing better exists. A bad playback is worse than none, because the
patient has no way to tell a bad rendering from a sign they do not know.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import clip_io  # noqa: E402
import features  # noqa: E402

OUT = ROOT / "app" / "public" / "model" / "_signs.json"
SOURCES = [("ncert", ROOT / "data" / "ncert_landmarks"),
           ("shiksha", ROOT / "data" / "shiksha_landmarks"),
           ("islgov", ROOT / "data" / "islgov_landmarks")]
FRAMES = 24          # what the existing library uses, at 14 fps
MIN_ACTIVE, PAD = 8, 2
DECIMALS = 2         # the canvas is 420x380 and auto-fits; 2 is well past visible

# Words worth carrying, beyond whatever the library already has. Clinical first.
WANTED = """
pain body pain injury hurt wound blood bandage injection medicine tablet
doctor nurse hospital emergency ambulance operation patient sick fever
headache stomach dizzy weak breathe heart head hand eye ear back throat tooth
please help sorry thank you yes no stop wait more less
water food eat drink hungry thirsty sleep bathroom
name time today tomorrow yesterday morning evening night day week month year
mother father sister brother family child baby man woman friend
house school work money phone car bus train ticket station airport hotel
police road right up good bad big small hot cold new old open close
understand know give take come sit stand walk happy sad afraid deaf
"""
COMPOUND = ("body pain", "thank you")


def norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()


def wanted_words() -> list[str]:
    line = " ".join(WANTED.split("\n"))
    out = []
    for c in COMPOUND:
        if c in line:
            out.append(c); line = line.replace(c, " ")
    out += line.split()
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w); uniq.append(w)
    return uniq


def playback_frames(path: Path) -> list | None:
    """One landmark npz -> FRAMES x 65 x 3, anchored, z zeroed."""
    pts, aspect = clip_io.load_points(path, features.POSE_KEEP, MIN_ACTIVE, PAD)
    if pts is None:
        return None
    seq = features.anchor(features.isotropic(pts, aspect))
    if not np.all(np.isfinite(seq)):
        return None
    t = seq.shape[0]
    take = np.round(np.arange(FRAMES) * (t - 1) / max(FRAMES - 1, 1)).astype(int)
    seq = seq[np.minimum(take, t - 1)]
    seq = np.round(seq[..., :2], DECIMALS)
    # x and y only. SignPlayer reads p[0] and p[1] and never p[2], so the depth
    # column was a third of the file doing nothing. A hand that was never
    # detected stays exactly zero, which is how the renderer knows not to draw
    # a collapsed claw where there was no hand.
    return [[[float(v) for v in p] for p in f] for f in seq]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap words added")
    ap.add_argument("--all", action="store_true", help="every word available")
    args = ap.parse_args()

    lib = json.loads(OUT.read_text()) if OUT.exists() else {}
    before = len(lib)
    have = {norm(k) for k in lib}

    index: dict[str, tuple[str, Path]] = {}
    for sname, d in SOURCES:
        if not d.is_dir():
            continue
        for wd in sorted(d.iterdir()):
            if not wd.is_dir():
                continue
            w = norm(wd.name)
            if not w or w in index:
                continue
            fs = sorted(wd.glob("*.npz"))
            if fs:
                index[w] = (sname, fs[0])

    targets = sorted(index) if args.all else [w for w in wanted_words() if w in index]
    targets = [w for w in targets if w not in have]
    if args.limit:
        targets = targets[:args.limit]
    print(f"library has {before} words; {len(index)} words available in landmarks")
    print(f"adding {len(targets)}\n")

    added, failed, by_src = 0, [], {}
    for w in targets:
        sname, path = index[w]
        fr = playback_frames(path)
        if fr is None:
            failed.append(w); continue
        lib[w] = fr
        by_src[sname] = by_src.get(sname, 0) + 1
        added += 1

    OUT.write_text(json.dumps(lib, separators=(",", ":")))
    mb = OUT.stat().st_size / 1e6
    print(f"  added {added} from {by_src}")
    print(f"  {before} -> {len(lib)} words, {mb:.2f} MB")
    if failed:
        print(f"  {len(failed)} failed the hand-presence test: {', '.join(failed[:12])}")
    missing = [w for w in wanted_words() if w not in index and norm(w) not in have]
    if missing:
        print(f"\n  {len(missing)} wanted words have no clip anywhere: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
