"""
Build unlabelled windows for self-supervised pretraining.

    .venv/bin/python train/preprocess_ssl.py [--include-asl]

Reads  data/islgov_landmarks/**/*.npz  (+ data/pretrain.npz with --include-asl)
Writes data/ssl.npz    X (N, 32, 65, 3), word (N,), words (W,)

The `word` array is NOT a training label. It exists so pretrain_ssl.py can form
contrastive positives: two clips of the SAME dictionary entry, signed by
different people, should land in the same place. That is signer invariance
taught directly, and it is the property the model most lacks. No classifier is
ever built over these 12,103 words -- with a median of one clip each, one could
not be.

Why self-supervision, and why this corpus
-----------------------------------------
The Government of India ISL Dictionary is 13,665 clips over 12,103 words, with a
MEDIAN OF ONE CLIP PER WORD. As supervised data that is nearly useless: a
12,103-class model cannot be trained on one example each, and importing such
classes manufactures words the model claims to know and cannot recognise
(ARCHITECTURE.md 2.2).

Self-supervision does not care. It needs no labels at all, so the median stops
mattering and 13,665 clips of *Indian* Sign Language -- the actual target
language, not borrowed ASL -- become usable in full. It is the largest
permissively-licensed ISL corpus that exists.

Two things this has to get right
--------------------------------
1. TRIMMING. These are dictionary entries, not clips: they average ~900 frames
   against INCLUDE's 61, and contain title cards and repeated demonstrations.
   Untrimmed, a model would spend most of its capacity on title cards. We cut to
   the span where a hand is actually visible -- a title card has no person in it,
   so hand presence removes it cleanly.

2. NOT OVER-WEIGHTING LONG CLIPS. A 1,700-frame entry would otherwise contribute
   50 windows while a 100-frame entry contributes 3, so the encoder would learn
   whichever words happen to have been filmed at length. MAX_WINDOWS caps it.

Geometry is features.isotropic + features.anchor, identical to the supervised
path, so the pretrained weights drop straight into build_model.

Deliberately NOT included: INCLUDE and CISLR. Pretraining on the same clips the
model is later tested against is transductive, and this repo has already been
burned once by evaluating a model on data it had seen (ARCHITECTURE.md 12). The
gain has to come from data the test set never touches.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "islgov_landmarks"
ASL = ROOT / "data" / "pretrain.npz"
OUT = ROOT / "data" / "ssl.npz"

POSE_KEEP = features.POSE_KEEP          # 23
WINDOW = features.SEQ_LEN               # 32
STRIDE = 16                             # 50% overlap
MAX_WINDOWS = 6                         # per clip; see the docstring
MIN_ACTIVE = 16
PAD = 3
# MEASURED, not assumed. 24 clips sampled straight from the repository: every
# one is 16:9 (1920x1080 and 1280x720 both appear, the aspect does not vary).
# This was 1.0 until the landmarks showed a nose/shoulder ratio of 0.938 against
# an anatomical 0.578 -- i.e. the whole SSL corpus would have been built with
# exactly the vertical stretch that cost 2.1% cross-corpus on INCLUDE.
ASPECT = 16 / 9


def active_span(npz) -> tuple[int, int] | None:
    present = (np.abs(npz["lh"]).sum(axis=(1, 2)) > 0) | \
              (np.abs(npz["rh"]).sum(axis=(1, 2)) > 0)
    idx = np.flatnonzero(present)
    if idx.size < MIN_ACTIVE:
        return None
    return max(int(idx[0]) - PAD, 0), min(int(idx[-1]) + PAD + 1, len(present))


def windows_of(pts: np.ndarray) -> list[np.ndarray]:
    """(T, 65, 3) -> up to MAX_WINDOWS anchored, resampled 32-frame windows."""
    t = len(pts)
    if t < WINDOW:
        return [features.resample(features.anchor(pts))]
    starts = list(range(0, t - WINDOW + 1, STRIDE))
    if len(starts) > MAX_WINDOWS:
        # spread the cap across the clip rather than taking the first N, so a
        # long entry contributes its whole range and not just its opening
        starts = [starts[i] for i in
                  np.linspace(0, len(starts) - 1, MAX_WINDOWS).astype(int)]
    out = []
    for s in starts:
        w = features.anchor(pts[s:s + WINDOW])
        if np.all(np.isfinite(w)):
            out.append(w.astype(np.float32))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-asl", action="store_true",
                    help="also fold in the MS-ASL + WLASL windows from pretrain.npz")
    args = ap.parse_args()

    rows: list[np.ndarray] = []
    row_word: list[int] = []
    word_id: dict[str, int] = {}
    files = sorted(SRC.rglob("*.npz"))
    print(f"ISL dictionary: {len(files)} clips extracted so far")
    short = bad = 0
    for i, path in enumerate(files):
        if i % 500 == 0 and i:
            print(f"  {i}/{len(files)}  ({len(rows)} windows)", flush=True)
        try:
            with np.load(path) as d:
                span = active_span(d)
                if span is None:
                    short += 1
                    continue
                lo, hi = span
                pts = np.concatenate([d["pose"][lo:hi, :POSE_KEEP, :3],
                                      d["lh"][lo:hi], d["rh"][lo:hi]],
                                     axis=1).astype(np.float64)
            w = windows_of(features.isotropic(pts, ASPECT))
            # the dictionary entry this clip demonstrates, from its directory
            wid = word_id.setdefault(path.parent.name.lower(), len(word_id))
            rows.extend(w)
            row_word.extend([wid] * len(w))
        except Exception:  # noqa: BLE001
            bad += 1

    print(f"  {len(rows)} windows  (dropped {short} too short, {bad} unusable)")
    if rows:
        r = features.check_isotropy(np.stack(rows[:400]), "ISL dictionary")
        print(f"  isotropy check: nose/shoulder {r:.3f} "
              f"(anatomical {features.ANATOMY_RATIO})")

    if args.include_asl and ASL.exists():
        d = np.load(ASL, allow_pickle=True)
        a, ay = d["X"], d["y"]
        print(f"MS-ASL + WLASL: {len(a)} windows")
        # Namespace ASL glosses away from ISL ones. "book" in ASL and "book" in
        # ISL are different signs, so treating them as one contrastive positive
        # would pull two unrelated movements together.
        off = len(word_id)
        rows.extend(a)
        row_word.extend((ay + off).tolist())
        for i in range(int(ay.max()) + 1):
            word_id[f"asl:{i}"] = off + i

    if not rows:
        print("nothing to write: let train/extract_islgov.py run further")
        return 1
    X = np.stack(rows).astype(np.float32)
    word = np.asarray(row_word, dtype=np.int32)
    words = np.array([w for w, _ in sorted(word_id.items(), key=lambda kv: kv[1])])
    multi = int((np.bincount(word, minlength=len(words)) >= 2).sum())
    print(f"\nX {X.shape}")
    print(f"{len(words)} distinct entries, {multi} with 2+ windows "
          f"(those supply cross-clip contrastive positives)")
    np.savez_compressed(OUT, X=X, word=word, words=words)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
