"""
Turn the sentence-aligned corpus into tensors a translator can train on.

    .venv/bin/python train/preprocess_sentences.py

Reads  data/isl_sentences/index.jsonl + the per-cue .npz
Writes data/translate.npz    src (N, T, 195), src_len, tgt (N, L), vocab, meta

Why this is a different shape from everything else here
-------------------------------------------------------
Every dataset built so far is (clip -> one label). This is (clip -> a sentence),
which changes three things:

  LENGTH      Isolated signs were resampled to a fixed 32 frames because one
              sign is one gesture. A sentence is many, and squashing an 18-word
              utterance and a 2-word one into the same 32 frames destroys the
              thing a decoder has to read. Sequences are padded to MAX_FRAMES
              and their true length is kept, so attention can mask the padding.

  ORDER       A classifier may pool over time and throw the order away. A
              translator cannot: word order is the output.

  TARGET      Text, not a class index. Word-level with a frequency floor, so
              rare proper nouns collapse to <unk> rather than each buying a
              softmax row they have one example to learn from.

Geometry is the same as everywhere else: features.isotropic with the aspect
STORED beside the landmarks, then features.anchor. No resample, deliberately.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "isl_sentences"
INDEX = SRC / "index.jsonl"
OUT = ROOT / "data" / "translate.npz"

MAX_FRAMES = 160        # 15 fps, so ~10.7s. Median cue is 98 frames.
MAX_TOKENS = 28         # median sentence is 12 words
MIN_FREQ = 2            # a word seen once cannot be learned; it becomes <unk>
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(s: str) -> list[str]:
    out, cur = [], []
    for ch in s.lower():
        if ch.isalnum() or ch == "'":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur)); cur = []
            if ch in ".,?!":
                out.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def main() -> int:
    if not INDEX.exists():
        print(f"missing {INDEX.relative_to(ROOT)} — run train/build_isl_sentences.py")
        return 1
    rows = [json.loads(l) for l in INDEX.read_text(encoding="utf8").splitlines() if l.strip()]
    print(f"{len(rows)} pairs from {len({r['video'] for r in rows})} bulletins")

    freq = Counter(t for r in rows for t in tokenize(r["text"]))
    vocab = SPECIALS + [w for w, c in freq.most_common() if c >= MIN_FREQ]
    idx = {w: i for i, w in enumerate(vocab)}
    cover = sum(c for w, c in freq.items() if c >= MIN_FREQ) / max(sum(freq.values()), 1)
    print(f"vocabulary: {len(vocab)} words (seen >={MIN_FREQ}x), "
          f"covering {cover*100:.1f}% of tokens")

    src = np.zeros((len(rows), MAX_FRAMES, features.N_POINTS * features.N_DIMS), np.float32)
    src_len = np.zeros(len(rows), np.int32)
    tgt = np.zeros((len(rows), MAX_TOKENS), np.int32)
    keep = 0
    short = bad = 0
    for r in rows:
        f = SRC / r["video"] / f"{r['cue']:04d}.npz"
        if not f.exists():
            bad += 1
            continue
        try:
            with np.load(f) as z:
                aspect = float(z["aspect"]) if "aspect" in z else 16 / 9
                pts = np.concatenate([z["pose"][:, :features.POSE_KEEP, :3],
                                      z["lh"], z["rh"]], axis=1).astype(np.float64)
        except Exception:
            bad += 1
            continue
        if len(pts) < 8:
            short += 1
            continue
        a = features.anchor(features.isotropic(pts, aspect))
        if not np.all(np.isfinite(a)):
            bad += 1
            continue
        # Truncate rather than resample: a translator needs real timing, and a
        # cue longer than MAX_FRAMES is rare enough that losing its tail beats
        # compressing every sequence onto one clock.
        a = a[:MAX_FRAMES]
        flat = a.reshape(len(a), -1).astype(np.float32)
        # standardise per clip, as features.extract does
        m, s = flat.mean(), max(float(flat.std()), 1e-6)
        src[keep, :len(flat)] = (flat - m) / s
        src_len[keep] = len(flat)

        toks = [BOS] + [idx.get(t, UNK) for t in tokenize(r["text"])][: MAX_TOKENS - 2] + [EOS]
        tgt[keep, :len(toks)] = toks
        keep += 1

    src, src_len, tgt = src[:keep], src_len[:keep], tgt[:keep]
    print(f"kept {keep}  (dropped {short} too short, {bad} unusable)")
    print(f"  src {src.shape}   frames: median {int(np.median(src_len))} "
          f"max {int(src_len.max())}")
    unk = float((tgt == UNK).sum() / max((tgt != PAD).sum(), 1))
    print(f"  tgt {tgt.shape}   <unk> rate {unk*100:.1f}%")

    # Split by VIDEO, never by pair: two cues from one bulletin share a signer,
    # a room and a camera, so a random split would leak all three.
    vids = sorted({r["video"] for r in rows})
    rng = np.random.default_rng(0)
    rng.shuffle(vids)
    cut = max(int(0.85 * len(vids)), 1)
    train_v = set(vids[:cut])
    row_v = [r["video"] for r in rows if (SRC / r["video"] / f"{r['cue']:04d}.npz").exists()][:keep]
    split = np.array([0 if v in train_v else 1 for v in row_v], np.int32)
    print(f"  split: {int((split==0).sum())} train / {int((split==1).sum())} val "
          f"across {len(vids)} bulletins, held out by video")

    np.savez_compressed(OUT, src=src, src_len=src_len, tgt=tgt,
                        split=split, vocab=np.array(vocab))
    print(f"\nwrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
