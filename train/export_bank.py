"""
Export a reference bank so the app can recognise words the classifier cannot.

    .venv-tf/bin/python train/export_bank.py

Reads  run/bank83.npz          embeddings of every dictionary clip
Writes app/public/model/_bank.json

What this is for
----------------
The shipped model is a closed-set classifier over 83 words. Pain, please, help,
water, blood, bandage and injection are not among them and cannot be added: the
only sources that have those words have one clip each, and a class with a
single example cannot be both taught and examined.

Retrieval does not need fifteen clips. Embed one reference clip per word, embed
the live sign with the SAME model, and take the nearest neighbour by cosine
similarity. Vocabulary then costs one clip per word.

Why the bank is curated and not the whole dictionary
----------------------------------------------------
Measured leave-one-clip-out, cross-source:

    13,648-word bank    20.8% top-1    30.2% top-5
       105-word bank    46.5% top-1    64.5% top-5

Bank size dominates. Searching all 13,648 words would be a worse product than
searching the 120 a patient actually needs, so the bank is the words that carry
clinical or travel weight, and nothing else. Adding a word is a deliberate act.

Honest limits, recorded because they will not be obvious from the UI
--------------------------------------------------------------------
1. Every reference is a studio clip from a government or education channel. A
   live webcam in a hospital corridor is a FOURTH domain that none of the
   measurements above cover, and it is likely worse than they suggest.
2. Cross-source scores here lean on NCERT queries against Government dictionary
   references. Both are official productions and may share studio conditions.
3. Words with one clip in total (pain, yes, no, fever, emergency, injury) are in
   the bank but could not be tested at all. They are retrievable in principle
   and unmeasured in practice.

Which is why the app labels these as dictionary matches with a shortlist, never
as a confident answer, and why the phrase board remains the reliable path.
"""
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "run" / "bank83.npz"
OUT = ROOT / "app" / "public" / "model" / "_bank.json"

# Deliberate vocabulary. Medical first, then the words needed to hold any
# conversation at all. Kept to roughly 120 because bank size dominates accuracy.
WANTED = """
pain body pain injury hurt wound blood bandage injection medicine tablet
doctor nurse hospital emergency ambulance operation patient treatment
sick fever headache stomach dizzy weak breathe heart
head hand leg eye ear back throat tooth bone skin
please help sorry thank you yes no stop wait more less
water food eat drink hungry thirsty tired sleep toilet bathroom
name time today tomorrow yesterday morning evening night day week month year
mother father sister brother family child baby man woman friend
home house school work money phone car bus train ticket station airport hotel
police station road left right up down near far
good bad big small hot cold new old open close
understand know want need give take come sit stand walk fall
happy sad afraid angry deaf hearing sign language interpreter
"""


def main() -> int:
    if not BANK.exists():
        print(f"missing {BANK.relative_to(ROOT)}, run train/build_bank.py first")
        return 1
    z = np.load(BANK, allow_pickle=True)
    E, W, S = z["emb"].astype(np.float64), z["word"], z["source"]

    by = defaultdict(list)
    for i, w in enumerate(W):
        by[w].append(i)

    wanted, seen = [], set()
    for tok in WANTED.split("\n"):
        tok = tok.strip()
        if not tok:
            continue
        # multi-word entries are written on their own conceptually; split on
        # single spaces only where the phrase is not a known compound
        for w in _phrases(tok):
            if w not in seen:
                seen.add(w); wanted.append(w)

    words, vecs, srcs, nref = [], [], [], []
    missing = []
    for w in wanted:
        if w not in by:
            missing.append(w)
            continue
        ix = by[w]
        # prototype: mean of every reference available for the word, renormalised.
        # More references from more corpora is strictly more information about
        # what the sign looks like across signers, which is the axis that hurts.
        v = E[ix].mean(axis=0)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-9:
            missing.append(w)
            continue
        words.append(w); vecs.append(v / n)
        srcs.append(sorted({str(S[i]) for i in ix})); nref.append(len(ix))

    V = np.stack(vecs)
    assert np.all(np.isfinite(V)), "non-finite reference vector"

    # int8 with a single global scale. The vectors are unit length so every
    # component is in [-1, 1]; 127 levels costs about 0.4% of cosine similarity
    # and cuts the file to a quarter.
    scale = 1.0 / 127.0
    q = np.clip(np.round(V / scale), -127, 127).astype(np.int8)
    err = float(np.abs(q.astype(np.float64) * scale - V).max())

    payload = {
        "dim": int(V.shape[1]),
        "scale": scale,
        "words": words,
        "refs": nref,
        "sources": srcs,
        "data": base64.b64encode(q.tobytes()).decode("ascii"),
        "note": "cosine nearest neighbour against the model's 256-d embedding",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    kb = OUT.stat().st_size / 1e3
    print(f"{len(words)} words in the bank, {sum(nref)} reference clips")
    print(f"  max quantisation error {err:.4f}")
    print(f"  wrote {OUT.relative_to(ROOT)}  ({kb:.1f} KB)")
    multi = sum(1 for n in nref if n > 1)
    print(f"  {multi} words have more than one reference")
    if missing:
        print(f"\n  {len(missing)} wanted words have no clip anywhere:")
        print("   " + ", ".join(missing))
    return 0


def _phrases(line: str) -> list[str]:
    """Split a line into words, keeping known two-word signs together."""
    COMPOUND = ("body pain", "thank you", "police station", "sign language")
    out, rest = [], line
    for c in COMPOUND:
        if c in rest:
            out.append(c)
            rest = rest.replace(c, " ")
    out += [w for w in rest.split() if w]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
