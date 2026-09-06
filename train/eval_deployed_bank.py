"""
Score the bank exactly as the app will use it.

    .venv-tf/bin/python train/eval_deployed_bank.py

Reads run/bank83.npz and app/public/model/_bank.json

Everything else measured so far answered a research question. This answers the
product one: given THE 116 words that shipped, THE prototype built by averaging
their reference clips, and THE model in the browser, how often is the right word
first, and how often is it in the top five the user is shown?

Leakage is the thing to get right. In production a prototype is the mean of
every reference the word has. In evaluation the query clip must be left out of
its own prototype, or the word is being matched against a vector partly built
from the answer. Every query here is rebuilt against a prototype computed
without it, which is why the numbers below are lower than a naive run and are
the ones worth quoting.
"""
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "run" / "bank83.npz"
SHIPPED = ROOT / "app" / "public" / "model" / "_bank.json"


def main() -> int:
    z = np.load(BANK, allow_pickle=True)
    E, W, S = z["emb"].astype(np.float64), z["word"], z["source"]
    spec = json.loads(SHIPPED.read_text())
    words = spec["words"]
    pos = {w: i for i, w in enumerate(words)}

    by = defaultdict(list)
    for i, w in enumerate(W):
        if w in pos:
            by[w].append(i)

    # verify the shipped file decodes to what we think it does
    q = np.frombuffer(base64.b64decode(spec["data"]), dtype=np.int8) \
          .reshape(len(words), spec["dim"]).astype(np.float64) * spec["scale"]
    print(f"shipped bank: {len(words)} words, dim {spec['dim']}, "
          f"{sum(spec['refs'])} reference clips")

    def proto(w, drop=None):
        ix = [i for i in by[w] if i != drop]
        if not ix:
            return None
        v = E[ix].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    # full-bank prototypes, used for every word that is not the query's own
    base = {w: proto(w) for w in words}

    rows, truth, cross = [], [], []
    for w in words:
        if len(by[w]) < 2:
            continue                      # cannot be tested: only one clip
        for qi in by[w]:
            held = proto(w, drop=qi)
            if held is None:
                continue
            M = np.stack([held if x == w else base[x] for x in words
                          if base[x] is not None])
            names = [x for x in words if base[x] is not None]
            sims = M @ E[qi]
            rows.append(sims); truth.append(names.index(w))
            other = {str(S[i]) for i in by[w] if i != qi}
            cross.append(str(S[qi]) not in other)

    if not rows:
        print("no testable words")
        return 1
    Sm = np.stack(rows); T = np.array(truth); C = np.array(cross)
    assert np.all(np.isfinite(Sm))
    order = np.argsort(-Sm, axis=1)

    def acc(mask, tag):
        if mask.sum() == 0:
            return
        o, t = order[mask], T[mask]
        r = {k: float((o[:, :k] == t[:, None]).any(1).mean()) * 100 for k in (1, 3, 5, 10)}
        print(f"  {tag:38s} n={int(mask.sum()):4d}   top-1 {r[1]:5.1f}%  "
              f"top-3 {r[3]:5.1f}%  top-5 {r[5]:5.1f}%  top-10 {r[10]:5.1f}%")
        return r

    print(f"\ntestable queries: {len(T)}  (words with >=2 clips)")
    print(f"chance top-1: {100/Sm.shape[1]:.2f}%\n")
    res = {}
    res["all"] = acc(np.ones(len(T), bool), "all queries")
    res["cross_source"] = acc(C, "query from an UNSEEN source")

    untestable = [w for w in words if len(by[w]) < 2]
    print(f"\n{len(untestable)} of {len(words)} words cannot be tested "
          f"(one clip in existence):")
    print("  " + ", ".join(untestable))
    (ROOT / "run" / "deployed_bank_eval.json").write_text(json.dumps(
        {"bank": len(words), "tested": int(len(T)),
         "untestable": untestable, "results": res}, indent=1))
    print("\nwrote run/deployed_bank_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
