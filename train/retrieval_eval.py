"""
Leave-one-clip-out retrieval over every word that has more than one clip.

    .venv-tf/bin/python train/retrieval_eval.py

Reads run/bank.npz (build_bank.py). Runs in seconds.

What this fixes about the first measurement
-------------------------------------------
The first pass compared Government-dictionary references against NCERT and
Shiksha queries and got 73.9% top-1 on words the encoder had never been trained
on. That was the right question but n was 23, and 23 samples put a confidence
interval of roughly plus or minus 18 points on the answer, which is too wide to
build on.

Here every clip of every multi-clip word takes a turn as the query, with a
reference drawn from a DIFFERENT SOURCE wherever one exists. That is a few
hundred trials instead of 23, and it keeps the hard part intact: the reference
is still a different person, filmed by a different organisation, from the query.

Cross-source is enforced rather than preferred where possible, because a
same-source pair would be two clips of the same studio setup and would flatter
the result. Where a word exists in only one source with two clips, the pair is
marked same-source and reported separately.

Bank size is the other axis. Retrieving one word out of 13,000 and one out of
100 are different problems, and the app would only ever search a curated
clinical list, so both are reported.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "run" / (sys.argv[1] if len(sys.argv) > 1 else "bank.npz")


def load():
    z = np.load(BANK, allow_pickle=True)
    return z["emb"].astype(np.float64), z["word"], z["source"]


def main() -> int:
    if not BANK.exists():
        print(f"missing {BANK.relative_to(ROOT)}, run train/build_bank.py first")
        return 1
    E, W, S = load()
    assert np.all(np.isfinite(E)), "cache holds non-finite vectors"
    print(f"bank: {len(E)} clips, {len(set(W))} words, sources {sorted(set(S))}\n")

    by_word = defaultdict(list)
    for i, w in enumerate(W):
        by_word[w].append(i)

    trained = set(json.loads(
        (ROOT / "app" / "public" / "model" / "labels.json").read_text()))
    import re
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()
    trained = {norm(t) for t in trained}

    CLIN = set("""pain body pain injury hurt wound please help thank you sorry yes no
water food eat drink doctor nurse medicine tablet injection hospital emergency
ambulance operation patient treatment sick ill fever cold hot headache stomach
head hand leg eye ear heart blood bandage breathe sleep toilet bathroom stop
wait more less good bad big small name time today tomorrow yesterday morning
evening night day week month year mother father sister brother family child baby
man woman friend home house school work money phone car bus train hungry thirsty
tired happy sad afraid angry understand know want need open close give take sit
stand walk run fall dizzy weak strong left right up down near far new old
""".split()) | {"body pain", "thank you"}

    multi = {w: ix for w, ix in by_word.items() if len(ix) >= 2}
    print(f"words with >=2 clips: {len(multi)}")

    def evaluate(bank_words: list[str], label: str, subset=None):
        """One reference per word; every other clip of those words is a query."""
        bw = [w for w in bank_words if w in by_word]
        if len(bw) < 10:
            print(f"{label}: bank too small ({len(bw)})")
            return None
        # reference: first clip of the word, preferring the dictionary source
        ref_i = []
        for w in bw:
            ix = by_word[w]
            gov = [i for i in ix if S[i] == "islgov"]
            ref_i.append(gov[0] if gov else ix[0])
        R = E[ref_i]
        pos = {w: k for k, w in enumerate(bw)}

        q_idx, q_true, q_cross = [], [], []
        for k, w in enumerate(bw):
            if subset is not None and w not in subset:
                continue
            r = ref_i[k]
            for i in by_word[w]:
                if i == r:
                    continue
                q_idx.append(i); q_true.append(k); q_cross.append(S[i] != S[r])
        if len(q_idx) < 10:
            print(f"{label}: too few queries ({len(q_idx)})")
            return None
        Q, T = E[q_idx], np.array(q_true)
        Sm = Q @ R.T
        assert np.all(np.isfinite(Sm)), "non-finite similarity"
        order = np.argsort(-Sm, axis=1)
        cross = np.array(q_cross)

        def acc(mask):
            if mask.sum() == 0:
                return None
            o, t = order[mask], T[mask]
            return {k: float((o[:, :k] == t[:, None]).any(1).mean()) * 100
                    for k in (1, 5, 10)}
        a, ac = acc(np.ones(len(T), bool)), acc(cross)
        ch = 100.0 / len(bw)
        line = (f"{label:34s} bank {len(bw):5d}  n={len(T):5d}  "
                f"top-1 {a[1]:5.1f}%  top-5 {a[5]:5.1f}%  top-10 {a[10]:5.1f}%  "
                f"(chance {ch:.2f}%)")
        print(line)
        if ac and cross.sum() != len(T):
            print(f"{'  of which cross-source':34s} {'':5s}  n={int(cross.sum()):5d}  "
                  f"top-1 {ac[1]:5.1f}%  top-5 {ac[5]:5.1f}%  top-10 {ac[10]:5.1f}%")
        return {"bank": len(bw), "n": int(len(T)), "all": a, "cross_source": ac}

    allw = sorted(by_word)
    clin = sorted(w for w in allw if w in CLIN)
    unseen_clin = sorted(w for w in clin if w not in trained)

    print("\n" + "=" * 84)
    print("LEAVE-ONE-CLIP-OUT RETRIEVAL")
    print("=" * 84)
    res = {}
    res["full"] = evaluate(allw, "full vocabulary")
    res["clinical"] = evaluate(clin, "clinical bank")
    res["clinical_unseen_q"] = evaluate(clin, "clinical bank, unseen words only",
                                        subset=set(unseen_clin))
    print()
    for n in (50, 100, 200, 400):
        res[f"bank{n}"] = evaluate(allw[:n], f"arbitrary {n}-word bank")

    (ROOT / "run" / "retrieval_eval.json").write_text(json.dumps(res, indent=1))
    print("\nwrote run/retrieval_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
