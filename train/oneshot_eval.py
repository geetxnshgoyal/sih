"""
Can a trained encoder recognise a sign it was never trained on, from ONE
reference clip?

    .venv-tf/bin/python train/oneshot_eval.py

This is the question that decides whether the long tail can ever reach the app.

The situation
-------------
Two shapes of data exist, and neither alone is enough:

  DEEP and NARROW   INCLUDE + CISLR: 4,894 clips over 264 words, roughly 15
                    clips per word. Enough to train a classifier, which is what
                    ships today, but the vocabulary is fixed at 83 words and
                    none of them are pain, please, help, yes or no.

  SHALLOW and WIDE  The Government ISL dictionary is 13,331 words with ONE clip
                    each. NCERT adds 400 and ISH Shiksha 448, also one each.
                    Every clinically necessary word lives here and nowhere else.

A classifier cannot use one clip per word: a class with a single example cannot
be both taught and examined. But retrieval can. Train the embedding on the deep
data, embed each dictionary clip once as a reference, and match a live sign by
nearest neighbour. That is the only route from 83 words to hundreds.

What this measures
------------------
References come from the Government dictionary. Queries come from NCERT and ISH
Shiksha, so every query is a DIFFERENT PERSON on a DIFFERENT CORPUS from its
reference, which is the condition the app actually runs in and the condition
that has punished this project before: INCLUDE to CISLR cross-corpus was 2.1%
top-1.

Two bank sizes are reported, because they answer different questions:

  full      every dictionary word is a candidate. The hard number.
  clinical  only the ~150 words a medical phrasebook needs. The number that
            matters if this ships, since the app would never search 13,331.

Read the result honestly. If clinical top-5 is not well clear of chance this
approach does not ship, and the finding gets written down so nobody repeats it.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import clip_io  # noqa: E402
import features  # noqa: E402

POSE_KEEP = features.POSE_KEEP
MIN_ACTIVE, PAD = 8, 2

REF_DIR = ROOT / "data" / "islgov_landmarks"
QRY_DIRS = {"ncert": ROOT / "data" / "ncert_landmarks",
            "shiksha": ROOT / "data" / "shiksha_landmarks"}

CLINICAL = """pain body pain injury hurt wound please help thank you sorry yes no
water food eat drink doctor nurse medicine tablet injection hospital emergency
ambulance operation patient treatment sick ill fever cold hot headache stomach
head hand leg eye ear heart blood bandage breathe sleep toilet bathroom stop go
come wait more less good bad big small name time today tomorrow yesterday now
morning evening night day week month year mother father sister brother family
child baby man woman friend home house school work money phone car bus train
hungry thirsty tired happy sad afraid angry understand know want need can cannot
open close give take sit stand walk run fall dizzy weak strong left right up
down front back inside outside near far first last new old""".split("\n")
CLINICAL = set(" ".join(CLINICAL).split()) | {
    "body pain", "thank you", "intensive care unit"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()


def clip_features(path: Path) -> np.ndarray | None:
    """One landmark npz -> the shared 32x65x3 feature contract."""
    pts, aspect = clip_io.load_points(path, POSE_KEEP, MIN_ACTIVE, PAD)
    if pts is None:
        return None
    v = features.extract(pts, aspect)
    return v.reshape(features.SEQ_LEN, features.N_POINTS, features.N_DIMS) \
        if np.all(np.isfinite(v)) else None


def load_dir(d: Path, limit_words: set | None = None, cap: int | None = None):
    """word -> list of feature arrays."""
    out = defaultdict(list)
    if not d.is_dir():
        return out
    for wd in sorted(d.iterdir()):
        if not wd.is_dir():
            continue
        w = norm(wd.name)
        if limit_words is not None and w not in limit_words:
            continue
        for f in sorted(wd.glob("*.npz")):
            v = clip_features(f)
            if v is not None:
                out[w].append(v)
            if cap and len(out.get(w, ())) >= cap:
                break
    # a directory whose every clip failed the hand-presence test leaves an empty
    # entry behind; drop those rather than carry a word with no reference
    return {w: v for w, v in out.items() if v}


def main() -> int:
    import tensorflow as tf

    mp = ROOT / "models" / "gloss_classifier.keras"
    if not mp.exists():
        print(f"missing {mp}")
        return 1
    full = tf.keras.models.load_model(mp, compile=False)
    # embedding = the layer feeding the softmax, i.e. everything the classifier
    # learned about sign shape minus its commitment to 83 specific words
    emb_layer = full.layers[-2]
    enc = tf.keras.Model(full.input, emb_layer.output)
    print(f"encoder: {mp.name}  embedding dim {enc.output_shape[-1]}\n")

    def embed(arrs):
        # the classifier takes (T, points*dims); features.extract returns the
        # flat contract, so fold it back to the shape the graph was built for
        X = np.stack(arrs).reshape(len(arrs), features.SEQ_LEN,
                                   features.N_POINTS * features.N_DIMS)
        E = enc.predict(X, batch_size=256, verbose=0).astype(np.float64)
        # a dead ReLU embedding is all zeros, and a few clips produce one. Left
        # alone it normalises to inf and then poisons every similarity it takes
        # part in, so it is zeroed and will simply never be anyone's neighbour.
        E[~np.isfinite(E)] = 0.0
        n = np.linalg.norm(E, axis=1, keepdims=True)
        E = np.divide(E, n, out=np.zeros_like(E), where=n > 1e-9)
        return E

    print("loading queries (NCERT + ISH Shiksha) ...")
    queries = defaultdict(list)
    for name, d in QRY_DIRS.items():
        got = load_dir(d)
        for w, vs in got.items():
            queries[w].extend(vs)
        print(f"  {name:9s} {len(got):4d} words")

    print("loading reference bank (Government ISL dictionary) ...")
    refs = load_dir(REF_DIR, cap=1)
    print(f"  islgov   {len(refs):5d} words with a usable clip")

    shared = sorted(set(refs) & set(queries))
    print(f"\nwords with BOTH a reference and an unseen-corpus query: {len(shared)}")
    if len(shared) < 20:
        print("too few to measure")
        return 1

    bank_words = sorted(refs)
    R = embed([refs[w][0] for w in bank_words])
    widx = {w: i for i, w in enumerate(bank_words)}

    qs, qw = [], []
    for w in shared:
        for v in queries[w]:
            qs.append(v)
            qw.append(w)
    Q = embed(qs)
    print(f"queries embedded: {len(qs)}\n")

    def report(name: str, keep: list[str]):
        cols = np.array([widx[w] for w in keep])
        sub = R[cols]
        ok = [i for i, w in enumerate(qw) if w in set(keep)]
        if not ok:
            print(f"{name}: no queries")
            return
        S = Q[ok] @ sub.T
        truth = np.array([keep.index(qw[i]) for i in ok])
        order = np.argsort(-S, axis=1)
        n = len(ok)
        r = {k: float((order[:, :k] == truth[:, None]).any(1).mean()) * 100
             for k in (1, 5, 10)}
        chance = 100.0 / len(keep)
        print(f"{name:28s} bank {len(keep):5d}  n={n:4d}   "
              f"top-1 {r[1]:5.1f}%  top-5 {r[5]:5.1f}%  top-10 {r[10]:5.1f}%   "
              f"(chance {chance:.2f}%)")
        return r

    # THE CONFOUND. The encoder was trained on 83 words. If retrieval only works
    # for those, the number says nothing about pain, please or help, which are
    # exactly the words that are not among them. Splitting is the whole point.
    trained = {norm(w) for w in json.loads(
        (ROOT / "app" / "public" / "model" / "labels.json").read_text())}

    print("=" * 78)
    print("ONE-SHOT RETRIEVAL: reference = 1 dictionary clip, query = unseen corpus")
    print("=" * 78)
    res = {}
    res["full"] = report("full dictionary", bank_words)
    clin = [w for w in bank_words if w in CLINICAL]
    res["clinical"] = report("clinical vocabulary", clin)

    print("\n--- split by whether the encoder was TRAINED on the word ---")
    qset = {w for w in qw}
    seen_q = sorted(qset & trained)
    unseen_q = sorted(qset - trained)
    res["clinical_seen"] = report("clinical, word in the 83", 
                                  [w for w in clin if w in trained])
    res["clinical_unseen"] = report("clinical, word NOT in the 83",
                                    [w for w in clin if w not in trained])
    res["full_unseen"] = report("full dict, word NOT in the 83",
                                [w for w in bank_words if w not in trained])
    print(f"\n  query words trained on: {len(seen_q)}   never trained on: {len(unseen_q)}")

    # per-word detail for the words this project actually needs
    TARGETS = ["pain", "body pain", "please", "help", "water", "blood", "bandage",
               "injection", "sorry", "stop", "food", "nurse", "doctor", "sick",
               "hospital", "medicine", "emergency", "operation", "injury"]
    clin_idx = {w: i for i, w in enumerate(clin)}
    print("\n--- per-word rank within the clinical bank ---")
    for t in TARGETS:
        rows = [i for i, w in enumerate(qw) if w == t and t in clin_idx]
        if not rows:
            print(f"  {t:16s} no query clip")
            continue
        sub = R[[widx[w] for w in clin]]
        sims = Q[rows] @ sub.T
        for r, row in enumerate(rows):
            rank = int((np.argsort(-sims[r]) == clin_idx[t]).nonzero()[0][0]) + 1
            tag = "TRAINED" if t in trained else "unseen"
            print(f"  {t:16s} rank {rank:4d} / {len(clin)}   ({tag})")
    (ROOT / "run" / "oneshot_eval.json").write_text(json.dumps(
        {"shared_words": len(shared), "queries": len(qs),
         "bank_full": len(bank_words), "results": res}, indent=1))
    print("\nwrote run/oneshot_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
