"""
How much accuracy is 264 classes costing us?

    .venv-tf/bin/python train/eval_vocab_size.py

Reads  data/dataset_merged.npz  +  models/encoder_pretrain.weights.h5
Writes run/vocab_eval.json

Every training-side change tried so far has landed at zero: face mesh, SL-GCN,
self-supervision on 60,713 ISL windows, hand dropout, test-time augmentation.
When five methods in a row do nothing, the method is not the problem.

This asks a different question, and it is a PRODUCT question rather than a
modelling one. 264 classes over ~19 clips each from about ten distinct signers
is a very thin slice of data per class, and every class is another way to be
wrong. A hospital does not need 264 signs. It needs the forty that come up when
somebody cannot breathe.

So: hold the data, the recipe and the protocol fixed, and vary only how many
classes the model has to separate. Held-out INCLUDE group 0, which is the
hardest of the three folds, so these are conservative.

Classes are picked by clip count, most-supported first. That is the best case
for any given N and therefore the right thing to measure: it says what a
well-chosen vocabulary of that size can achieve, not what an arbitrary one does.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent))
import augment as aug
from train import BATCH, EPOCHS, build_model, close_range, standardise

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_merged.npz"
ENC = ROOT / "models" / "encoder_pretrain.weights.h5"
OUT = ROOT / "run" / "vocab_eval.json"
SEED = 0
VAL_FRACTION = 0.15
SIZES = [20, 40, 80, 150, 264]

# What a clinical kiosk actually has to say. Reported alongside each vocabulary
# so the trade is visible: accuracy is worthless if the words are not the ones
# a patient needs.
CLINICAL = {"pain", "help", "water", "doctor", "medicine", "fever", "head",
            "stomach", "hospital", "sick", "yes", "no", "please", "hot", "cold",
            "eat", "drink", "sleep", "today", "tomorrow", "name", "time"}


def warm_start(model):
    if not ENC.exists():
        return 0
    src = build_model(32, 195, 2186)
    try:
        src.load_weights(ENC)
    except Exception:  # noqa: BLE001
        return 0
    moved = 0
    for a, b in zip(src.layers, model.layers):
        wa, wb = a.get_weights(), b.get_weights()
        if len(wa) != len(wb) or any(x.shape != z.shape for x, z in zip(wa, wb)):
            continue
        b.set_weights(wa)
        moved += 1
    return moved


def run(X, y, tr_m, te_m, n_classes, rng):
    idx = np.flatnonzero(tr_m)
    rng.shuffle(idx)
    cut = max(int((1 - VAL_FRACTION) * len(idx)), 1)
    core, va = idx[:cut], idx[cut:]
    Xa, ya = aug.augment_batch(X[core], y[core], rng, factor=4)
    m = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    warm_start(m)
    m.fit(standardise(Xa), ya,
          validation_data=(standardise(X[va]), y[va]),
          epochs=EPOCHS, batch_size=BATCH,
          callbacks=[
              keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                            restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                                patience=5, min_lr=1e-5),
          ], verbose=0)
    Xc, yt = close_range(X[te_m]), y[te_m]
    p = m.predict(standardise(Xc), verbose=0)
    t1 = float((p.argmax(1) == yt).mean())
    t5 = float(np.mean([yt[i] in p[i].argsort()[-5:] for i in range(len(yt))]))
    keras.backend.clear_session()
    return t1, t5


def main() -> int:
    d = np.load(DATA, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    labels = [str(s) for s in d["labels"]]
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    counts = np.bincount(y, minlength=len(labels))
    order = np.argsort(-counts)          # best-supported classes first
    print(f"{len(X)} clips | {len(labels)} classes | held out: INCLUDE group 0\n")

    rows = []
    for n in SIZES:
        keep = set(order[:n].tolist())
        m = np.isin(y, list(keep))
        remap = -np.ones(len(labels), np.int64)
        for new, old in enumerate(order[:n]):
            remap[old] = new
        Xs, ys = X[m], remap[y[m]]
        sg, cp = signer[m], corpus[m]
        te = (sg == 0) & (cp == 0)
        tr = ~te
        if te.sum() < 30 or tr.sum() < 100:
            continue
        t1, t5 = run(Xs, ys, tr, te, n, rng)
        names = {labels[i].lower() for i in order[:n]}
        clin = len(names & CLINICAL)
        print(f"  {n:>3} classes  {int(tr.sum()):>4} train  {int(te.sum()):>4} test   "
              f"top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%   "
              f"clinical words covered {clin}/{len(CLINICAL)}", flush=True)
        rows.append({"classes": n, "top1": t1, "top5": t5,
                     "clinical_covered": clin, "train": int(tr.sum()),
                     "test": int(te.sum())})

    print("\n" + "=" * 62)
    for r in rows:
        bar = "#" * int(r["top1"] * 50)
        print(f"  {r['classes']:>3}  {r['top1']*100:5.1f}%  {bar}")
    print("=" * 62)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
