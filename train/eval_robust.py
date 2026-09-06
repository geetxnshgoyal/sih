"""
Two cheap ideas that have never been tried, measured against a control.

    .venv-tf/bin/python train/eval_robust.py

Reads  data/dataset_merged.npz  +  models/encoder_pretrain.weights.h5
Writes run/robust_eval.json

Self-supervised pretraining on 60,713 windows of Indian Sign Language bought
+1.3 points (run/ssl_eval.json). Before reaching for more data, two things that
cost almost nothing and have never been measured here:

  HAND DROPOUT      Every augmentation in augment.py perturbs geometry. None
                    reproduces the failure that actually happens: the tracker
                    losing a hand. Presence is 0.89 on INCLUDE and 0.79 on the
                    dictionary, so the model meets zeroed hands constantly at
                    inference and has never seen one in training.

  TEST-TIME AUG     Average the prediction over the clip and a few perspective
                    variants. Costs nothing at training time and nothing to
                    ship; the only price is a few extra forward passes through
                    a 2.1 MB model.

Four conditions from two trainings per fold, because TTA is inference-only and
free to add to either arm:

    baseline            baseline + TTA
    hand dropout        hand dropout + TTA

Leave-one-group-out over the three INCLUDE signer groups, validation carved from
train, test scored once. Identical protocol to train_production.py, so 64.8% is
the number to beat.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent))
import augment as aug
from train import BATCH, EPOCHS, build_model, close_range, standardise, train_val_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_merged.npz"
ENC = ROOT / "models" / "encoder_pretrain.weights.h5"
OUT = ROOT / "run" / "robust_eval.json"
SEED = 0
VAL_FRACTION = 0.15
FACTOR = 4


def warm_start(model, n_classes):
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
    keras.backend.clear_session()
    return moved


def fit(X, y, tr_m, n_classes, rng):
    idx = np.flatnonzero(tr_m)
    core_i, va_i = train_val_split(len(idx), rng, VAL_FRACTION)
    core, va = idx[core_i], idx[va_i]
    Xa, ya = aug.augment_batch(X[core], y[core], rng, factor=FACTOR)
    m = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    warm_start(m, n_classes)
    m.fit(standardise(Xa), ya,
          validation_data=(standardise(X[va]), y[va]),
          epochs=EPOCHS, batch_size=BATCH,
          callbacks=[
              keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                            restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                                patience=5, min_lr=1e-5),
          ], verbose=0)
    return m


def predict(model, X4, tta: bool):
    """Plain prediction, or the mean over the clip plus perspective variants.

    Averaging PROBABILITIES rather than logits: the exported graph model bakes
    softmax in, so probabilities are what the browser would actually have to
    combine. Measuring anything else would not transfer.
    """
    p = model.predict(standardise(X4), verbose=0)
    if not tta:
        return p
    acc = [p]
    for d in (2.2, 3.5):
        V = X4.copy()
        denom = np.maximum(1.0 + V[..., 2] / d, 0.25)
        V[..., 0] /= denom
        V[..., 1] /= denom
        acc.append(model.predict(standardise(V), verbose=0))
    return np.mean(acc, axis=0)


def score(p, y):
    t1 = float((p.argmax(1) == y).mean())
    t5 = float(np.mean([y[i] in p[i].argsort()[-5:] for i in range(len(y))]))
    return t1, t5


def main() -> int:
    d = np.load(DATA, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    n_classes = len(d["labels"])
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)
    print(f"{len(X)} clips | {n_classes} classes | encoder: {ENC.exists()}\n")

    rows = []
    for g in (0, 1, 2):
        te = (signer == g) & (corpus == 0)
        tr = ~te
        Xc, yt = close_range(X[te]), y[te]
        for drop in (False, True):
            aug.USE_HAND_DROPOUT = drop
            m = fit(X, y, tr, n_classes, rng)
            for tta in (False, True):
                t1, t5 = score(predict(m, Xc, tta), yt)
                name = ("hand-dropout" if drop else "baseline") + (" + TTA" if tta else "")
                print(f"  group {g}  {name:22s} top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%",
                      flush=True)
                rows.append({"group": int(g), "dropout": drop, "tta": tta,
                             "top1": t1, "top5": t5})
            keras.backend.clear_session()
    aug.USE_HAND_DROPOUT = False

    print("\n" + "=" * 60)
    best = None
    for drop in (False, True):
        for tta in (False, True):
            sel = [r for r in rows if r["dropout"] == drop and r["tta"] == tta]
            m1 = float(np.mean([r["top1"] for r in sel]))
            m5 = float(np.mean([r["top5"] for r in sel]))
            name = ("hand-dropout" if drop else "baseline") + (" + TTA" if tta else "")
            print(f"  {name:24s} {m1*100:5.1f}%   top-5 {m5*100:5.1f}%")
            if best is None or m1 > best[1]:
                best = (name, m1, m5)
    print(f"\n  best: {best[0]} at {best[1]*100:.1f}%")
    print(f"  shipped model (same protocol): 64.8%")
    print("=" * 60)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "best": best[0],
                               "best_top1": best[1], "shipped": 0.648}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
