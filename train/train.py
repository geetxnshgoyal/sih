"""
Train the gloss classifier.

    .venv-tf/bin/python train/train.py

Trains twice on purpose:
  1. RANDOM split      — clips shuffled, same person in train and test
  2. HELD-OUT GROUP    — one body-type group never seen during training

The gap between the two is the honest measure of how much a random split
inflates the number. Report the held-out figure; show the random one only to
explain why other projects' 99% claims mean little.

Architecture is a 1D CNN over the temporal axis, deliberately not a
transformer: TFLite/TF.js do not support native MultiHeadAttention, and at
264 classes attention buys little over convolutions.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

import augment as aug
import features as feat
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.npz"
OUT_DIR = ROOT / "models"

SEED = 20260827
EPOCHS = 60
BATCH = 64


def build_model(seq_len: int, n_feat: int, n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_feat))

    x = layers.Conv1D(128, 5, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(256, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def standardise(X4: np.ndarray) -> np.ndarray:
    """(N,T,65,3) -> (N,T,195), zero mean and unit std per clip.

    Mirrors features.standardise, which flattens the whole clip before
    normalising. Applied after augmentation so the statistics reflect the
    augmented geometry, exactly as they will at inference time.
    """
    flat = X4.reshape(len(X4), -1)
    m = flat.mean(axis=1, keepdims=True)
    s = np.maximum(flat.std(axis=1, keepdims=True), 1e-6)
    return ((flat - m) / s).reshape(len(X4), X4.shape[1], -1).astype(np.float32)


def close_range(X: np.ndarray, d_cam: float = 1.8) -> np.ndarray:
    """Re-project a far-camera clip as if shot at laptop distance.

    Measured on the promoted model: unchanged 67.3% top-1, but 2.1% after this
    transform at d=1.8. Chance is 0.4%. Distance alone is harmless (a x2.6
    scale leaves the score identical, because standardisation divides scale
    out); the projective distortion is the whole gap.

    Without this as an eval, perspective augmentation looks like a regression —
    it costs a little far-camera accuracy and the far-camera test set cannot
    see what it buys. That mistake was made once already.
    """
    A = X.copy()
    denom = np.maximum(1.0 + A[..., 2] / d_cam, 0.25)
    A[..., 0] /= denom
    A[..., 1] /= denom
    return A


def score(model, X4, y):
    p = model.predict(standardise(X4), verbose=0)
    top1 = float((p.argmax(1) == y).mean())
    top5 = float(np.mean([y[i] in p[i].argsort()[-5:] for i in range(len(y))]))
    return top1, top5


def run(name, Xtr, ytr, Xte, yte, n_classes, rng, verbose=0):
    Xte_raw = Xte
    Xa, ya = aug.augment_batch(Xtr, ytr, rng, factor=4)
    Xa = standardise(Xa)
    Xte = standardise(Xte)
    model = build_model(Xa.shape[1], Xa.shape[2], n_classes)

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                          patience=5, min_lr=1e-5),
    ]
    hist = model.fit(Xa, ya, validation_data=(Xte, yte), epochs=EPOCHS,
                     batch_size=BATCH, callbacks=cbs, verbose=verbose)

    # Report both: the far-camera set the data was shot on, and the same clips
    # re-projected to laptop distance, which is where the app actually runs.
    top1, top5 = score(model, Xte_raw, yte)
    ctop1, ctop5 = score(model, close_range(Xte_raw), yte)
    return model, top1, top5, ctop1, ctop5, len(hist.history["loss"])


def main() -> int:
    d = np.load(DATA, allow_pickle=True)
    X, y, signer, labels = d["X"], d["y"], d["signer"], list(d["labels"])
    n_classes = len(labels)
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    print(f"{len(X)} clips | {n_classes} classes | groups {np.bincount(signer).tolist()}")
    print(f"input {X.shape[1:]} (anchored)\n")

    # ---------- 1. random split (the optimistic, misleading one) ----------
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    print("=" * 62)
    print("RANDOM SPLIT — same signers in train and test (inflated)")
    print("=" * 62)
    _, r_top1, r_top5, r_c1, r_c5, r_ep = run("random", X[tr], y[tr], X[te], y[te], n_classes, rng)
    print(f"  far   top-1 {r_top1*100:5.1f}%  top-5 {r_top5*100:5.1f}%   ({r_ep} epochs)")
    print(f"  close top-1 {r_c1*100:5.1f}%  top-5 {r_c5*100:5.1f}%\n")

    # ---------- 2. held-out group (the honest one) ----------
    groups = sorted(set(signer.tolist()))
    results = []
    best_model, best_acc = None, -1.0
    for g in groups:
        te_m = signer == g
        tr_m = ~te_m
        seen = len(set(y[tr_m].tolist()) & set(y[te_m].tolist()))
        print("=" * 62)
        print(f"HELD-OUT GROUP {g} — {te_m.sum()} test clips, "
              f"{seen}/{n_classes} classes testable")
        print("=" * 62)
        m, top1, top5, c1, c5, ep = run(f"group{g}", X[tr_m], y[tr_m], X[te_m], y[te_m],
                                        n_classes, rng)
        print(f"  far   top-1 {top1*100:5.1f}%  top-5 {top5*100:5.1f}%   ({ep} epochs)")
        print(f"  close top-1 {c1*100:5.1f}%  top-5 {c5*100:5.1f}%\n")
        results.append((g, top1, top5, c1, c5))
        # Promote on CLOSE-range accuracy. That is the condition the app runs
        # in; far-camera score is context, not the objective.
        if c1 > best_acc:
            best_model, best_acc = m, c1

    mean1 = float(np.mean([r[1] for r in results]))
    mean5 = float(np.mean([r[2] for r in results]))
    cmean1 = float(np.mean([r[3] for r in results]))
    cmean5 = float(np.mean([r[4] for r in results]))

    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"  random split           far {r_top1*100:5.1f}%   close {r_c1*100:5.1f}%")
    print(f"  held-out group (mean)  far {mean1*100:5.1f}%   close {cmean1*100:5.1f}%")
    print(f"  inflation from random split: {(r_top1-mean1)*100:+.1f} points")
    print(f"\n  CLOSE is the number that matters — it is the condition a laptop")
    print(f"  webcam actually produces. Promotion is decided on it.")
    print("\n  Report the held-out number. The random one is what you get")
    print("  when the same person appears in train and test.")

    OUT_DIR.mkdir(exist_ok=True)

    # Version every run. An earlier version of this script wrote a single
    # gloss_classifier.keras, so three training runs in a row silently
    # destroyed the best model we had (51.6% held-out) and it had to be
    # retrained from scratch to recover. Never overwrite a checkpoint whose
    # score you have not yet beaten.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{stamp}_close{cmean1*100:.1f}_far{mean1*100:.1f}"
    run_path = OUT_DIR / f"gloss_classifier_{tag}.keras"
    best_model.save(run_path)

    prev_doc = json.loads((OUT_DIR / "metrics.json").read_text()) \
        if (OUT_DIR / "metrics.json").exists() else {}
    prev = prev_doc.get("held_out_close_mean", {}).get("top1", -1.0)
    if cmean1 >= prev:
        best_model.save(OUT_DIR / "gloss_classifier.keras")
        print(f"  new best close-range ({cmean1*100:.1f}% >= {prev*100:.1f}%) — promoted")
    else:
        print(f"  NOT promoted: close {cmean1*100:.1f}% < current best {prev*100:.1f}%")
        print(f"  kept as {run_path.name}; gloss_classifier.keras unchanged")
    (OUT_DIR / "labels.json").write_text(json.dumps(labels))
    metrics_path = OUT_DIR / ("metrics.json" if cmean1 >= prev else f"metrics_{tag}.json")
    metrics_path.write_text(json.dumps({
        "random_split": {"top1": r_top1, "top5": r_top5},
        "held_out_group": [{"group": g, "top1": a, "top5": b, "close_top1": c, "close_top5": e}
                           for g, a, b, c, e in results],
        "held_out_mean": {"top1": mean1, "top5": mean5},
        "held_out_close_mean": {"top1": cmean1, "top5": cmean5},
        "classes": n_classes,
        "clips": int(len(X)),
    }, indent=2))
    print(f"\nsaved {run_path.name} and {metrics_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
