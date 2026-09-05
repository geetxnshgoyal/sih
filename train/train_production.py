"""
Train and calibrate the model that actually ships.

    .venv-tf/bin/python train/train_production.py

Reads  data/dataset_merged.npz  +  models/encoder_pretrain.weights.h5
Writes models/gloss_classifier.keras, models/labels.json, models/metrics.json
       run/production.json  (including the temperature to put in calibrate.ts)

How this differs from train.py
------------------------------
train.py is the measurement tool: it runs leave-one-group-out and promotes the
best single fold. That throws away two thirds of the data in the model it ships.
This script separates the two jobs properly:

  1. ESTIMATE   leave-one-group-out over all 7 signer groups, encoder
                warm-started from ASL. Gives the honest accuracy AND, more
                importantly, out-of-fold predictions.
  2. CALIBRATE  fit the softmax temperature on those out-of-fold predictions.
                Every prediction used came from a model that had never seen that
                signer, which is the only way the resulting confidence means
                anything. The old T=2.69 was fitted on one group; refitting is
                mandatory after any retrain, because T is a property of the
                trained weights, not of the architecture.
  3. SHIP       retrain once on ALL data with the same recipe. More data is
                strictly better for the deployed model, and its accuracy is
                estimated by step 1 rather than measured on itself.

Warm-starting from MS-ASL + WLASL is worth +15.7 points on a held-out signer
(ARCHITECTURE.md 5.1). Without models/encoder_pretrain.weights.h5 this still
runs, from scratch, and says so.
"""
import json
import sys
from datetime import datetime
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
OUT_DIR = ROOT / "models"
RUN = ROOT / "run" / "production.json"
SEED = 0
VAL_FRACTION = 0.15
FACTOR = 4

# The bands the app presents in. Fixed here so the fitted temperature is chosen
# against the thresholds it will actually be read through. See calibrate.ts.
CONFIDENT, UNCERTAIN = 0.70, 0.40


def warm_start(model: keras.Model, n_classes: int) -> int:
    """Copy the pretrained encoder in, skipping the classifier head.

    Matched by position and shape: build_model does not name its layers, and a
    shape mismatch is exactly the head, which is what we want to skip.
    """
    if not ENC.exists():
        return 0
    src = build_model(32, 195, 2186)          # the pretraining head size
    try:
        src.load_weights(ENC)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not load {ENC.name}: {exc}")
        return 0
    moved = 0
    for a, b in zip(src.layers, model.layers):
        wa, wb = a.get_weights(), b.get_weights()
        if len(wa) != len(wb) or any(x.shape != z.shape for x, z in zip(wa, wb)):
            continue
        b.set_weights(wa)
        moved += 1
    return moved


def fit(Xtr, ytr, n_classes, rng, verbose=0):
    idx = rng.permutation(len(Xtr))
    cut = max(int((1.0 - VAL_FRACTION) * len(idx)), 1)
    tr, va = idx[:cut], idx[cut:]
    Xa, ya = aug.augment_batch(Xtr[tr], ytr[tr], rng, factor=FACTOR)
    model = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    moved = warm_start(model, n_classes)
    hist = model.fit(
        standardise(Xa), ya,
        validation_data=(standardise(Xtr[va]), ytr[va]),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                              patience=5, min_lr=1e-5),
        ],
        verbose=verbose,
    )
    return model, moved, len(hist.history["loss"])


def temperature_scale(p: np.ndarray, T: float) -> np.ndarray:
    """Exactly calibrate() in app/src/lib/calibrate.ts, vectorised.

    The shipped graph model bakes softmax into its last layer, so true logits
    are unavailable at runtime. log(p) recovers them up to a constant and
    softmax is invariant to that constant, so this is equivalent to scaling the
    logits without re-exporting the model.
    """
    s = np.log(np.maximum(p, 1e-12)) / T
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: mean gap between confidence and accuracy."""
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Pick T by negative log-likelihood on out-of-fold predictions.

    NLL rather than ECE: ECE is a binned statistic and its minimum sits on a
    plateau, so small changes in binning move the answer. NLL is smooth and is
    the loss the probabilities are actually meant to be good under.
    """
    grid = np.arange(0.5, 8.01, 0.01)
    best_T, best_nll = 1.0, np.inf
    rows = np.arange(len(y))
    for T in grid:
        q = temperature_scale(p, T)
        nll = -np.log(np.maximum(q[rows, y], 1e-12)).mean()
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA.relative_to(ROOT)} — run train/preprocess_cislr.py")
        return 1
    d = np.load(DATA, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    labels = [str(s) for s in d["labels"]]
    n_classes = len(labels)
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    print(f"{len(X)} clips | {n_classes} classes | groups {np.bincount(signer).tolist()}")
    print(f"  INCLUDE {int((corpus == 0).sum())}  CISLR {int((corpus == 1).sum())}")
    print(f"  encoder: {'MS-ASL + WLASL' if ENC.exists() else 'NONE — training from scratch'}\n")

    # ---------- 1. estimate, and collect out-of-fold predictions ----------
    groups = sorted(set(signer.tolist()))
    oof_p = np.zeros((len(X), n_classes), dtype=np.float32)
    oof_seen = np.zeros(len(X), dtype=bool)
    per_group = []

    for g in groups:
        te = signer == g
        tr = ~te
        model, moved, ep = fit(X[tr], y[tr], n_classes, rng)
        # Score at close range: that is the condition the app runs in, and it is
        # also the distribution the temperature must be fitted against.
        Xc = close_range(X[te])
        p = model.predict(standardise(Xc), verbose=0)
        oof_p[te], oof_seen[te] = p, True
        t1 = float((p.argmax(1) == y[te]).mean())
        t5 = float(np.mean([y[te][i] in p[i].argsort()[-5:] for i in range(int(te.sum()))]))
        src = "INCLUDE" if corpus[te][0] == 0 else "CISLR"
        per_group.append({"group": int(g), "corpus": src, "clips": int(te.sum()),
                          "close_top1": t1, "close_top5": t5, "epochs": ep})
        print(f"  group {g} ({src:7s}, {int(te.sum()):4d} clips)  "
              f"close top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%   "
              f"({ep} epochs, {moved} encoder layers)", flush=True)
        keras.backend.clear_session()

    inc_g = [r for r in per_group if r["corpus"] == "INCLUDE"]
    cis_g = [r for r in per_group if r["corpus"] == "CISLR"]
    inc1 = float(np.mean([r["close_top1"] for r in inc_g]))
    inc5 = float(np.mean([r["close_top5"] for r in inc_g]))
    cis1 = float(np.mean([r["close_top1"] for r in cis_g])) if cis_g else 0.0
    print(f"\n  INCLUDE groups mean  close top-1 {inc1*100:5.1f}%  top-5 {inc5*100:5.1f}%")
    print(f"  CISLR groups mean    close top-1 {cis1*100:5.1f}%"
          f"   (small groups, few classes each — noisy)")

    # ---------- 2. calibrate on the out-of-fold predictions ----------
    p, yy = oof_p[oof_seen], y[oof_seen]
    T = fit_temperature(p, yy)
    before, after = ece(p, yy), ece(temperature_scale(p, T), yy)
    cal = temperature_scale(p, T)
    print(f"\n  temperature T = {T:.2f}   ECE {before*100:.1f}pp -> {after*100:.1f}pp"
          f"   (on {len(yy)} out-of-fold predictions)")

    bands = []
    for floor in (0.30, UNCERTAIN, 0.50, CONFIDENT, 0.90):
        m = cal.max(1) >= floor
        rate = float(m.mean())
        acc = float((cal[m].argmax(1) == yy[m]).mean()) if m.any() else 0.0
        bands.append({"floor": floor, "speaks": rate, "accuracy": acc})
        print(f"    floor {floor:.2f}   speaks {rate*100:4.1f}%   and is right {acc*100:5.1f}%")

    # ---------- 3. train the model that ships, on everything ----------
    print("\n  final model, all groups...", flush=True)
    final, moved, ep = fit(X, y, n_classes, rng)
    print(f"  done ({ep} epochs, {moved} encoder layers transferred)")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{stamp}_close{inc1*100:.1f}"
    # Never overwrite a checkpoint whose score has not been beaten: an earlier
    # version of train.py wrote a single filename and three runs destroyed the
    # best model we had.
    final.save(OUT_DIR / f"gloss_classifier_{tag}.keras")
    final.save(OUT_DIR / "gloss_classifier.keras")
    (OUT_DIR / "labels.json").write_text(json.dumps(labels))

    doc = {
        "trained": stamp,
        "clips": int(len(X)), "classes": n_classes,
        "pretrained_encoder": ENC.exists(),
        "held_out_group": per_group,
        "held_out_close_mean": {"top1": inc1, "top5": inc5},
        "cislr_groups_close_top1": cis1,
        "temperature": T,
        "ece_before": before, "ece_after": after,
        "bands": bands,
        "final_epochs": ep,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(doc, indent=2))
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(doc, indent=2))

    print(f"\nsaved models/gloss_classifier.keras  (+ {tag} kept)")
    print(f"\nNEXT: set TEMPERATURE = {T:.2f} in app/src/lib/calibrate.ts,")
    print(f"      then .venv-tf/bin/python train/export_tfjs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
