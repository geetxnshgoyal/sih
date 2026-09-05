"""
Pretrain the encoder on American Sign Language, fine-tune on Indian.

    .venv-tf/bin/python train/pretrain.py

Reads  data/pretrain.npz  +  data/dataset_merged.npz
Writes models/encoder_pretrain.weights.h5, run/pretrain_eval.json

The question this answers
------------------------
ARCHITECTURE.md 5.1: a model trained on INCLUDE scores 2.1% on CISLR against
0.38% chance. It learned one corpus, not the task. Two causes were found; this
script tests the fix for the second.

  1. GEOMETRY: MediaPipe normalises x by frame width and y by frame height, so
     INCLUDE's 16:9 video came out stretched 1.78x vertically while every other
     corpus is square or near it. Fixed in features.isotropic, and the fix is
     already in the data this script loads.

  2. VARIETY: INCLUDE is seven students, one room, one camera. No architecture
     change can make that generalise, and 610 more CISLR clips bought only
     +1.4 points. What is missing is people and conditions, and ISL data of that
     kind does not exist for free.

MS-ASL and WLASL are a different LANGUAGE, which is exactly why they are safe:
their signs cannot leak answers. What transfers is the encoder, what a
handshape looks like on a webcam, how a trajectory unfolds, which joints move
together. 38,758 clips from 130+ signers off YouTube carry far more variety than
4,894 studio clips from seven students ever will. The head is discarded and
retrained on ISL. See preprocess_pretrain.py for what is in the set and why
AUTSL is not.

Measured with MS-ASL alone (17,664 clips): cross-corpus 4.9% -> 12.6%,
within-corpus 38.4% -> 55.4%.

Arms
----
  P0  scratch     INCLUDE -> CISLR        cross-corpus, no pretraining
  P1  pretrained  INCLUDE -> CISLR        the same, encoder warm-started
  P2  scratch     INCLUDE+CISLR -> grp 0  within-corpus, no pretraining
  P3  pretrained  INCLUDE+CISLR -> grp 0  the same, encoder warm-started

P1 - P0 is the number that matters. P3 - P2 says whether the transfer also helps
the easier within-corpus case, or only rescues the hard one.

Protocol: validation is carved out of TRAIN. The test set is scored once, at the
end. See train.py's run() for why that is stated so loudly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent))
import augment as aug
from train import BATCH, build_model, close_range, standardise

ROOT = Path(__file__).resolve().parent.parent
PRE = ROOT / "data" / "pretrain.npz"
ISL = ROOT / "data" / "dataset_merged.npz"
ENC = ROOT / "models" / "encoder_pretrain.weights.h5"
OUT = ROOT / "run" / "pretrain_eval.json"
SEED = 0

# Pretraining is an initialisation, not a deliverable: a cheaper budget than the
# fine-tune is the right trade. factor=1 rather than 4 because 38,758 real clips
# from 130+ signers already carry the variety augmentation is a substitute for , 
# and because the augmented array is held in memory whole (factor=2 on this set
# is 1.9 GB before standardise copies it again).
PRE_EPOCHS, PRE_FACTOR = 30, 1
FT_EPOCHS, FT_FACTOR = 60, 4
VAL_FRACTION = 0.15


def split_train_val(n, rng):
    idx = rng.permutation(n)
    cut = max(int((1.0 - VAL_FRACTION) * n), 1)
    return idx[:cut], idx[cut:]


def transfer_encoder(src: keras.Model, dst: keras.Model) -> int:
    """Copy every layer except the classifier head. Returns layers copied.

    Matched by position and shape rather than by name: build_model does not name
    its layers, and Keras auto-names are not stable across two calls in one
    process. A shape mismatch means the head, which is exactly what we skip.
    """
    moved = 0
    for a, b in zip(src.layers, dst.layers):
        wa, wb = a.get_weights(), b.get_weights()
        if len(wa) != len(wb) or any(x.shape != z.shape for x, z in zip(wa, wb)):
            continue
        b.set_weights(wa)
        moved += 1
    return moved


def pretrain(rng) -> keras.Model:
    d = np.load(PRE, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    n_classes = len(d["labels"])

    # Validation is the corpora's OWN signer-independent split, not a random
    # slice. Early stopping therefore selects the epoch that generalises to
    # unseen signers, which is the exact property the ISL model lacks. A random
    # split would select for memorising signers, the failure being fixed here.
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split != "train")
    print(f"pretrain corpus: {len(X)} clips, {n_classes} classes "
          f"({len(tr)} train / {len(va)} val, signer-independent)")

    Xa, ya = aug.augment_batch(X[tr], y[tr], rng, factor=PRE_FACTOR)
    model = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    hist = model.fit(
        standardise(Xa), ya,
        validation_data=(standardise(X[va]), y[va]),
        epochs=PRE_EPOCHS, batch_size=BATCH,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                              patience=4, min_lr=1e-5),
        ],
        verbose=2,
    )
    acc = max(hist.history["val_accuracy"])
    print(f"  pretrain val top-1 {acc*100:.1f}% over {n_classes} ASL classes, "
          f"held-out signers ({len(hist.history['loss'])} epochs)\n")
    ENC.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(ENC)
    return model


def arm(name, desc, src, X, y, tr_m, te_m, n_classes, rng):
    tr_idx = np.flatnonzero(tr_m)
    core, va = split_train_val(len(tr_idx), rng)
    core, va = tr_idx[core], tr_idx[va]

    Xa, ya = aug.augment_batch(X[core], y[core], rng, factor=FT_FACTOR)
    model = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    moved = transfer_encoder(src, model) if src is not None else 0

    hist = model.fit(
        standardise(Xa), ya,
        validation_data=(standardise(X[va]), y[va]),
        epochs=FT_EPOCHS, batch_size=BATCH,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                              patience=5, min_lr=1e-5),
        ],
        verbose=0,
    )
    def score(A):
        p = model.predict(standardise(A), verbose=0)
        yt = y[te_m]
        return (float((p.argmax(1) == yt).mean()),
                float(np.mean([yt[i] in p[i].argsort()[-5:] for i in range(len(yt))])))
    t1, t5 = score(X[te_m])
    c1, c5 = score(close_range(X[te_m]))
    print("=" * 66)
    print(f"ARM {name}, {desc}")
    print(f"  train {len(core)}  val {len(va)}  test {int(te_m.sum())}"
          f"   encoder layers transferred: {moved}")
    print(f"  far   top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%   "
          f"({len(hist.history['loss'])} epochs)")
    print(f"  close top-1 {c1*100:5.1f}%  top-5 {c5*100:5.1f}%\n", flush=True)
    return {"arm": name, "desc": desc, "pretrained": src is not None,
            "layers_transferred": moved, "train": len(core), "test": int(te_m.sum()),
            "far_top1": t1, "far_top5": t5, "close_top1": c1, "close_top5": c5,
            "epochs": len(hist.history["loss"])}


def main() -> int:
    for p in (PRE, ISL):
        if not p.exists():
            print(f"missing {p.relative_to(ROOT)}")
            return 1
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    print("=" * 66)
    print("PRETRAINING on MS-ASL + WLASL")
    print("=" * 66)
    src = pretrain(rng)

    d = np.load(ISL, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    n_classes = len(d["labels"])
    inc, cis = corpus == 0, corpus == 1
    print(f"ISL: {len(X)} clips, {n_classes} classes "
          f"(INCLUDE {inc.sum()}, CISLR {cis.sum()})\n")

    results = [
        arm("P0", "scratch    INCLUDE -> CISLR", None, X, y, inc, cis, n_classes, rng),
        arm("P1", "pretrained INCLUDE -> CISLR", src, X, y, inc, cis, n_classes, rng),
        arm("P2", "scratch    INCLUDE+CISLR -> INCLUDE group 0", None,
            X, y, (inc & (signer != 0)) | cis, inc & (signer == 0), n_classes, rng),
        arm("P3", "pretrained INCLUDE+CISLR -> INCLUDE group 0", src,
            X, y, (inc & (signer != 0)) | cis, inc & (signer == 0), n_classes, rng),
    ]

    print("=" * 66)
    print("SUMMARY  (close-range)")
    print("=" * 66)
    for r in results:
        print(f"  {r['arm']}  {r['close_top1']*100:5.1f}%  "
              f"top-5 {r['close_top5']*100:5.1f}%   {r['desc']}")
    x1 = (results[1]["close_top1"] - results[0]["close_top1"]) * 100
    x2 = (results[3]["close_top1"] - results[2]["close_top1"]) * 100
    print(f"\n  cross-corpus  P1 - P0 = {x1:+.1f} points")
    print(f"  within-corpus P3 - P2 = {x2:+.1f} points")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"classes": n_classes, "arms": results}, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
