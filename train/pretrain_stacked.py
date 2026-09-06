"""
Stack the two pretraining stages instead of choosing between them.

    .venv-tf/bin/python train/pretrain_stacked.py

Reads  data/ssl.npz, data/pretrain.npz, data/dataset_merged.npz
Writes models/encoder_stacked.weights.h5, run/stacked_eval.json

SSL and supervised ASL have only ever been tested as ALTERNATIVES:

    scratch                     46.4%
    + SSL on 13,662 ISL clips   47.7%   (+1.3)
    + ASL supervised (38,758)   64.8%   (+18.4)

Never in sequence. This runs

    SSL init  ->  ASL supervised  ->  ISL fine-tune

on the theory that starting the ASL stage from a representation that already
knows how Indian hands move gives it a better place to begin, and that the two
signals compound rather than one replacing the other.

The honest prior is that they do NOT compound much: the ASL stage trains every
encoder layer, so it is free to overwrite whatever SSL learned. The reason to
run it anyway is that it is cheap, both encoders already exist, and it is the
last untested combination.

Measured on the 38-class clinical vocabulary, which is the current best model
(73.2% held-out signer), so a gain here is a gain to the thing that would ship.

Four arms, identical but for the encoder they start from:
    none        no pretraining at all
    ssl         SSL only
    asl         supervised ASL only          <- what ships today
    stacked     SSL then supervised ASL
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
SSL = ROOT / "data" / "ssl.npz"
ASL = ROOT / "data" / "pretrain.npz"
ISL = ROOT / "data" / "dataset_merged.npz"
ENC_SSL = ROOT / "models" / "encoder_ssl.weights.h5"
ENC_ASL = ROOT / "models" / "encoder_pretrain.weights.h5"
ENC_OUT = ROOT / "models" / "encoder_stacked.weights.h5"
OUT = ROOT / "run" / "stacked_eval.json"
SEED = 0
VAL_FRACTION = 0.15
ASL_EPOCHS, ASL_FACTOR = 30, 1


def copy_encoder(src: keras.Model, dst: keras.Model) -> int:
    """Position-and-shape matched copy. A mismatch is the head, which we skip."""
    moved = 0
    for a, b in zip(src.layers, dst.layers):
        wa, wb = a.get_weights(), b.get_weights()
        if len(wa) != len(wb) or any(x.shape != z.shape for x, z in zip(wa, wb)):
            continue
        b.set_weights(wa)
        moved += 1
    return moved


def load_ssl_encoder() -> keras.Model | None:
    """The SSL encoder was saved as its own Model of build_model's layers 0-11."""
    if not ENC_SSL.exists():
        return None
    sys.path.insert(0, str(ROOT / "train"))
    from pretrain_ssl import build_encoder
    enc = build_encoder(32, 195)
    enc.load_weights(ENC_SSL)
    return enc


def train_asl(init_from: keras.Model | None, rng) -> keras.Model:
    """Supervised ASL, optionally warm-started. Returns the trained model."""
    d = np.load(ASL, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    n_classes = len(d["labels"])
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split != "train")
    Xa, ya = aug.augment_batch(X[tr], y[tr], rng, factor=ASL_FACTOR)
    m = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    if init_from is not None:
        n = copy_encoder(init_from, m)
        print(f"    ASL stage warm-started from SSL: {n} layers", flush=True)
    h = m.fit(standardise(Xa), ya,
              validation_data=(standardise(X[va]), y[va]),
              epochs=ASL_EPOCHS, batch_size=BATCH,
              callbacks=[
                  keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                                restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                                    patience=4, min_lr=1e-5),
              ], verbose=0)
    print(f"    ASL val top-1 {max(h.history['val_accuracy'])*100:.1f}% "
          f"over {n_classes} classes", flush=True)
    return m


def finetune(src, X, y, tr_m, te_m, n_classes, rng):
    idx = np.flatnonzero(tr_m)
    rng.shuffle(idx)
    cut = max(int((1 - VAL_FRACTION) * len(idx)), 1)
    core, va = idx[:cut], idx[cut:]
    Xa, ya = aug.augment_batch(X[core], y[core], rng, factor=4)
    m = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    if src is not None:
        copy_encoder(src, m)
    m.fit(standardise(Xa), ya,
          validation_data=(standardise(X[va]), y[va]),
          epochs=EPOCHS, batch_size=BATCH,
          callbacks=[
              keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                            restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                                patience=5, min_lr=1e-5),
          ], verbose=0)
    p = m.predict(standardise(close_range(X[te_m])), verbose=0)
    yt = y[te_m]
    t1 = float((p.argmax(1) == yt).mean())
    t5 = float(np.mean([yt[i] in p[i].argsort()[-5:] for i in range(len(yt))]))
    keras.backend.clear_session()
    return t1, t5


def main() -> int:
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    d = np.load(ISL, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    labels = [str(s) for s in d["labels"]]
    keep = json.loads((ROOT / "models" / "clinical" / "labels.json").read_text())
    ids = [labels.index(w) for w in keep]
    remap = -np.ones(len(labels), np.int64)
    for new, old in enumerate(ids):
        remap[old] = new
    m = np.isin(y, ids)
    X, y, signer, corpus = X[m], remap[y[m]], signer[m], corpus[m]
    n_classes = len(keep)
    print(f"clinical vocabulary: {n_classes} classes, {len(X)} clips\n")

    # ---- build the encoders once ----
    print("encoders:")
    ssl_enc = load_ssl_encoder()
    print(f"  ssl      : {'loaded' if ssl_enc else 'MISSING'}")
    asl_plain = build_model(32, 195, 2186)
    asl_plain.load_weights(ENC_ASL)
    print("  asl      : loaded (as shipped)")
    print("  stacked  : training ASL from the SSL init...")
    stacked = train_asl(ssl_enc, rng)
    stacked.save_weights(ENC_OUT)
    print()

    arms = {"none": None, "ssl": ssl_enc, "asl": asl_plain, "stacked": stacked}
    rows = []
    for g in (0, 1, 2):
        te = (signer == g) & (corpus == 0)
        tr = ~te
        for name, src in arms.items():
            t1, t5 = finetune(src, X, y, tr, te, n_classes, rng)
            print(f"  group {g}  {name:8s} top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%",
                  flush=True)
            rows.append({"group": int(g), "arm": name, "top1": t1, "top5": t5})

    print("\n" + "=" * 56)
    best = None
    for name in arms:
        sel = [r for r in rows if r["arm"] == name]
        m1 = float(np.mean([r["top1"] for r in sel]))
        m5 = float(np.mean([r["top5"] for r in sel]))
        print(f"  {name:10s} top-1 {m1*100:5.1f}%   top-5 {m5*100:5.1f}%")
        if best is None or m1 > best[1]:
            best = (name, m1, m5)
    print(f"\n  best: {best[0]} at {best[1]*100:.1f}% top-1, {best[2]*100:.1f}% top-5")
    print("=" * 56)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "best": best[0],
                               "best_top1": best[1], "best_top5": best[2]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
