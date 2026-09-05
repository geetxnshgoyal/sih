"""
Self-supervised pretraining on unlabelled Indian Sign Language.

    .venv-tf/bin/python train/pretrain_ssl.py

Reads  data/ssl.npz  +  data/dataset_merged.npz
Writes models/encoder_ssl.weights.h5, run/ssl_eval.json

The idea
--------
Mask a third of the frames in a clip and make the network reconstruct them. To
fill in a hidden frame it has to learn how a hand moves -- where a trajectory was
heading, which joints travel together, what a handshape does between two
observed moments. No labels are involved anywhere, so the ISL dictionary's fatal
flaw as supervised data (median ONE clip per word across 12,103 words) simply
does not apply.

Then throw the reconstruction head away and keep the encoder.

Why this might beat the ASL pretraining already shipping
--------------------------------------------------------
train_production.py warm-starts from MS-ASL + WLASL: 38,758 clips, but of a
DIFFERENT LANGUAGE. This is Indian Sign Language -- the actual target -- so the
handshapes, the grammar of movement and the signing space are the ones the model
will meet in the field.

Architecture
------------
Layers 0-11 are build_model's, unchanged and in the same order, so the learned
weights transfer by position with no name matching or surgery. Where the
classifier pools and predicts a class, this upsamples and predicts coordinates:

    build_model     ... Conv1D(256,3) BN ReLU | GAP  Dropout Dense(n_classes)
    this script     ... Conv1D(256,3) BN ReLU | UpSampling Conv1D(195,3)

GlobalAveragePooling is exactly what a reconstruction head cannot use -- it
discards the time axis the task depends on -- which is why the split is there.

Protocol: leave-one-INCLUDE-group-out, validation carved from train, test group
scored once. Same as train_production.py, so 64.8% is a fair comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).parent))
import augment as aug
from train import BATCH, EPOCHS, build_model, close_range, standardise

ROOT = Path(__file__).resolve().parent.parent
SSL = ROOT / "data" / "ssl.npz"
ISL = ROOT / "data" / "dataset_merged.npz"
ENC = ROOT / "models" / "encoder_ssl.weights.h5"
OUT = ROOT / "run" / "ssl_eval.json"

SEED = 0
MASK_FRACTION = 0.33
SSL_EPOCHS, SSL_BATCH = 40, 128
VAL_FRACTION = 0.15
FT_FACTOR = 4


def build_ssl(seq_len: int, n_feat: int) -> keras.Model:
    """build_model's encoder, with a reconstruction head instead of a classifier."""
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

    # --- reconstruction head; discarded after pretraining ---
    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(n_feat, 3, padding="same")(x)

    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def mask_batch(X: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Zero a random third of frames. Returns (corrupted, mask) — target is X."""
    n, t, _ = X.shape
    m = rng.random((n, t)) < MASK_FRACTION
    # never mask a whole clip: with nothing observed there is nothing to infer
    # from, and those rows contribute only noise to the gradient
    empty = m.all(axis=1)
    if empty.any():
        m[empty, rng.integers(0, t, empty.sum())] = False
    Xc = X.copy()
    Xc[m] = 0.0
    return Xc, m


def transfer(src: keras.Model, dst: keras.Model) -> int:
    """Copy layers that match by position and shape. Mismatch = the head."""
    moved = 0
    for a, b in zip(src.layers, dst.layers):
        wa, wb = a.get_weights(), b.get_weights()
        if len(wa) != len(wb) or any(x.shape != z.shape for x, z in zip(wa, wb)):
            continue
        b.set_weights(wa)
        moved += 1
    return moved


def pretrain(rng) -> keras.Model:
    X = np.load(SSL, allow_pickle=True)["X"]
    flat = standardise(X)                       # (N, 32, 195), per-clip
    n_val = max(int(0.05 * len(flat)), 1)
    idx = rng.permutation(len(flat))
    va, tr = idx[:n_val], idx[n_val:]
    print(f"SSL corpus: {len(flat)} windows ({len(tr)} train / {len(va)} val)")

    model = build_ssl(flat.shape[1], flat.shape[2])
    Xtr, Xva = flat[tr], flat[va]

    best, best_w, patience = np.inf, None, 0
    for ep in range(SSL_EPOCHS):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), SSL_BATCH):
            b = Xtr[order[i:i + SSL_BATCH]]
            Xc, m = mask_batch(b, rng)
            # Score ONLY the masked frames. Keras reduces MSE over the feature
            # axis to give one value per timestep, so a (batch, timesteps)
            # sample_weight zeroes out the visible ones. Without this the model
            # wins by copying its input through -- the identity is a perfect
            # score on every frame it can already see, and it would learn
            # nothing about how a hand moves.
            model.train_on_batch(Xc, b, sample_weight=m.astype(np.float32))
        Xc, m = mask_batch(Xva, rng)
        pred = model.predict(Xc, verbose=0)
        vloss = float(((pred - Xva) ** 2)[m].mean())
        if vloss < best - 1e-5:
            best, best_w, patience = vloss, model.get_weights(), 0
        else:
            patience += 1
            if patience >= 6:
                break
        if ep % 5 == 0 or ep == SSL_EPOCHS - 1:
            print(f"  epoch {ep:3d}  masked-MSE {vloss:.4f}", flush=True)
    if best_w:
        model.set_weights(best_w)
    print(f"  best masked-MSE {best:.4f} ({ep + 1} epochs)\n")
    ENC.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(ENC)
    return model


def arm(name, src, X, y, tr_m, te_m, n_classes, rng):
    idx = np.flatnonzero(tr_m)
    rng.shuffle(idx)
    cut = max(int((1 - VAL_FRACTION) * len(idx)), 1)
    core, va = idx[:cut], idx[cut:]
    Xa, ya = aug.augment_batch(X[core], y[core], rng, factor=FT_FACTOR)
    model = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    moved = transfer(src, model) if src is not None else 0
    hist = model.fit(
        standardise(Xa), ya, validation_data=(standardise(X[va]), y[va]),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                              patience=5, min_lr=1e-5),
        ], verbose=0)
    Xc = close_range(X[te_m])
    p = model.predict(standardise(Xc), verbose=0)
    yt = y[te_m]
    t1 = float((p.argmax(1) == yt).mean())
    t5 = float(np.mean([yt[i] in p[i].argsort()[-5:] for i in range(len(yt))]))
    print(f"  {name:22s} top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%"
          f"   ({len(hist.history['loss'])} epochs, {moved} layers)", flush=True)
    keras.backend.clear_session()
    return {"arm": name, "top1": t1, "top5": t5}


def main() -> int:
    if not SSL.exists():
        print(f"missing {SSL.relative_to(ROOT)} — run train/preprocess_ssl.py")
        return 1
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    print("=" * 62)
    print("SELF-SUPERVISED PRETRAINING — masked frame reconstruction")
    print("=" * 62)
    src = pretrain(rng)

    d = np.load(ISL, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    n_classes = len(d["labels"])
    print(f"ISL: {len(X)} clips, {n_classes} classes\n")

    results = []
    for g in (0, 1, 2):
        te = (signer == g) & (corpus == 0)
        tr = ~te
        print(f"held-out INCLUDE group {g} ({int(te.sum())} clips)")
        results.append({**arm("scratch", None, X, y, tr, te, n_classes, rng),
                        "group": int(g)})
        results.append({**arm("ssl-pretrained", src, X, y, tr, te, n_classes, rng),
                        "group": int(g)})

    sc = float(np.mean([r["top1"] for r in results if r["arm"] == "scratch"]))
    ss = float(np.mean([r["top1"] for r in results if r["arm"] == "ssl-pretrained"]))
    print("\n" + "=" * 62)
    print(f"  scratch            mean top-1 {sc*100:5.1f}%")
    print(f"  SSL on ISL         mean top-1 {ss*100:5.1f}%   ({(ss-sc)*100:+.1f})")
    print(f"  shipped (ASL sup.) mean top-1  64.8%")
    print("=" * 62)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"folds": results, "scratch_mean": sc,
                               "ssl_mean": ss, "shipped": 0.648}, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
