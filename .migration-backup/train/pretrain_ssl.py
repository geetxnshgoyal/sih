"""
Self-supervised pretraining on unlabelled Indian Sign Language.

    .venv-tf/bin/python train/pretrain_ssl.py

Reads  data/ssl.npz  +  data/dataset_merged.npz
Writes models/encoder_ssl.weights.h5, run/ssl_eval.json

Two objectives, trained together
--------------------------------
1. MASKED RECONSTRUCTION. Hide a third of the frames and rebuild them. To fill a
   hidden frame the network must learn where a trajectory was heading, which
   joints travel together, what a handshape does between two observed moments.

2. CONTRASTIVE. Two views of the same signing land together; different signing
   lands apart. Positives come from two places: a window augmented two ways, and
   -- where the dictionary has them -- two clips of the SAME entry, signed by
   different people. That second kind is the valuable one: it teaches signer
   invariance directly, which is precisely what the model lacks (a model trained
   on one corpus scores 2.1% on another, ARCHITECTURE.md 5.1).

Neither needs a class label, so the dictionary's fatal flaw as supervised data
-- median ONE clip per word across 12,103 words -- simply does not apply. Then
both heads are thrown away and the encoder is kept.

Why NOT video-to-text contrastive
---------------------------------
GFSLT-VLP (ICCV 2023) aligns video against text, CLIP-style, and that is the
obvious thing to copy. It is wrong here. Sign form is linguistically ARBITRARY:
"cat" and "dog" are close in any text embedding and are signed nothing alike, so
aligning to word semantics would pull visually unrelated movements together and
fight the visual task. It works in GFSLT-VLP because there the text is a
translation of a whole utterance, not a label for one sign. Same-entry positives
give the correspondence we actually want without importing that error.

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
PROJ_DIM = 128
TEMPERATURE = 0.1
LAMBDA_CONTRAST = 0.5   # weight on the contrastive term relative to reconstruction
VAL_FRACTION = 0.15
FT_FACTOR = 4


def build_encoder(seq_len: int, n_feat: int) -> keras.Model:
    """Exactly build_model's layers 0-11, in the same order.

    Kept as its own Model so the weights transfer by POSITION into the
    classifier. Building the two heads inline would interleave their layers into
    model.layers and silently break that alignment -- the copy would still
    report success while putting the wrong tensors in the wrong places.
    """
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
    return keras.Model(inp, x, name="encoder")


def build_ssl(seq_len: int, n_feat: int):
    """encoder -> (reconstruction, projection). Both heads are discarded after."""
    enc = build_encoder(seq_len, n_feat)
    inp = keras.Input(shape=(seq_len, n_feat))
    h = enc(inp)

    recon = layers.Conv1D(n_feat, 3, padding="same", name="recon")(
        layers.UpSampling1D(2)(h))

    # Projection head, as in SimCLR: contrast in a space separate from the one
    # that gets transferred, so the encoder is not forced to discard information
    # the contrastive task happens not to need.
    z = layers.GlobalAveragePooling1D()(h)
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Dense(PROJ_DIM, name="proj")(z)
    return keras.Model(inp, [recon, z], name="ssl"), enc


def mask_batch(X: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Zero a random third of frames. Returns (corrupted, mask), target is X."""
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


def two_views(X4: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Two independently augmented versions of the same clips, (N,T,65,3) each.

    The augmentations ARE the invariances we want: mirroring covers handedness,
    perspective covers camera distance, rotation covers how squarely someone sits.
    Contrasting two views teaches the encoder to ignore exactly those, which is
    the axis it currently fails on.
    """
    def one(A):
        A = A.copy()
        if aug.USE_MIRROR and rng.random() < 0.5:
            A = aug.mirror(A.reshape(-1, *A.shape[2:])).reshape(A.shape)
        for j in range(len(A)):
            a = A[j]
            if aug.USE_PERSPECTIVE:
                a = aug.perspective(a, rng)
            if aug.USE_ROTATE:
                a = aug.rotate(a, rng)
            if aug.USE_TIME_MASK and rng.random() < 0.5:
                a = aug.time_mask(a, rng)
            A[j] = a
        return A
    return one(X4), one(X4)


def positive_mask(n, word):
    """(2n, 2n) bool: which pairs among the two views count as positives.

    Two sources, unioned:
      - a window and its own other view          (i, i+n)
      - any two windows of the same dictionary entry, including across clips,
        which is what teaches signer invariance

    Split out from nt_xent so it can be asserted directly. Inferring it from the
    loss value does not work: when positives happen to be similar anyway, adding
    more of them barely moves the number, so a broken mask looks fine.
    """
    w = tf.concat([word, word], axis=0)
    same_word = tf.equal(w[:, None], w[None, :])
    idx = tf.range(2 * n)
    partner = tf.equal(tf.math.floormod(idx[:, None], n),
                       tf.math.floormod(idx[None, :], n))
    return tf.logical_or(same_word, partner)


def nt_xent(z1, z2, word, temperature=TEMPERATURE):
    """NT-Xent over 2N views, with same-entry pairs counted as positives too.

    Standard SimCLR treats only (i, i+N) as positive. Here two windows of the
    same dictionary entry are ALSO positive, so two people signing the same word
    are pulled together -- signer invariance, taught directly rather than hoped
    for. A row with several positives is scored against all of them.
    """
    z = tf.math.l2_normalize(tf.concat([z1, z2], axis=0), axis=1)
    n = tf.shape(z1)[0]
    sim = tf.matmul(z, z, transpose_b=True) / temperature
    # a view is never its own positive
    eye = tf.eye(2 * n, dtype=tf.bool)
    sim = tf.where(eye, tf.fill(tf.shape(sim), -1e9), sim)

    pos = tf.logical_and(positive_mask(n, word), tf.logical_not(eye))

    log_prob = sim - tf.math.reduce_logsumexp(sim, axis=1, keepdims=True)
    pos_f = tf.cast(pos, log_prob.dtype)
    n_pos = tf.reduce_sum(pos_f, axis=1)
    # a row with no positive contributes nothing rather than a division by zero
    safe = tf.maximum(n_pos, 1.0)
    per_row = tf.reduce_sum(log_prob * pos_f, axis=1) / safe
    return -tf.reduce_sum(per_row * tf.cast(n_pos > 0, per_row.dtype)) / \
        tf.maximum(tf.reduce_sum(tf.cast(n_pos > 0, per_row.dtype)), 1.0)


def pretrain(rng) -> keras.Model:
    d = np.load(SSL, allow_pickle=True)
    X4 = d["X"]
    word = d["word"] if "word" in d else np.arange(len(X4), dtype=np.int32)
    n_val = max(int(0.05 * len(X4)), 1)
    idx = rng.permutation(len(X4))
    va, tr = idx[:n_val], idx[n_val:]
    multi = int((np.bincount(word) >= 2).sum())
    print(f"SSL corpus: {len(X4)} windows ({len(tr)} train / {len(va)} val)")
    print(f"  {multi} entries have 2+ windows -> cross-clip positives available")

    model, enc = build_ssl(X4.shape[1], X4.shape[2] * X4.shape[3])
    opt = keras.optimizers.Adam(1e-3)

    @tf.function
    def step(v1, v2, target, mask, w):
        with tf.GradientTape() as tape:
            r1, z1 = model(v1, training=True)
            _, z2 = model(v2, training=True)
            # reconstruction is scored ONLY where frames were hidden; including
            # visible frames lets the model win with the identity and learn
            # nothing about how a hand moves
            se = tf.reduce_mean(tf.square(r1 - target), axis=-1)
            m = tf.cast(mask, se.dtype)
            recon = tf.reduce_sum(se * m) / tf.maximum(tf.reduce_sum(m), 1.0)
            contrast = nt_xent(z1, z2, w)
            loss = recon + LAMBDA_CONTRAST * contrast
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                model.trainable_variables))
        return recon, contrast

    best, best_w, patience, ep = np.inf, None, 0, 0
    for ep in range(SSL_EPOCHS):
        order = rng.permutation(len(tr))
        for i in range(0, len(order), SSL_BATCH):
            b = tr[order[i:i + SSL_BATCH]]
            if len(b) < 8:
                continue                      # too few for a contrastive batch
            a1, a2 = two_views(X4[b], rng)
            f1, f2 = standardise(a1), standardise(a2)
            corrupt, mask = mask_batch(f1, rng)
            step(tf.constant(corrupt), tf.constant(f2), tf.constant(f1),
                 tf.constant(mask), tf.constant(word[b]))

        fv = standardise(X4[va])
        cv, mv = mask_batch(fv, rng)
        pred, _ = model.predict(cv, verbose=0)
        vloss = float(((pred - fv) ** 2).mean(axis=-1)[mv].mean())
        if vloss < best - 1e-5:
            best, best_w, patience = vloss, model.get_weights(), 0
        else:
            patience += 1
            if patience >= 6:
                break
        if ep % 5 == 0:
            print(f"  epoch {ep:3d}  masked-MSE {vloss:.4f}", flush=True)
    if best_w:
        model.set_weights(best_w)
    print(f"  best masked-MSE {best:.4f} ({ep + 1} epochs)\n")
    ENC.parent.mkdir(parents=True, exist_ok=True)
    enc.save_weights(ENC)
    return enc


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
        print(f"missing {SSL.relative_to(ROOT)}, run train/preprocess_ssl.py")
        return 1
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    print("=" * 62)
    print("SELF-SUPERVISED PRETRAINING, masked frame reconstruction")
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
