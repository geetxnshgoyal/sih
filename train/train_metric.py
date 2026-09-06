"""
Train an encoder for RETRIEVAL, not classification, and measure it on words it
was never taught.

    .venv-tf/bin/python train/train_metric.py --holdout 44
    .venv-tf/bin/python train/train_metric.py --all      # production encoder

Reads  data/dataset_merged.npz   4,894 clips, 264 classes, INCLUDE + CISLR
Writes models/encoder_metric.keras
       run/metric_eval.json

Why this exists
---------------
The shipped model is a closed-set classifier over 83 words. It cannot answer
"pain", "please" or "help" at all, because they are not among its 83 outputs,
and they cannot be added: every source that has them has exactly one clip, and
a class with one example cannot be both taught and examined.

Retrieval sidesteps that. Embed one reference clip per word, embed the live
sign, take the nearest neighbour. Vocabulary then costs one clip per word
instead of fifteen, which is the difference between 83 words and thousands.

Measured with the existing classifier's penultimate layer as the embedding,
that already works better than it has any right to: 48.1% top-1 and 73.1%
top-5 on clinical words the encoder was never trained on. But a classifier's
penultimate layer is not built for this. It is trained to be linearly separable
into N fixed answers, not to put two clips of the same unseen sign near each
other. ArcFace is: it normalises both the embedding and the class weights, so
training optimises cosine angles directly, which is the same quantity retrieval
searches on.

The honest test
---------------
Holding out CLIPS would measure nothing useful, because the class would still
have been trained on. This holds out whole CLASSES. The encoder never sees a
single example of the held-out words, then is asked to match them from one
reference each. That is exactly the situation "pain" is in.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

DATA = ROOT / "data" / "dataset_merged.npz"
SEED = 20260827


class ArcFace:
    """Additive angular margin head. Kept as a builder, not a saved layer, so
    the exported encoder is a plain functional model with no custom objects to
    register at load time."""


def build(seq_len, n_feat, n_classes, dim=256, margin=0.3, scale=30.0):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(seq_len, n_feat), name="clip")
    lab = keras.Input(shape=(), dtype="int32", name="label")

    x = layers.Conv1D(128, 5, padding="same", use_bias=False, name="enc1")(inp)
    x = layers.BatchNormalization(name="enc1_bn")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(256, 5, padding="same", use_bias=False, name="enc2")(x)
    x = layers.BatchNormalization(name="enc2_bn")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, padding="same", use_bias=False, name="enc3")(x)
    x = layers.BatchNormalization(name="enc3_bn")(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    emb = layers.Dense(dim, use_bias=False, name="embedding")(x)
    # UnitNormalization rather than a Lambda: a Lambda cannot infer its output
    # shape on reload, so the saved encoder would not open again
    emb_n = layers.UnitNormalization(name="embedding_l2")(emb)

    encoder = keras.Model(inp, emb_n, name="encoder")

    W = tf.Variable(tf.random.normal([dim, n_classes], stddev=0.05),
                    trainable=True, name="arc_W")

    class Head(layers.Layer):
        def call(self, inputs):
            e, y = inputs
            Wn = tf.math.l2_normalize(W, axis=0)
            cos = tf.clip_by_value(tf.matmul(e, Wn), -1.0 + 1e-7, 1.0 - 1e-7)
            theta = tf.acos(cos)
            oh = tf.one_hot(tf.cast(y, tf.int32), n_classes)
            # the margin is added only on the true class, which is what pushes
            # same-word clips together and different-word clips apart in angle
            target = tf.cos(theta + margin)
            return tf.nn.softmax(scale * (oh * target + (1.0 - oh) * cos))

    out = Head(name="arcface")([emb_n, lab])
    train_model = keras.Model([inp, lab], out)
    train_model.add_weight  # no-op; W is tracked via closure below
    train_model._arc_W = W
    train_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])
    return train_model, encoder


def retrieval(E, y, words, tag):
    """One reference per class, every other clip a query."""
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    classes = sorted(set(y.tolist()))
    ref, qi, qt = [], [], []
    for k, c in enumerate(classes):
        ix = np.flatnonzero(y == c)
        if len(ix) < 2:
            continue
        ref.append(ix[0])
        for i in ix[1:]:
            qi.append(i); qt.append(len(ref) - 1)
    if len(qi) < 10:
        print(f"  {tag}: too few queries")
        return None
    R, Q, T = E[ref], E[qi], np.array(qt)
    S = Q @ R.T
    assert np.all(np.isfinite(S))
    order = np.argsort(-S, axis=1)
    r = {k: float((order[:, :k] == T[:, None]).any(1).mean()) * 100
         for k in (1, 5, 10)}
    print(f"  {tag:34s} bank {len(ref):4d}  n={len(T):5d}  "
          f"top-1 {r[1]:5.1f}%  top-5 {r[5]:5.1f}%  top-10 {r[10]:5.1f}%  "
          f"(chance {100/len(ref):.2f}%)")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=44,
                    help="classes withheld entirely; 0 trains on everything")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--out", default="models/encoder_metric.keras")
    ap.add_argument("--all", action="store_true", help="alias for --holdout 0")
    args = ap.parse_args()
    if args.all:
        args.holdout = 0

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(SEED)

    d = np.load(DATA, allow_pickle=True)
    X, y, labels = d["X"], d["y"], d["labels"]
    X = X.reshape(len(X), X.shape[1], -1).astype(np.float32)
    # the shared contract standardises per clip; do the same here
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / \
        np.maximum(X.std(axis=(1, 2), keepdims=True), 1e-6)
    print(f"{len(X)} clips, {len(labels)} classes, features {X.shape[1:]}")

    rng = np.random.default_rng(SEED)
    counts = np.bincount(y, minlength=len(labels))
    eligible = np.flatnonzero(counts >= 3)
    held = set()
    if args.holdout:
        held = set(rng.choice(eligible, size=min(args.holdout, len(eligible)),
                              replace=False).tolist())
        print(f"holding out {len(held)} classes ENTIRELY (never trained on)")

    tr = np.array([i for i in range(len(y)) if y[i] not in held])
    ho = np.array([i for i in range(len(y)) if y[i] in held])
    keep = sorted(set(y[tr].tolist()))
    remap = {c: i for i, c in enumerate(keep)}
    ytr = np.array([remap[c] for c in y[tr]], np.int32)
    print(f"  train {len(tr)} clips over {len(keep)} classes"
          + (f";  held out {len(ho)} clips over {len(held)} classes" if len(held) else ""))

    model, encoder = build(X.shape[1], X.shape[2], len(keep),
                           margin=args.margin, scale=args.scale)
    cb = [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5,
                                            patience=6, min_lr=1e-5, verbose=0),
          keras.callbacks.EarlyStopping(monitor="loss", patience=15,
                                        restore_best_weights=True, verbose=0)]
    model.fit([X[tr], ytr], ytr, epochs=args.epochs, batch_size=args.batch,
              callbacks=cb, verbose=2)

    print("\n" + "=" * 84)
    print("RETRIEVAL AFTER METRIC TRAINING")
    print("=" * 84)
    res = {}
    Etr = encoder.predict(X[tr], batch_size=256, verbose=0)
    res["seen"] = retrieval(Etr, y[tr], labels, "classes it WAS trained on")
    if len(ho):
        Eho = encoder.predict(X[ho], batch_size=256, verbose=0)
        res["unseen"] = retrieval(Eho, y[ho], labels,
                                  "classes NEVER trained on")

        # the number that matters: the same held-out clips judged against a
        # bank that also contains every trained class, which is what a real
        # dictionary lookup faces
        Eall = encoder.predict(X, batch_size=256, verbose=0)
        res["unseen_full_bank"] = retrieval(Eall, y, labels,
                                            "all 264 in the bank")

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(op)
    print(f"\nsaved encoder to {op.relative_to(ROOT)}")
    (ROOT / "run" / "metric_eval.json").write_text(json.dumps(
        {"holdout": len(held), "train_clips": int(len(tr)),
         "epochs": args.epochs, "margin": args.margin, "scale": args.scale,
         "results": res}, indent=1))
    print("wrote run/metric_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
