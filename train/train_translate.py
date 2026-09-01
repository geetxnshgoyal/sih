"""
Sign language TRANSLATION: landmark sequence in, English sentence out.

    .venv-tf/bin/python train/train_translate.py

Reads  data/translate.npz
Writes models/translate/, run/translate_eval.json

What is different about this model
----------------------------------
Everything else in this repo is a CLASSIFIER: one clip, one of N labels. It
structurally cannot produce a sentence, no matter how it is trained, because its
output is a softmax over a fixed vocabulary of signs.

This is an encoder-decoder. The encoder reads a variable-length landmark
sequence; the decoder emits words one at a time, attending back over the whole
sequence. Word ORDER is now part of the output, so nothing may pool time away.

    landmarks (T, 195) -> Conv1D downsample -> transformer encoder
                                                     |
                       words <- transformer decoder <-+

The Conv1D front end is not decoration. Attention is quadratic in sequence
length and these are up to 160 frames, so two strided convolutions cut that to
40 before any attention runs, and they also reuse the shape of the encoder that
already works for isolated signs.

Honest expectations
-------------------
Sign language translation is hard and the published numbers are low: GFSLT-VLP
reports BLEU-4 around 21 on PHOENIX14T with 8,257 pairs and four GPUs. On a few
hundred pairs and a CPU this will produce something close to noise, and that is
the expected result rather than a failure. The point of building it now is that
the pipeline is ready when the corpus finishes, and BLEU on a held-out set is
the only way to know whether more data is helping.

Validation holds out whole BULLETINS, never individual cues: two cues from one
bulletin share a signer, a room and a camera.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "translate.npz"
OUT_DIR = ROOT / "models" / "translate"
RUN = ROOT / "run" / "translate_eval.json"

PAD, BOS, EOS, UNK = 0, 1, 2, 3
D_MODEL, HEADS, FF, LAYERS, DROPOUT = 192, 4, 384, 2, 0.15
EPOCHS, BATCH, LR = 120, 16, 3e-4
SEED = 0


def positional(length: int, depth: int) -> np.ndarray:
    pos = np.arange(length)[:, None]
    i = np.arange(depth // 2)[None, :]
    ang = pos / np.power(10000.0, (2 * i) / depth)
    pe = np.zeros((length, depth), np.float32)
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    return pe


def build(src_len: int, tgt_len: int, n_vocab: int) -> keras.Model:
    src = keras.Input(shape=(src_len, 195), name="src")
    tgt = keras.Input(shape=(tgt_len,), dtype="int32", name="tgt")

    # Downsample before attention: 160 frames -> 40, so self-attention runs on a
    # quarter of the length and a sixteenth of the cost.
    x = layers.Conv1D(D_MODEL, 5, strides=2, padding="same", use_bias=False)(src)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(D_MODEL, 5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    enc_len = src_len // 4
    x = x + positional(enc_len, D_MODEL)[None]

    for _ in range(LAYERS):
        a = layers.MultiHeadAttention(HEADS, D_MODEL // HEADS, dropout=DROPOUT)(x, x)
        x = layers.LayerNormalization()(x + a)
        f = layers.Dense(FF, activation="relu")(x)
        f = layers.Dense(D_MODEL)(f)
        x = layers.LayerNormalization()(x + layers.Dropout(DROPOUT)(f))
    memory = x

    y = layers.Embedding(n_vocab, D_MODEL, mask_zero=False)(tgt)
    y = y + positional(tgt_len, D_MODEL)[None]
    for _ in range(LAYERS):
        # Causal self-attention: a decoder must not read the word it is about to
        # predict. use_causal_mask does that; forgetting it produces a model
        # that scores brilliantly in training and emits nothing at inference.
        s = layers.MultiHeadAttention(HEADS, D_MODEL // HEADS, dropout=DROPOUT)(
            y, y, use_causal_mask=True)
        y = layers.LayerNormalization()(y + s)
        c = layers.MultiHeadAttention(HEADS, D_MODEL // HEADS, dropout=DROPOUT)(y, memory)
        y = layers.LayerNormalization()(y + c)
        f = layers.Dense(FF, activation="relu")(y)
        f = layers.Dense(D_MODEL)(f)
        y = layers.LayerNormalization()(y + layers.Dropout(DROPOUT)(f))

    out = layers.Dense(n_vocab, name="logits")(y)
    m = keras.Model([src, tgt], out)
    m.compile(
        optimizer=keras.optimizers.Adam(LR),
        # Padding contributes no loss; without masking, a model that predicts
        # <pad> everywhere would look excellent.
        loss=masked_loss, metrics=[masked_acc],
    )
    return m


def masked_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD), tf.float32)
    ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.reduce_sum(ce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def masked_acc(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD), tf.float32)
    hit = tf.cast(tf.equal(tf.cast(y_true, tf.int64),
                           tf.argmax(y_pred, axis=-1)), tf.float32)
    return tf.reduce_sum(hit * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def greedy(model, src, tgt_len: int) -> np.ndarray:
    """Decode one token at a time, feeding predictions back in."""
    n = len(src)
    out = np.zeros((n, tgt_len), np.int32)
    out[:, 0] = BOS
    for t in range(1, tgt_len):
        logits = model.predict([src, out], verbose=0)
        out[:, t] = logits[:, t - 1].argmax(-1)
        if np.all((out[:, t] == EOS) | (out[:, t] == PAD)):
            break
    return out


def bleu(refs: list[list[str]], hyps: list[list[str]], n_max: int = 4) -> list[float]:
    """Corpus BLEU-1..n with the standard brevity penalty."""
    from collections import Counter
    scores = []
    for n in range(1, n_max + 1):
        num = den = 0
        for ref, hyp in zip(refs, hyps):
            hn = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            rn = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            num += sum(min(c, rn[g]) for g, c in hn.items())
            den += max(sum(hn.values()), 0)
        p = num / den if den else 0.0
        rl = sum(len(r) for r in refs)
        hl = sum(len(h) for h in hyps)
        bp = 1.0 if hl > rl else np.exp(1 - rl / max(hl, 1))
        scores.append(float(bp * p))
    return scores


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA.relative_to(ROOT)} — run train/preprocess_sentences.py")
        return 1
    d = np.load(DATA, allow_pickle=True)
    src, tgt, split = d["src"], d["tgt"], d["split"]
    vocab = [str(s) for s in d["vocab"]]
    tf.random.set_seed(SEED)

    tr, va = split == 0, split == 1
    print(f"{len(src)} pairs | vocab {len(vocab)} | {tr.sum()} train / {va.sum()} val")
    if va.sum() < 8 or tr.sum() < 32:
        print("too few pairs to train or measure; let the corpus build further")
        return 1

    # The decoder is fed tgt[:, :-1] and predicts tgt[:, 1:], so it is built one
    # position shorter than the stored sequence. Building it at the full length
    # is a shape error at the first batch, which is the good kind of mistake.
    dec_len = tgt.shape[1] - 1
    model = build(src.shape[1], dec_len, len(vocab))
    print(f"parameters: {model.count_params():,}\n")

    # teacher forcing: the decoder reads tgt[:-1] and predicts tgt[1:]
    hist = model.fit(
        [src[tr], tgt[tr][:, :-1]], tgt[tr][:, 1:],
        validation_data=([src[va], tgt[va][:, :-1]], tgt[va][:, 1:]),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=6, min_lr=1e-5),
        ], verbose=2)

    pred = greedy(model, src[va], dec_len)
    def words(seq):
        out = []
        for t in seq:
            if t in (PAD, BOS): continue
            if t == EOS: break
            out.append(vocab[t] if t < len(vocab) else "<unk>")
        return out
    refs = [words(r) for r in tgt[va]]
    hyps = [words(r) for r in pred]
    b = bleu(refs, hyps)
    print("\n" + "=" * 58)
    for i, s in enumerate(b, 1):
        print(f"  BLEU-{i}  {s*100:5.2f}")
    print(f"  (GFSLT-VLP reports BLEU-4 ~21 on 8,257 German pairs, 4 GPUs)")
    print("=" * 58)
    print("\n  samples:")
    for i in range(min(4, len(refs))):
        print(f"    ref: {' '.join(refs[i])[:74]}")
        print(f"    got: {' '.join(hyps[i])[:74] or '(empty)'}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "translate.keras")
    (OUT_DIR / "vocab.json").write_text(json.dumps(vocab))
    doc = {"pairs": int(len(src)), "train": int(tr.sum()), "val": int(va.sum()),
           "vocab": len(vocab), "params": int(model.count_params()),
           "bleu": {f"bleu{i}": s for i, s in enumerate(b, 1)},
           "epochs": len(hist.history["loss"]),
           "val_token_acc": float(max(hist.history.get("val_masked_acc", [0])))}
    (OUT_DIR / "metrics.json").write_text(json.dumps(doc, indent=2))
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(doc, indent=2))
    print(f"saved {OUT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
