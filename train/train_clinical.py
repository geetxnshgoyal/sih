"""
Train the clinical vocabulary: fewer signs, each one worth saying.

    .venv-tf/bin/python train/train_clinical.py

Reads  data/dataset_merged.npz  +  models/encoder_pretrain.weights.h5
Writes models/clinical/  (gloss_classifier.keras, labels.json, metrics.json)
       run/clinical_eval.json

Why a smaller vocabulary
------------------------
Measured on held-out INCLUDE group 0, holding data, recipe and protocol fixed
and varying only the number of classes (run/vocab_eval.json):

    classes    top-1    top-5
       20      76.1%    94.8%
       80      66.0%    92.8%
      264      56.1%    81.4%

264 classes costs about ten points of top-1 and thirteen of top-5. Each extra
class is another way to be wrong across ~19 clips from about ten signers, and
most of them are words a clinic will never need. Yellow, Grey and Thursday are
not worth the accuracy they take from Doctor.

What this vocabulary CANNOT say
-------------------------------
INCLUDE is a general-purpose corpus, so the most important clinical words are
simply not in it at any count:

    pain  water  help  yes  no  please  fever  breathe  emergency  head  stomach

Those stay on the phrase board, which is exact, offline and needs no model. This
is the honest division of labour: the board carries what must never be wrong,
and recognition carries what is convenient. Anyone reading a headline accuracy
number for this model should read that list first.
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
from train_production import VAL_FRACTION, ece, fit_temperature, temperature_scale

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_merged.npz"
# Stacked encoder: SSL on the ISL dictionary, THEN supervised ASL. Measured on
# this same vocabulary, identical arms, held-out signer (run/stacked_eval.json):
#
#     none      54.5%      asl       69.1%   <- what shipped
#     ssl       52.8%      stacked   72.2%   (+3.1)
#
# The two signals compound rather than one overwriting the other: starting the
# ASL stage from a representation that already knows how Indian hands move
# produces a better ASL encoder, and that survives fine-tuning. It is also the
# only use in which the ISL dictionary paid for itself -- alone it was worth
# +1.3, as an initialiser it is worth +3.1 on top of ASL.
ENCODER = ROOT / "models" / "encoder_stacked.weights.h5"
ENCODER_FALLBACK = ROOT / "models" / "encoder_pretrain.weights.h5"
OUT_DIR = ROOT / "models" / "clinical"
RUN = ROOT / "run" / "clinical_eval.json"
SEED = 0
FACTOR = 4

# Chosen for what a consultation needs, then filtered to what the data actually
# supports. Grouped so the gaps are visible rather than buried.
CLINICAL = [
    # who is speaking, and about whom
    "I", "you", "he", "she", "we", "they",
    # opening and closing a conversation
    "Hello", "Thank you", "How are you", "Alright", "Pleased",
    # the two words that carry most of a triage answer
    "good", "bad",
    # state of health
    "sick", "healthy", "weak", "strong", "alive", "Death",
    # who and where
    "Doctor", "Hospital", "Medicine", "Patient", "Deaf", "Blind",
    # sensation, the closest this corpus gets to a symptom
    "hot", "cold", "warm", "cool",
    # when: onset and duration are most of a history
    "Today", "Tomorrow", "Yesterday", "Time", "Morning", "Evening", "Night",
    # where in a building, for orientation and mobility
    "Bathroom", "Bedroom",
]


def warm_start(model, n_classes) -> int:
    """Copy the pretrained encoder in, skipping the classifier head.

    Matched by position and shape; a mismatch is the head, which is exactly what
    we want to skip. Both candidate encoders were trained with a 2,186-class
    head, so that is the shape to rebuild before loading.
    """
    src_path = ENCODER if ENCODER.exists() else ENCODER_FALLBACK
    if not src_path.exists():
        return 0
    src = build_model(32, 195, 2186)
    try:
        src.load_weights(src_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not load {src_path.name}: {exc}")
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


def main() -> int:
    d = np.load(DATA, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    labels = [str(s) for s in d["labels"]]
    index = {l: i for i, l in enumerate(labels)}
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)
    which = ENCODER if ENCODER.exists() else ENCODER_FALLBACK
    print(f"encoder: {which.name}")

    keep, missing = [], []
    for w in CLINICAL:
        (keep if w in index else missing).append(w)
    if missing:
        print(f"not in the corpus, dropped: {', '.join(missing)}")
    ids = [index[w] for w in keep]
    remap = -np.ones(len(labels), np.int64)
    for new, old in enumerate(ids):
        remap[old] = new

    m = np.isin(y, ids)
    Xs, ys, sg, cp = X[m], remap[y[m]], signer[m], corpus[m]
    counts = np.bincount(ys, minlength=len(keep))
    print(f"\n{len(keep)} classes | {len(Xs)} clips "
          f"(min {counts.min()}, median {int(np.median(counts))}, max {counts.max()})")
    print(f"dropped {len(X) - len(Xs)} clips of the other {len(labels) - len(keep)} classes\n")

    # ---- leave-one-group-out: honest accuracy AND out-of-fold predictions ----
    groups = sorted(set(sg.tolist()))
    oof = np.zeros((len(Xs), len(keep)), np.float32)
    seen = np.zeros(len(Xs), bool)
    per = []
    for g in groups:
        te = sg == g
        if te.sum() < 20:
            continue
        tr = ~te
        idx = np.flatnonzero(tr)
        rng.shuffle(idx)
        cut = max(int((1 - VAL_FRACTION) * len(idx)), 1)
        core, va = idx[:cut], idx[cut:]
        Xa, ya = aug.augment_batch(Xs[core], ys[core], rng, factor=FACTOR)
        model = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], len(keep))
        warm_start(model, len(keep))
        model.fit(standardise(Xa), ya,
                  validation_data=(standardise(Xs[va]), ys[va]),
                  epochs=EPOCHS, batch_size=BATCH,
                  callbacks=[
                      keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                                    restore_best_weights=True),
                      keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                                        factor=0.5, patience=5,
                                                        min_lr=1e-5),
                  ], verbose=0)
        p = model.predict(standardise(close_range(Xs[te])), verbose=0)
        oof[te], seen[te] = p, True
        t1 = float((p.argmax(1) == ys[te]).mean())
        t5 = float(np.mean([ys[te][i] in p[i].argsort()[-5:] for i in range(int(te.sum()))]))
        src = "INCLUDE" if cp[te][0] == 0 else "CISLR"
        per.append({"group": int(g), "corpus": src, "clips": int(te.sum()),
                    "top1": t1, "top5": t5})
        print(f"  group {g} ({src:7s}, {int(te.sum()):4d} clips)  "
              f"top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%", flush=True)
        keras.backend.clear_session()

    inc = [r for r in per if r["corpus"] == "INCLUDE"]
    m1 = float(np.mean([r["top1"] for r in inc]))
    m5 = float(np.mean([r["top5"] for r in inc]))
    print(f"\n  INCLUDE mean  top-1 {m1*100:5.1f}%  top-5 {m5*100:5.1f}%")
    print(f"  264-class model on the same protocol: 64.8% top-1, 86.6% top-5")

    # ---- calibrate on the out-of-fold predictions ----
    p, yy = oof[seen], ys[seen]
    T = fit_temperature(p, yy)
    before, after = ece(p, yy), ece(temperature_scale(p, T), yy)
    cal = temperature_scale(p, T)
    print(f"\n  temperature {T:.2f}   ECE {before*100:.1f}pp -> {after*100:.1f}pp")
    bands = []
    for floor in (0.30, 0.40, 0.50, 0.70, 0.90):
        sel = cal.max(1) >= floor
        rate = float(sel.mean())
        acc = float((cal[sel].argmax(1) == yy[sel]).mean()) if sel.any() else 0.0
        bands.append({"floor": floor, "speaks": rate, "accuracy": acc})
        print(f"    floor {floor:.2f}  speaks {rate*100:4.1f}%  right {acc*100:5.1f}%")

    # ---- final model on everything ----
    print("\n  final model, all groups...", flush=True)
    idx = rng.permutation(len(Xs))
    cut = max(int((1 - VAL_FRACTION) * len(idx)), 1)
    core, va = idx[:cut], idx[cut:]
    Xa, ya = aug.augment_batch(Xs[core], ys[core], rng, factor=FACTOR)
    final = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], len(keep))
    warm_start(final, len(keep))
    final.fit(standardise(Xa), ya,
              validation_data=(standardise(Xs[va]), ys[va]),
              epochs=EPOCHS, batch_size=BATCH,
              callbacks=[
                  keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                                restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                                    patience=5, min_lr=1e-5),
              ], verbose=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final.save(OUT_DIR / "gloss_classifier.keras")
    (OUT_DIR / "labels.json").write_text(json.dumps(keep))
    doc = {"classes": len(keep), "labels": keep, "clips": int(len(Xs)),
           "encoder": which.name,
           "held_out_group": per, "held_out_mean": {"top1": m1, "top5": m5},
           "temperature": T, "ece_before": before, "ece_after": after,
           "bands": bands, "cannot_say": ["pain", "water", "help", "yes", "no",
                                          "please", "fever", "breathe"]}
    (OUT_DIR / "metrics.json").write_text(json.dumps(doc, indent=2))
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(doc, indent=2))
    print(f"\nsaved models/clinical/  (temperature {T:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
