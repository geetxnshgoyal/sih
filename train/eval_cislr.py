"""
Does adding CISLR signers actually help? A controlled answer.

    .venv-tf/bin/python train/eval_cislr.py

Reads  data/dataset_merged.npz    (train/preprocess_cislr.py)
Writes run/cislr_eval.json

Why this exists
---------------
Every previous experiment — face mesh, SL-GCN, calibration — pointed at the
same conclusion: the ceiling is signer diversity, not architecture. CISLR is
the first chance to test that claim directly, because it is the first data we
have from people who are not the seven students INCLUDE recorded.

Four arms, all the same model, the same augmentation and the same epoch budget,
differing ONLY in which clips are in the training set:

  A  INCLUDE          -> INCLUDE group 0      the standing baseline
  B  INCLUDE + CISLR  -> INCLUDE group 0      does adding signers move it?
  C  INCLUDE          -> CISLR                what happens on a wholly unseen
                                              corpus: new people, new rooms,
                                              new cameras. The wild number.
  D  INCLUDE + CISLR' -> CISLR held-out group does training on SOME of a corpus
                                              transfer to the rest of it?

A vs B is the decisive comparison and the reason for the whole exercise. C is
the number to quote when anyone asks what this does outside its own dataset;
expect it to be low, and report it anyway.

Protocol note
-------------
train.py passes the TEST set as validation_data with restore_best_weights, so
the stopping epoch is chosen on the test set — that inflates every figure it
prints, including the 40.4% baseline. Here validation is carved out of TRAIN by
signer group where possible, so the test set is touched exactly once, at the
end. Arm A is therefore NOT directly comparable to the published 40.4%; it is
the honest re-measurement of it, and B must be compared against A, not against
the old number.
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
DATA = ROOT / "data" / "dataset_merged.npz"
OUT = ROOT / "run" / "cislr_eval.json"
SEED = 0

INCLUDE, CISLR = 0, 1


def fit(Xtr, ytr, Xva, yva, n_classes, rng):
    """Validation comes from TRAIN. The test set is never seen during fitting."""
    Xa, ya = aug.augment_batch(Xtr, ytr, rng, factor=4)
    model = build_model(Xa.shape[1], Xa.shape[2] * Xa.shape[3], n_classes)
    hist = model.fit(
        standardise(Xa), ya,
        validation_data=(standardise(Xva), yva),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                              patience=5, min_lr=1e-5),
        ],
        verbose=0,
    )
    return model, len(hist.history["loss"])


def score(model, X4, y, n_classes):
    p = model.predict(standardise(X4), verbose=0)
    top1 = float((p.argmax(1) == y).mean())
    top5 = float(np.mean([y[i] in p[i].argsort()[-5:] for i in range(len(y))]))
    return top1, top5


def arm(name, desc, X, y, tr_m, te_m, va_group, signer, n_classes, rng):
    """One arm. va_group picks the training signer used for validation."""
    va_m = tr_m & (signer == va_group)
    core_m = tr_m & ~va_m
    if core_m.sum() < 200 or va_m.sum() < 30:      # fall back to a random slice
        idx = np.flatnonzero(tr_m)
        rng.shuffle(idx)
        cut = int(0.85 * len(idx))
        core_m = np.zeros(len(X), bool); core_m[idx[:cut]] = True
        va_m = np.zeros(len(X), bool); va_m[idx[cut:]] = True

    testable = len(set(y[core_m].tolist()) & set(y[te_m].tolist()))
    print("=" * 66)
    print(f"ARM {name} — {desc}")
    print(f"  train {core_m.sum()}  val {va_m.sum()}  test {te_m.sum()}"
          f"   ({testable} of {n_classes} test classes seen in training)")
    model, ep = fit(X[core_m], y[core_m], X[va_m], y[va_m], n_classes, rng)
    t1, t5 = score(model, X[te_m], y[te_m], n_classes)
    c1, c5 = score(model, close_range(X[te_m]), y[te_m], n_classes)
    print(f"  far   top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%   ({ep} epochs)")
    print(f"  close top-1 {c1*100:5.1f}%  top-5 {c5*100:5.1f}%\n", flush=True)
    return {"arm": name, "desc": desc, "train": int(core_m.sum()),
            "test": int(te_m.sum()), "testable_classes": testable,
            "far_top1": t1, "far_top5": t5, "close_top1": c1, "close_top5": c5,
            "epochs": ep}


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

    inc, cis = corpus == INCLUDE, corpus == CISLR
    cis_groups = sorted(set(signer[cis].tolist()))
    print(f"{len(X)} clips | {n_classes} classes")
    print(f"  INCLUDE {inc.sum()} (groups {sorted(set(signer[inc].tolist()))})")
    print(f"  CISLR   {cis.sum()} (groups {cis_groups})\n")

    held = cis_groups[-1]           # one CISLR group reserved for arm D
    results = [
        arm("A", "INCLUDE -> INCLUDE group 0   (baseline)",
            X, y, inc & (signer != 0), inc & (signer == 0), 1, signer, n_classes, rng),
        arm("B", "INCLUDE + CISLR -> INCLUDE group 0   (does it help?)",
            X, y, (inc & (signer != 0)) | cis, inc & (signer == 0), 1, signer, n_classes, rng),
        arm("C", "INCLUDE -> CISLR   (unseen corpus: the wild number)",
            X, y, inc, cis, 2, signer, n_classes, rng),
        arm("D", f"INCLUDE + CISLR -> CISLR group {held}   (within-corpus transfer)",
            X, y, inc | (cis & (signer != held)), cis & (signer == held),
            cis_groups[0], signer, n_classes, rng),
    ]

    print("=" * 66)
    print("SUMMARY  (close-range — the condition the app runs in)")
    print("=" * 66)
    for r in results:
        print(f"  {r['arm']}  {r['close_top1']*100:5.1f}%  top-5 {r['close_top5']*100:5.1f}%"
              f"   {r['desc']}")
    delta = (results[1]["close_top1"] - results[0]["close_top1"]) * 100
    print(f"\n  B - A = {delta:+.1f} points."
          f"  {'CISLR signers help.' if delta > 1 else 'No measurable gain.'}")
    print(f"  Arm C is what this model does on data it has never seen the like of.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"classes": n_classes, "arms": results}, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
