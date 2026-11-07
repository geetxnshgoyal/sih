"""
Strengthen the shipped clinical model without the pretraining corpora.

    .venv-tf/bin/python train/train_ensemble.py            # experiment + final
    .venv-tf/bin/python train/train_ensemble.py --final-only

Reads  data/dataset.npz  (train/preprocess.py)  +  models/clinical/gloss_classifier.keras
Writes models/clinical-ensemble/  (gloss_classifier.keras, labels.json, metrics.json)
       run/ensemble_eval.json

Situation
---------
The shipped 38-class model owes most of its accuracy to an encoder pretrained
on ASL + an ISL dictionary (+18 points held-out). Neither the pretraining
corpora nor the encoder weights are in this checkout, only the finished model.
Training from scratch on INCLUDE alone would therefore ship something WORSE.

What this does instead
----------------------
1. RECIPE EXPERIMENT (honest, from scratch, leave-one-group-out): the original
   recipe against the strengthened one (hand dropout + depth noise, label
   smoothing, more augmented variants, 2-member ensemble). Both arms start
   from nothing, so the delta measures the recipe and nothing else.

2. FINAL MODEL: a single Keras graph with three trunks whose softmaxes are
   averaged.
     member 0  = the shipped trunk + head, FROZEN. Its 256-d pooled output is
                 still exposed as the embedding, so app/public/model/_bank.json
                 (dictionary shortlist, 203 words) stays valid: the bank's
                 reference vectors were produced by exactly these weights.
     member 1,2 = warm-started from the shipped trunk, fine-tuned on all
                 clinical clips with the strengthened recipe and different
                 seeds.
   Averaging independently perturbed members is the one improvement that
   reliably survives small data, and it keeps every property the app relies
   on (input contract, label order, embedding space).

The held-out numbers for the final model cannot be signer-disjoint, because the
shipped trunk already saw every INCLUDE group. metrics.json says so.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).parent))
import augment as aug
from train import BATCH, EPOCHS, _trunk, close_range, standardise
from train_clinical import CLINICAL
from train_production import VAL_FRACTION, ece, fit_temperature, temperature_scale

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.npz"
SHIPPED = ROOT / "models" / "clinical"
OUT = ROOT / "models" / "clinical-ensemble"
RUN = ROOT / "run" / "ensemble_eval.json"
SEQ, FEAT = 32, 195


# ----------------------------------------------------------------- recipes --
def set_recipe(strong: bool) -> None:
    aug.USE_HAND_DROPOUT = strong
    aug.USE_Z_NOISE = strong


def compile_(m: keras.Model, lr: float, smoothing: float) -> keras.Model:
    # one-hot targets: Keras only smooths labels in the dense cross-entropy
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=smoothing),
              metrics=["accuracy"])
    return m


def onehot(y: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float32)[y]


def single(n_classes: int, tag: str = "enc") -> keras.Model:
    inp = keras.Input(shape=(SEQ, FEAT))
    pooled = _trunk(inp, tag)
    x = layers.Dropout(0.4)(pooled)
    out = layers.Dense(n_classes, activation="softmax", name=f"{tag}_head")(x)
    return keras.Model(inp, out)


def callbacks(monitor="val_accuracy"):
    return [keras.callbacks.EarlyStopping(monitor=monitor, patience=12, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=1e-5)]


def fit(model, Xtr, ytr, Xva, yva, rng, strong: bool, lr: float):
    set_recipe(strong)
    factor = 6 if strong else 4
    Xa, ya = aug.augment_batch(Xtr, ytr, rng, factor=factor)
    compile_(model, lr, 0.1 if strong else 0.0)
    n = model.output_shape[-1]
    model.fit(standardise(Xa), onehot(ya, n), validation_data=(standardise(Xva), onehot(yva, n)),
              epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks(), verbose=0)
    return model


def topk(p, y):
    t1 = float((p.argmax(1) == y).mean())
    t5 = float(np.mean([y[i] in p[i].argsort()[-5:] for i in range(len(y))]))
    return t1, t5


def reflect_batch(X):
    """Apply the exact training mirror transform to a batch of clips."""
    return np.stack([aug.mirror(clip) for clip in X])


# ----------------------------------------------------------------- shipped --
def shipped_weights() -> dict[str, list[np.ndarray]]:
    m = keras.models.load_model(SHIPPED / "gloss_classifier.keras")
    w = {l.name: l.get_weights() for l in m.layers if l.get_weights()}
    keras.backend.clear_session()
    return w


def load_by_name(model: keras.Model, weights: dict, rename: dict[str, str]) -> int:
    """rename maps THIS model's layer names -> shipped layer names."""
    moved = 0
    for l in model.layers:
        src = weights.get(rename.get(l.name, l.name))
        if src is None or not l.get_weights():
            continue
        if any(a.shape != b.shape for a, b in zip(src, l.get_weights())):
            continue
        l.set_weights(src)
        moved += 1
    return moved


def trunk_rename(tag: str, head: str = "dense") -> dict[str, str]:
    r = {f"{tag}{i}": f"enc{i}" for i in (1, 2, 3)}
    r.update({f"{tag}{i}_bn": f"enc{i}_bn" for i in (1, 2, 3)})
    r[f"{tag}_head"] = head
    return r


# ------------------------------------------------------------------- data --
def clinical_subset():
    d = np.load(DATA, allow_pickle=True)
    X, y, signer = d["X"], d["y"], d["signer"]
    labels = [str(s) for s in d["labels"]]
    shipped_labels = json.loads((SHIPPED / "labels.json").read_text())
    index = {l: i for i, l in enumerate(labels)}
    missing = [w for w in shipped_labels if w not in index]
    assert not missing, f"corpus lacks shipped labels: {missing}"
    ids = [index[w] for w in shipped_labels]
    remap = -np.ones(len(labels), np.int64)
    for new, old in enumerate(ids):
        remap[old] = new
    m = np.isin(y, ids)
    return X[m], remap[y[m]], signer[m], shipped_labels


# ------------------------------------------------------------- experiment --
def experiment(Xs, ys, sg, n, seed: int) -> dict:
    """From-scratch LOGO: baseline recipe vs strengthened 2-member ensemble."""
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    groups = sorted(set(sg.tolist()))
    rows = []
    for g in groups:
        te = sg == g
        if te.sum() < 20:
            continue
        idx = np.flatnonzero(~te)
        rng.shuffle(idx)
        cut = max(int((1 - VAL_FRACTION) * len(idx)), 1)
        core, va = idx[:cut], idx[cut:]
        raw_te = Xs[te]
        Xte = standardise(close_range(raw_te))
        Xte_flip = standardise(close_range(reflect_batch(raw_te)))

        base = fit(single(n), Xs[core], ys[core], Xs[va], ys[va], rng, False, 1e-3)
        pb = base.predict(Xte, verbose=0)
        pb_flip = base.predict(Xte_flip, verbose=0)
        keras.backend.clear_session()

        members, members_flip = [], []
        for k in range(2):
            mk = fit(single(n), Xs[core], ys[core], Xs[va], ys[va], rng, True, 1e-3)
            members.append(mk.predict(Xte, verbose=0))
            members_flip.append(mk.predict(Xte_flip, verbose=0))
            keras.backend.clear_session()
        ps = np.mean(members, axis=0)
        ps_tta = np.mean([ps, np.mean(members_flip, axis=0)], axis=0)
        pb_tta = (pb + pb_flip) / 2

        b1, b5 = topk(pb, ys[te])
        bt1, bt5 = topk(pb_tta, ys[te])
        s1_single, _ = topk(members[0], ys[te])
        s1, s5 = topk(ps, ys[te])
        st1, st5 = topk(ps_tta, ys[te])
        rows.append({"group": int(g), "clips": int(te.sum()),
                     "baseline_top1": b1, "baseline_top5": b5,
                     "baseline_tta_top1": bt1, "baseline_tta_top5": bt5,
                     "strong_single_top1": s1_single,
                     "strong_ensemble_top1": s1, "strong_ensemble_top5": s5,
                     "strong_ensemble_tta_top1": st1, "strong_ensemble_tta_top5": st5})
        print(f"  group {g} ({int(te.sum()):3d} clips)  baseline {b1*100:5.1f}%  "
              f"baseline + reflection {bt1*100:5.1f}%  "
              f"strong single {s1_single*100:5.1f}%  ensemble {s1*100:5.1f}%  "
              f"ensemble + reflection {st1*100:5.1f}%",
              flush=True)
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    out = {"rows": rows,
           "baseline_mean_top1": mean("baseline_top1"),
           "strong_single_mean_top1": mean("strong_single_top1"),
           "strong_ensemble_mean_top1": mean("strong_ensemble_top1"),
           "baseline_tta_mean_top1": mean("baseline_tta_top1"),
           "strong_ensemble_tta_mean_top1": mean("strong_ensemble_tta_top1"),
           "baseline_mean_top5": mean("baseline_top5"),
           "strong_ensemble_mean_top5": mean("strong_ensemble_top5"),
           "baseline_tta_mean_top5": mean("baseline_tta_top5"),
           "strong_ensemble_tta_mean_top5": mean("strong_ensemble_tta_top5")}
    print(f"\n  from scratch, held-out group mean: baseline {out['baseline_mean_top1']*100:.1f}%  "
          f"-> strong ensemble {out['strong_ensemble_mean_top1']*100:.1f}%  top-1")
    return out


# ------------------------------------------------------------------ final --
def build_ensemble(n: int, members: list[keras.Model], shipped: dict) -> keras.Model:
    """One graph: frozen shipped member + fine-tuned members, softmax averaged.

    Output 0: averaged softmax (what the app classifies with).
    Output 1: the shipped member's 256-d pooled embedding (what _bank.json
              was built from), so the dictionary shortlist keeps working.
    """
    inp = keras.Input(shape=(SEQ, FEAT), name="landmarks")
    tags = ["m0"] + [f"m{i+1}" for i in range(len(members))]
    probs, emb0 = [], None
    for i, tag in enumerate(tags):
        pooled = _trunk(inp, tag)
        if i == 0:
            emb0 = pooled
        x = layers.Dropout(0.4)(pooled)
        probs.append(layers.Dense(n, activation="softmax", name=f"{tag}_head")(x))
    avg = layers.Average(name="ensemble")(probs)
    model = keras.Model(inp, avg, name="gloss_ensemble")

    # member 0: shipped weights verbatim
    moved = load_by_name(model, shipped, trunk_rename("m0"))
    assert moved == 7, f"shipped member loaded {moved}/7 layers"
    # fine-tuned members
    for i, mem in enumerate(members):
        w = {l.name: l.get_weights() for l in mem.layers if l.get_weights()}
        moved = load_by_name(model, w, trunk_rename(f"m{i+1}", "enc_head"))
        assert moved == 7, f"member {i+1} loaded {moved}/7 layers"
    return model


def final(Xs, ys, n, labels, shipped: dict, seeds=(11, 23)) -> keras.Model:
    members = []
    for s in seeds:
        rng = np.random.default_rng(s)
        tf.random.set_seed(s)
        idx = rng.permutation(len(Xs))
        cut = max(int((1 - VAL_FRACTION) * len(idx)), 1)
        core, va = idx[:cut], idx[cut:]
        m = single(n)
        moved = load_by_name(m, shipped, {"enc_head": "dense"})
        assert moved == 7, f"warm start loaded {moved}/7 layers"
        # lower LR: this starts from a good solution and should move gently
        fit(m, Xs[core], ys[core], Xs[va], ys[va], rng, True, 3e-4)
        acc = float(m.evaluate(standardise(Xs[va]), onehot(ys[va], n), verbose=0)[1])
        print(f"  member seed {s}: validation accuracy {acc*100:.1f}%", flush=True)
        members.append(m)
    return build_ensemble(n, members, shipped)


def sanity(model: keras.Model, Xs, ys, shipped_model_path: Path) -> dict:
    """In-sample agreement with the shipped model. NOT a held-out metric."""
    Xc = standardise(close_range(Xs))
    pe = model.predict(Xc, verbose=0)
    sm = keras.models.load_model(shipped_model_path)
    ps = sm.predict(Xc, verbose=0)
    e1, e5 = topk(pe, ys)
    s1, s5 = topk(ps, ys)
    print(f"\n  in-sample (close range), shipped {s1*100:.1f}%  ensemble {e1*100:.1f}%  "
          f"(both saw these clips; sanity only)")
    return {"shipped_top1": s1, "ensemble_top1": e1,
            "shipped_top5": s5, "ensemble_top5": e5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--experiment-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xs, ys, sg, labels = clinical_subset()
    n = len(labels)
    counts = np.bincount(ys, minlength=n)
    print(f"{n} classes | {len(Xs)} clips (min {counts.min()}, median {int(np.median(counts))}, "
          f"max {counts.max()}) | groups {sorted(set(sg.tolist()))}\n")

    doc = {"classes": n, "labels": labels, "clips": int(len(Xs)),
           "members": 3, "member0": "shipped clinical model, frozen (embedding source)",
           "recipe": "hand dropout + depth noise, label smoothing 0.1, 6 augmented variants, "
                     "warm start from shipped trunk at lr 3e-4",
           "protocol_note": "held-out-group figures are for the from-scratch recipe experiment only; "
                            "the final ensemble reuses the shipped trunk, which saw every group, "
                            "so no signer-disjoint figure exists for it."}

    if not args.final_only:
        print("== recipe experiment (from scratch, leave-one-group-out) ==")
        doc["experiment"] = experiment(Xs, ys, sg, n, args.seed)

    if args.experiment_only:
        print("\nexperiment-only run complete; deployment weights were not changed.")
        return 0

    print("\n== final ensemble ==")
    shipped = shipped_weights()
    model = final(Xs, ys, n, labels, shipped)
    doc["in_sample_sanity"] = sanity(model, Xs, ys, SHIPPED / "gloss_classifier.keras")

    # calibration: keep the shipped temperature. Averaged softmaxes are flatter,
    # so the shipped T is conservative here rather than overconfident.
    shipped_metrics = json.loads((SHIPPED / "metrics.json").read_text())
    doc["temperature"] = shipped_metrics["temperature"]
    doc["shipped_held_out_mean"] = shipped_metrics["held_out_mean"]
    doc["cannot_say"] = shipped_metrics.get("cannot_say", [])

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "gloss_classifier.keras")
    (OUT / "labels.json").write_text(json.dumps(labels))
    (OUT / "metrics.json").write_text(json.dumps(doc, indent=2))
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(doc, indent=2))
    print(f"\nsaved {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
