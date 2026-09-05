"""
Three-arm ablation: does the face mesh help, and does graph structure help?

    .venv-tf/bin/python train/train_ablation.py

Reads  data/dataset_face.npz   (X_head 65pt, X_full 113pt, same clips)
Writes models/ablation_face.json

The arms
--------
  HEAD_ONLY   65 points, Conv1D over a flattened 195-vector   (the baseline)
  FULL_FACE   113 points, same Conv1D                         (does the face help?)
  SL-GCN      65 points as a skeleton GRAPH                   (does structure help?)

Two independent questions, deliberately in one harness so they share the exact
same clips, signers, classes, seed, augmentation draws and held-out-group
protocol. Only one thing varies per comparison.

Q1: the face. `FACE_MODE = "FULL_FACE"` adds 48 eyebrow, eye-aperture and lip
landmarks. ISL marks yes/no questions with raised brows, wh-questions with
furrowed brows, and negation with a head shake, so there is a linguistic reason
to expect a gain. That is an argument, not evidence, which is why it is measured.

Q2: the architecture. The baseline flattens each frame's landmarks into a
195-vector, discarding the fact that they form a skeleton: nothing tells it that
wrist-elbow-shoulder are connected. SL-GCN convolves ALONG the graph instead.
AI4Bharat benchmarked SL-GCN at 93.5% on INCLUDE against 91.2% for ST-GCN and
90.4% for a transformer, so on this exact dataset it is the best-evidenced
architecture available.

Held at 0.90x the baseline's parameter count (461,830 vs 512,994) on purpose. At
the natural width it carries 2.39x, and a win there would be uninterpretable , 
"structure helps" and "capacity helps" would be perfectly confounded.

Epoch budgets are per-architecture: see ARCH_EPOCHS below.

Scope, stated plainly
---------------------
This runs on whatever Zenodo parts have been extracted so far, the scope is
read from disk at runtime and written into the output JSON, because a hardcoded
scope string silently becomes a lie as more data lands. It is NOT the 4,284-clip
/ 264-class result and must never be quoted as one.

The verdict rule, for both questions: if the mean delta is smaller than the
spread between signer groups, the answer is INCONCLUSIVE. With three groups
there is no power to call a small difference, and saying otherwise would be the
overclaiming this project avoids everywhere else (HANDOFF.md §6).
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
import augment as aug  # noqa: E402
from train import EPOCHS, BATCH, SEED, build_model, standardise, close_range, score  # noqa: E402
from slgcn import build_slgcn, standardise_graph  # noqa: E402

# Epoch budget is PER ARCHITECTURE, not shared.
#
# EPOCHS=60 is tuned for the Conv1D baseline. SL-GCN converges far more slowly , 
# measured on a 20-class slice, it read 12.8% at 6 epochs and reached 100% train
# / 33% val by 40. Running both arms at 60 would compare a converged model against
# an undertrained one and conclude "graphs do not help", which would be a fact
# about the epoch budget rather than about the architecture.
#
# Early stopping (patience 12, restore_best_weights) still governs when each arm
# actually stops, so the larger budget is a CEILING, not a mandate to overfit.
# The literature agrees on the direction: SL-GCN trains ~4x slower than ST-GCN.
ARCH_EPOCHS = {"cnn": EPOCHS, "slgcn": 180}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_face.npz"
OUT = ROOT / "models" / "ablation_face.json"

# Where ISL puts its non-manual grammar: questions on the eyebrows, negation on a
# head shake. These are the categories that can actually exercise the face block,
# and the reason fetch_video.sh pulls them first.
NON_MANUAL_CATEGORIES = {"Pronouns", "Society"}


def categories_present() -> set[str]:
    """Categories actually extracted, read from disk.

    The scope of this ablation changes every time another part lands, so it must
    be observed rather than hardcoded, a stale scope string turns an honest
    result into a misleading one, and this file's whole job is honest reporting.
    """
    src = ROOT / "data" / "video_landmarks"
    if not src.is_dir():
        return set()
    return {p.name for p in src.iterdir() if p.is_dir()}


def parts_extracted() -> int:
    return len(list((ROOT / "data" / "video").glob("*.extracted")))


def run_arm(Xtr, ytr, Xte, yte, n_classes, rng, arch="cnn"):
    """One train+eval. Identical to train.run, minus the checkpoint bookkeeping.

    `arch` selects the architecture. Everything else, augmentation draws, seed,
    epochs, callbacks, held-out protocol, is held identical across arms, so the
    only thing that varies is the model. The two paths differ solely in whether
    the normalised clip keeps its (T,V,C) graph shape or is flattened to (T,195);
    the normalisation statistics are bit-identical either way.
    """
    from tensorflow import keras

    Xte_raw = Xte
    Xa, ya = aug.augment_batch(Xtr, ytr, rng, factor=4)
    if arch == "slgcn":
        Xa = standardise_graph(Xa)
        model = build_slgcn(Xa.shape[1], Xa.shape[2], Xa.shape[3], n_classes)
    else:
        Xa = standardise(Xa)
        model = build_model(Xa.shape[1], Xa.shape[2], n_classes)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                          patience=5, min_lr=1e-5),
    ]
    norm = standardise_graph if arch == "slgcn" else standardise
    model.fit(Xa, ya, validation_data=(norm(Xte), yte),
              epochs=ARCH_EPOCHS.get(arch, EPOCHS), batch_size=BATCH,
              callbacks=cbs, verbose=0)

    def _score(X4):
        p = model.predict(norm(X4), verbose=0)
        t1 = float((p.argmax(1) == yte).mean())
        t5 = float(np.mean([yte[i] in p[i].argsort()[-5:] for i in range(len(yte))]))
        return t1, t5

    top1, top5 = _score(Xte_raw)
    c1, c5 = _score(close_range(Xte_raw))
    return top1, top5, c1, c5


def evaluate(X, y, signer, n_classes, arm_name, arch="cnn"):
    """Held-out-group protocol, mean over groups, same as train.py."""
    rows = []
    for g in sorted(set(signer.tolist())):
        # Reseed per arm+group so both arms see identical augmentation draws.
        rng = np.random.default_rng(SEED + g)
        tf.random.set_seed(SEED + g)
        te_m = signer == g
        top1, top5, c1, c5 = run_arm(X[~te_m], y[~te_m], X[te_m], y[te_m],
                                     n_classes, rng, arch=arch)
        print(f"  {arm_name} group {g}: far {top1*100:5.1f}%  close {c1*100:5.1f}%")
        rows.append({"group": int(g), "top1": top1, "top5": top5,
                     "close_top1": c1, "close_top5": c5})
    mean_far = float(np.mean([r["top1"] for r in rows]))
    mean_close = float(np.mean([r["close_top1"] for r in rows]))
    return rows, mean_far, mean_close


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA.relative_to(ROOT)}, run train/preprocess_face.py first")
        return 1

    d = np.load(DATA, allow_pickle=True)
    X_head, X_full = d["X_head"], d["X_full"]
    y, signer, labels = d["y"], d["signer"], list(d["labels"])
    n_classes = len(labels)

    print(f"{len(y)} clips | {n_classes} classes | groups {np.bincount(signer).tolist()}")
    print(f"HEAD_ONLY input {X_head.shape[1:]}   FULL_FACE input {X_full.shape[1:]}")
    print(f"chance {100/n_classes:.1f}%\n")

    print("=" * 62)
    print("HEAD_ONLY: 65 points, no face mesh (baseline)")
    print("=" * 62)
    head_rows, head_far, head_close = evaluate(X_head, y, signer, n_classes, "head")

    print()
    print("=" * 62)
    print("FULL_FACE: 113 points, + eyebrows / eyes / lips")
    print("=" * 62)
    full_rows, full_far, full_close = evaluate(X_full, y, signer, n_classes, "full")

    # Third arm: same 65 points as HEAD_ONLY, same everything, only the
    # architecture differs. The flat model discards the fact that the landmarks
    # form a skeleton; SL-GCN convolves along it. Held at 0.90x the baseline's
    # parameters so a win cannot be explained by capacity.
    print()
    print("=" * 62)
    print("SL-GCN: 65 points as a GRAPH, not a flat vector")
    print("=" * 62)
    gcn_rows, gcn_far, gcn_close = evaluate(X_head, y, signer, n_classes, "gcn",
                                            arch="slgcn")

    d_far = (full_far - head_far) * 100
    d_close = (full_close - head_close) * 100

    # Per-group deltas. The mean alone hides the spread, and with only three
    # groups the spread is the thing that decides whether the mean means anything.
    per_group_close = [
        (f["group"], (f["close_top1"] - h["close_top1"]) * 100)
        for h, f in zip(head_rows, full_rows)
    ]
    spread = float(np.std([d for _, d in per_group_close]))

    print()
    print("=" * 62)
    print("RESULT: held-out group mean")
    print("=" * 62)
    print(f"  HEAD_ONLY   far {head_far*100:5.1f}%   close {head_close*100:5.1f}%   Conv1D, 65pt")
    print(f"  FULL_FACE   far {full_far*100:5.1f}%   close {full_close*100:5.1f}%   Conv1D, 113pt")
    print(f"  SL-GCN      far {gcn_far*100:5.1f}%   close {gcn_close*100:5.1f}%   graph,  65pt")
    print()
    print(f"  face delta        far {d_far:+5.1f}pp  close {d_close:+5.1f}pp")
    print(f"  architecture delta far {(gcn_far-head_far)*100:+5.1f}pp  "
          f"close {(gcn_close-head_close)*100:+5.1f}pp   (SL-GCN vs Conv1D)")
    print()
    print("  per-group close delta: " +
          "  ".join(f"g{g} {d:+.1f}pp" for g, d in per_group_close))
    print(f"  spread (sd) {spread:.1f}pp over {len(per_group_close)} groups")
    print()

    # Verdict. Promotion follows train.py in caring about close range, but a mean
    # smaller than the between-group spread is not a result, with three groups
    # there is no power to call it, and saying otherwise would be exactly the
    # overclaiming this project refuses to do elsewhere.
    if abs(d_close) < spread:
        print("  -> INCONCLUSIVE. The mean difference is smaller than the spread")
        print("     between groups, so this subset cannot separate the two arms.")
        if d_close > 0 and d_far < 0:
            print("     Direction is worth noting though: close range gains while far")
            print("     loses: the same shape as the perspective-augmentation result")
            print("     in HANDOFF.md §6. Judging FULL_FACE on far-camera score alone")
            print("     would reject it for the wrong reason.")
        # Whether the decisive categories are present changes what "inconclusive"
        # means, so derive it rather than asserting it. Stated wrongly this is
        # worse than silence: it sends you to fetch data you already have.
        have_nm = sorted(categories_present() & NON_MANUAL_CATEGORIES)
        if not have_nm:
            print("     Re-run when the question-heavy categories "
                  f"({', '.join(sorted(NON_MANUAL_CATEGORIES))}) are extracted;")
            print("     the categories present here carry little non-manual grammar.")
        else:
            print(f"     Note: {', '.join(have_nm)} IS included here, the categories")
            print("     where ISL marks questions and negation. So this is no longer")
            print("     a vocabulary gap. What remains is statistical power: three")
            print("     signer groups cannot resolve a difference this small. More")
            print("     GROUPS, not more categories, is what would settle it.")
    elif d_close > 0:
        print("  -> FULL_FACE helps, beyond the between-group spread.")
        print("     Set FACE_MODE='FULL_FACE' and mirror it in app/src/lib/features.ts.")
    else:
        print("  -> FULL_FACE hurts, beyond the between-group spread.")
        print("     Do NOT switch on this evidence.")

    gcn_delta = (gcn_close - head_close) * 100
    gcn_per_group = [(g["group"], (g["close_top1"] - h["close_top1"]) * 100)
                     for h, g in zip(head_rows, gcn_rows)]
    gcn_spread = float(np.std([d for _, d in gcn_per_group]))
    print("  SL-GCN per-group close delta: " +
          "  ".join(f"g{g} {d:+.1f}pp" for g, d in gcn_per_group))
    print(f"  spread (sd) {gcn_spread:.1f}pp")
    if abs(gcn_delta) < gcn_spread:
        print("  -> SL-GCN INCONCLUSIVE on this subset (mean < between-group spread).")
    elif gcn_delta > 0:
        print("  -> SL-GCN WINS beyond the spread, at 0.90x the parameters.")
        print("     Structure, not capacity. Worth porting: the ops are einsum/")
        print("     matmul/conv2d, all supported by TF.js.")
    else:
        print("  -> SL-GCN loses beyond the spread. Keep Conv1D.")
    print()

    payload = {
        "scope": {
            "clips": int(len(y)), "classes": n_classes,
            "groups": np.bincount(signer).tolist(),
            "categories": sorted(categories_present()),
            "parts_extracted": parts_extracted(),
            "note": (
                f"{parts_extracted()} of 46 Zenodo parts, "
                f"{', '.join(sorted(categories_present()))}. "
                "NOT the 264-class result."
            ),
        },
        "head_only": {"per_group": head_rows, "mean_far": head_far, "mean_close": head_close},
        "full_face": {"per_group": full_rows, "mean_far": full_far, "mean_close": full_close},
        "sl_gcn": {"per_group": gcn_rows, "mean_far": gcn_far, "mean_close": gcn_close,
                   "delta_close_pp": gcn_delta, "spread_pp": gcn_spread,
                   "conclusive": bool(abs(gcn_delta) >= gcn_spread)},
        "delta_pp": {"far": d_far, "close": d_close},
        "per_group_close_delta_pp": {str(g): d for g, d in per_group_close},
        "between_group_spread_pp": spread,
        "conclusive": bool(abs(d_close) >= spread),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
