"""
Run candidate models against your own recorded takes.

    .venv-tf/bin/python train/eval_on_takes.py setu-recordings-*.json

The direct test. No geometry inference, no simulated camera, no assumption
about why the domain differs, just: which checkpoint reads your signing?

Every model in models/*.keras is scored, so the perspective-augmented one can
be compared against the one trained without it on the same real frames.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
import features
from train import standardise

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    takes = []
    for pattern in sys.argv[1:]:
        p = Path(pattern)
        for path in ([p] if p.exists() else sorted(Path().glob(pattern))):
            doc = json.loads(path.read_text())
            takes.extend(doc.get("takes", []))
    if not takes:
        print("no takes found")
        return 1

    labels = json.loads((MODELS / "labels.json").read_text())
    label_id = {g.casefold(): i for i, g in enumerate(labels)}

    X, y, names = [], [], []
    unknown = set()
    bad = 0
    for t in takes:
        g = str(t.get("gloss", "")).strip()
        seq = np.asarray(t.get("frames", []), dtype=np.float64)
        if not g or seq.ndim != 3 or seq.shape[1] != features.N_POINTS:
            bad += 1
            continue
        aspect = float(t.get("aspect") or 16 / 9)
        X.append(features.resample(features.anchor(features.isotropic(seq, aspect))))
        names.append(g)
        key = g.casefold()
        if key in label_id:
            y.append(label_id[key])
        else:
            y.append(-1)
            unknown.add(g)
    if not X:
        print("no valid takes found")
        return 1
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y)

    print(f"{len(X)} takes, {len(set(names))} distinct labels")
    if bad:
        print(f"skipped {bad} malformed take(s)")
    if unknown:
        print(f"not in the 264-class vocabulary (scored as top-3 only): {sorted(unknown)}")
    scored = y >= 0
    print()

    ckpts = sorted(MODELS.glob("gloss_classifier*.keras"))
    for ck in ckpts:
        model = tf.keras.models.load_model(ck)
        probs = model.predict(standardise(X), verbose=0)
        line = f"{ck.name:52s}"
        if scored.any():
            top1 = (probs[scored].argmax(1) == y[scored]).mean() * 100
            top5 = np.mean([y[i] in probs[i].argsort()[-5:]
                            for i in np.where(scored)[0]]) * 100
            line += f" top-1 {top1:5.1f}%  top-5 {top5:5.1f}%"
        else:
            line += "  (no takes match a known class)"
        print(line)

        # show what it thinks, which is useful even for unknown labels
        for i in range(min(5, len(X))):
            top = probs[i].argsort()[-3:][::-1]
            guesses = ", ".join(f"{labels[j]} {probs[i][j]:.2f}" for j in top)
            print(f"    {names[i]:18s} -> {guesses}")
        print()

    print("Chance is %.1f%%." % (100 / len(labels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
