"""
Regenerate the browser parity fixtures against the model that actually ships.

    .venv-tf/bin/python train/refresh_parity_fixtures.py

Rewrites app/public/model/_ref.json   single-input weight-transfer check
         app/public/model/_demo.json  end-to-end frames -> features -> model

Why this exists
---------------
Both fixtures were written by hand once and never regenerated. When the model
went from 264 classes to 83 they were left behind, so _ref.json still carried
264 probabilities and expected the top-1 to be "loud", a word the shipped model
has no output for. devParity.ts had been printing FAIL on every dev load since,
and a check that always fails is a check nobody reads.

The clips themselves are kept exactly as they were. Only the expectations are
recomputed, because the point of the fixture is "does the browser agree with
Python about THIS input", not "which words happen to be in the vocabulary".

Run this after any train/export_tfjs.py that changes the model.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import features  # noqa: E402

OUT = ROOT / "app" / "public" / "model"
import os
# Follow whatever export_tfjs.py ships, or the fixtures are regenerated against
# a model the browser is not running and the parity check silently compares the
# wrong things. That is how it drifted to 264 classes last time.
KERAS = ROOT / "models" / os.environ.get("SETU_VOCAB", "clinical") / "gloss_classifier.keras"


def main() -> int:
    import tensorflow as tf
    if not KERAS.exists():
        print(f"missing {KERAS}")
        return 1
    model = tf.keras.models.load_model(KERAS, compile=False)
    labels = json.loads((OUT / "labels.json").read_text())
    n = model.output_shape[-1]
    if n != len(labels):
        print(f"model has {n} outputs but labels.json has {len(labels)}")
        return 1
    print(f"model: {n} classes\n")

    # _ref.json: one already-extracted feature block
    ref_p = OUT / "_ref.json"
    ref = json.loads(ref_p.read_text())
    X = np.array(ref["input"], dtype=np.float32)[None, ...]
    p = model.predict(X, verbose=0)[0]
    was = ref.get("top1")
    ref["probs"] = [float(v) for v in p]
    ref["top1"] = labels[int(p.argmax())]
    ref_p.write_text(json.dumps(ref))
    print(f"_ref.json   {len(ref['probs'])} probs   top1 '{was}' -> "
          f"'{ref['top1']}' ({p.max():.4f})")

    # _demo.json: raw unit frames, run through the shared feature contract
    demo_p = OUT / "_demo.json"
    clips = json.loads(demo_p.read_text())
    feats = []
    for c in clips:
        seq = np.array(c["frames"], dtype=np.float64)
        aspect = float(c.get("aspect", 16 / 9))
        feats.append(features.extract(seq, aspect))
    F = np.stack(feats).reshape(len(feats), features.SEQ_LEN,
                                features.N_POINTS * features.N_DIMS)
    P = model.predict(F, verbose=0)
    changed = 0
    for c, row in zip(clips, P):
        new = labels[int(row.argmax())]
        if c.get("pred") != new:
            changed += 1
        c["pred"] = new
        c["conf"] = float(row.max())
    demo_p.write_text(json.dumps(clips))
    right = sum(1 for c in clips if str(c["true"]).lower() == str(c["pred"]).lower())
    print(f"_demo.json  {len(clips)} clips, {changed} expectations updated")
    print(f"            python gets {right}/{len(clips)} of them right")
    print("\nreload the dev server: devParity should now print PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
