"""
Export the shipping models to TFLite, for a native mobile app.

    .venv-tf/bin/python train/export_tflite.py

Reads  models/clinical/, models/universal/
Writes models/tflite/<name>_fp32.tflite and <name>_int8.tflite

Why this exists
---------------
The browser build is not the whole answer. Most people in India reach a service
through a phone app, not a laptop browser, and TF.js in a mobile browser is
slower than TFLite running natively on the same hardware.

The good news is that the model is not the obstacle. The same Keras weights
convert straight to TFLite, and int8 weight quantisation takes it to about half
a megabyte:

    clinical    fp32 1942 KB    int8 500 KB
    universal   fp32 1987 KB    int8 512 KB

What a native app would still have to rebuild is the FEATURE TRANSFORM, which is
the parity-critical part: MediaPipe Tasks has Android and iOS bindings that give
the same landmarks, but features.py/features.ts would need a third
implementation, and test_parity.py would need to assert against it too. That is
the real cost of going native, not the model.

Verification here is deliberately narrow: the quantised model is checked to
agree with Keras on the argmax for a random input. That catches a broken
conversion. It does NOT establish that int8 preserves accuracy on real data,
which needs a run over the held-out set before anything ships.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "tflite"
MODELS = ("clinical", "universal")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ok = True
    for name in MODELS:
        src = ROOT / "models" / name / "gloss_classifier.keras"
        if not src.exists():
            print(f"  ! missing {src.relative_to(ROOT)}")
            continue
        m = tf.keras.models.load_model(src)

        fp32 = tf.lite.TFLiteConverter.from_keras_model(m).convert()
        (OUT / f"{name}_fp32.tflite").write_bytes(fp32)

        conv = tf.lite.TFLiteConverter.from_keras_model(m)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        int8 = conv.convert()
        (OUT / f"{name}_int8.tflite").write_bytes(int8)

        # Does the quantised model still pick the same class? Catches a broken
        # conversion, not a subtle accuracy loss -- see the module docstring.
        agree = 0
        trials = 24
        for _ in range(trials):
            x = rng.standard_normal((1, 32, 195)).astype(np.float32)
            ref = m.predict(x, verbose=0)
            it = tf.lite.Interpreter(model_content=int8)
            it.allocate_tensors()
            i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
            it.set_tensor(i0["index"], x)
            it.invoke()
            agree += int(ref.argmax() == it.get_tensor(o0["index"]).argmax())
        if agree < trials:
            ok = False
        print(f"  {name:10s} fp32 {len(fp32)/1024:6.0f} KB   int8 {len(int8)/1024:6.0f} KB"
              f"   argmax agrees {agree}/{trials}")

    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if not ok:
        print("  ! quantisation changed a prediction; check before shipping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
