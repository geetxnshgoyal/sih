"""
Convert the trained Keras model to TF.js for in-browser inference.

    .venv-tf/bin/python train/export_tfjs.py

Writes app/public/model/  (model.json + weight shards) and labels.json.

Why TF.js and not ONNX: the app is client-side, and tfjs converts straight
from Keras with no intermediate format. Going PyTorch -> ONNX -> onnxruntime-web
would add a conversion step between us and a working demo.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--clinical", action="store_true",
                 help="export models/clinical/ to app/public/model/clinical/")
_args, _ = _ap.parse_known_args()

# Two models ship. The 264-class one is the general vocabulary; the 38-class
# clinical one trades words a clinic never needs for accuracy on the ones it
# does, and is measurably better where it counts: 73.9% vs 64.8% top-1 and
# 96.3% vs 86.6% top-5 on a held-out signer.
_SRC = ROOT / ("models/clinical" if _args.clinical else "models")
MODEL = _SRC / "gloss_classifier.keras"
LABELS = _SRC / "labels.json"
METRICS = _SRC / "metrics.json"
OUT = ROOT / "app" / "public" / "model" / ("clinical" if _args.clinical else "")


def patch_keras3_topology(model_json: Path) -> None:
    """Make a Keras 3 topology loadable by TF.js layers models.

    Keras 3 writes InputLayer as {"batch_shape": [...]}, but tfjs still expects
    {"batch_input_shape": [...]} and throws "An InputLayer should be passed
    either a batchInputShape or an inputShape". Same graph, different key.
    """
    spec = json.loads(model_json.read_text())
    layers = spec["modelTopology"]["model_config"]["config"]["layers"]
    patched = 0
    for layer in layers:
        cfg = layer.get("config", {})
        if layer.get("class_name") == "InputLayer" and "batch_shape" in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
            patched += 1
        # Keras 3 adds keys tfjs does not know; harmless but noisy
        for dead in ("optional", "sparse", "ragged"):
            cfg.pop(dead, None)
    model_json.write_text(json.dumps(spec))
    print(f"patched {patched} InputLayer(s) for TF.js compatibility")


def main() -> int:
    if not MODEL.exists():
        print(f"missing {MODEL}: run train/train.py first")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)

    # tensorflowjs_converter ships as a console script in the same venv
    converter = Path(sys.executable).parent / "tensorflowjs_converter"
    if not converter.exists():
        print("tensorflowjs_converter not found. Install with:")
        print(f"  {sys.executable} -m pip install tensorflowjs")
        return 1

    import tensorflow as tf  # noqa: PLC0415
    from tensorflow import keras  # noqa: PLC0415

    # Export as a TF SavedModel and convert to a tfjs GRAPH model, not a layers
    # model. Keras 3 serialises Functional graphs in a shape tfjs's layers
    # loader rejects (batch_shape vs batch_input_shape, object-form
    # inbound_nodes, sequential/ weight prefixes). A frozen graph carries no
    # Keras config at all, so none of that can go wrong. Inference is identical;
    # the app loads it with tf.loadGraphModel instead of tf.loadLayersModel.
    model = keras.models.load_model(MODEL)
    sm = ROOT / "models" / "saved_model"
    if sm.exists():
        shutil.rmtree(sm)
    model.export(sm)
    print(f"exported SavedModel to {sm.name}")

    cmd = [
        str(converter),
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        str(sm),
        str(OUT),
    ]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        return proc.returncode

    shutil.copy(LABELS, OUT / "labels.json")
    if METRICS.exists():
        shutil.copy(METRICS, OUT / "metrics.json")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({total/1e6:.2f} MB total)")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size/1e3:8.1f} KB")

    labels = json.loads(LABELS.read_text())
    print(f"\n{len(labels)} classes. Browser loads it from /model/model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
