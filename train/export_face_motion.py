#!/usr/bin/env python3
"""Convert the reviewed face-motion candidate without touching deployment."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "models/face-motion-candidate"
OUT = SOURCE / "tfjs"


def main() -> int:
    model_path = SOURCE / "classifier.keras"
    metrics_path = SOURCE / "metrics.json"
    labels_path = SOURCE / "labels.json"
    if not all(path.exists() for path in (model_path, metrics_path, labels_path)):
        print("Missing candidate files. Run train/train_face_motion.py first.")
        return 1
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("deployed") is not False or not metrics.get("test_clips"):
        print("Candidate metrics are incomplete; export refused.")
        return 1
    converter = Path(sys.executable).parent / "tensorflowjs_converter"
    if not converter.exists():
        print(f"Install tensorflowjs in this environment: {sys.executable} -m pip install tensorflowjs")
        return 1
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    embedding = model.get_layer("embedding").output
    export_model = keras.Model(model.inputs, [model.output, embedding], name="face_motion")
    saved = SOURCE / "saved_model"
    if saved.exists():
        shutil.rmtree(saved)
    if OUT.exists():
        shutil.rmtree(OUT)
    export_model.export(saved)
    result = subprocess.run([
        str(converter), "--input_format=tf_saved_model", "--output_format=tfjs_graph_model",
        "--signature_name=serving_default", "--saved_model_tags=serve", str(saved), str(OUT),
    ], check=False)
    if result.returncode:
        return result.returncode
    shutil.copy(labels_path, OUT / "labels.json")
    shutil.copy(metrics_path, OUT / "metrics.json")
    spec = json.loads((OUT / "model.json").read_text())
    if len(spec.get("signature", {}).get("inputs", {})) != 2:
        print("Export did not preserve both body and face inputs.")
        return 1
    print(f"Candidate browser bundle written to {OUT.relative_to(ROOT)}.")
    print("Deployment remains unchanged; review metrics and camera evaluation before promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
