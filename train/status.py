"""Read-only pipeline inventory; works without TensorFlow or numpy installed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def status():
    public = ROOT / 'app/public/model'
    labels = json.loads((public / 'labels.json').read_text())
    library = json.loads((public / '_signs.json').read_text())
    spec = json.loads((public / 'model.json').read_text())
    return {
        'runtime': 'Browser TensorFlow.js; no inference API server is required',
        'recognition_labels': len(labels), 'playback_signs': len(library),
        'labels_without_playback': [g for g in labels if g not in library],
        'missing_weight_shards': [p for group in spec['weightsManifest'] for p in group['paths'] if not (public / p).is_file()],
        'training_inputs': {str(p): (ROOT / p).exists() for p in [
            'data/Pose_Signs', 'data/dataset.npz', 'data/own.npz', 'data/signer_index.json',
            'models/gloss_classifier.keras', 'app/public/vision/holistic_landmarker.task']},
    }
if __name__ == '__main__':
    print(json.dumps(status(), indent=2))
