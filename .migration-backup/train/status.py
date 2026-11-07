"""Read-only pipeline inventory; works without TensorFlow or numpy installed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def status():
    public = ROOT / 'app/public/model'
    labels = json.loads((public / 'labels.json').read_text())
    library = json.loads((public / '_signs.json').read_text())
    sign_index = ROOT / 'app/public/signs/index.json'
    index = json.loads(sign_index.read_text()) if sign_index.exists() else None
    spec = json.loads((public / 'model.json').read_text())
    extracted = {}
    include = ROOT / 'data/Pose_Signs'
    extracted['include'] = sum(1 for _ in include.rglob('*.pkl')) if include.exists() else 0
    for name in ('ncert', 'islgov', 'shiksha', 'cislr'):
        folder = ROOT / f'data/{name}_landmarks'
        extracted[name] = sum(1 for _ in folder.rglob('*.npz')) if folder.exists() else 0
    return {
        'runtime': 'Browser TensorFlow.js; no inference API server is required',
        'recognition_labels': len(labels), 'playback_signs': len(library),
        'playback_shards': len(index.get('shards', [])) if index else 0,
        'labels_without_playback': [g for g in labels if g not in library],
        'missing_weight_shards': [p for group in spec['weightsManifest'] for p in group['paths'] if not (public / p).is_file()],
        'training_inputs': {str(p): (ROOT / p).exists() for p in [
            'data/Pose_Signs', 'data/dataset.npz', 'data/own.npz', 'data/signer_index.json',
            'models/gloss_classifier.keras', 'app/public/vision/holistic_landmarker.task']},
        'extracted_sources': extracted,
        'next_download_command': 'python train/pipeline.py sync --priority clinical --batch-size 100',
    }
if __name__ == '__main__':
    print(json.dumps(status(), indent=2))
