"""Export a playback clip for EVERY label in a preprocessed dataset.

python3 train/export_playback.py --data data/dataset.npz --own data/own.npz
The dataset is anchored before standardization, exactly the player's coordinates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from datasets import load_datasets

ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/dataset.npz')
    parser.add_argument('--own', type=Path, default=ROOT / 'data/own.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'app/public/model/_signs.json')
    args = parser.parse_args()
    paths = list(dict.fromkeys(p for p in (args.data, args.own) if p.is_file()))
    if not paths:
        print('No training tensors found. Import recordings or preprocess the INCLUDE corpus first.'); return 1
    X, y, _, labels, _ = load_datasets(paths)
    library = {}
    # Extend an existing library so an own-only export does not drop other signs.
    if args.output.exists():
        library = json.loads(args.output.read_text())
    for i, label in enumerate(labels):
        candidates = X[y == i]
        if not len(candidates):
            raise ValueError(f'No playback example for {label}')
        # Prefer the example closest to the class median (not arbitrary first file).
        center = np.median(candidates, axis=0)
        best = np.argmin(np.mean((candidates - center) ** 2, axis=(1, 2, 3)))
        library[label] = np.round(candidates[best].astype(float), 5).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(library, separators=(',', ':'))); temporary.replace(args.output)
    print(f'{len(library)} playback signs exported to {args.output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
