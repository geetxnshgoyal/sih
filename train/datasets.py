"""Validated datasets with global label remapping and separate signer IDs."""
import hashlib
from pathlib import Path
import numpy as np
import features


def load_datasets(paths):
    arrays, labels_all = [], set()
    signature = hashlib.sha256()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Missing training data: {path}. Run preprocess.py or ingest_recordings.py first.")
        signature.update(hashlib.sha256(path.read_bytes()).digest())
        with np.load(path, allow_pickle=False) as data:
            X, y, signer, labels = data['X'], data['y'], data['signer'], data['labels'].tolist()
        if not labels or len(set(labels)) != len(labels) or any(not isinstance(g, str) or not g.strip() for g in labels):
            raise ValueError(f"{path}: labels must be distinct nonempty strings")
        if X.ndim != 4 or X.shape[1:] != (features.SEQ_LEN, features.N_POINTS, 3) or not len(X) or not np.isfinite(X).all():
            raise ValueError(f"{path}: invalid feature tensor {X.shape}")
        if y.shape != (len(X),) or signer.shape != y.shape or y.dtype.kind not in 'iu' or signer.dtype.kind not in 'iu' or y.min() < 0 or y.max() >= len(labels):
            raise ValueError(f"{path}: invalid label/signer indices")
        arrays.append((X, y, signer, labels)); labels_all.update(labels)
    labels = sorted(labels_all); ids = {g: i for i, g in enumerate(labels)}
    xs, ys, groups, offset = [], [], [], 0
    for X, y, signer, source_labels in arrays:
        xs.append(X); ys.append(np.array([ids[source_labels[i]] for i in y], dtype=np.int32))
        source_groups = {g: offset + i for i, g in enumerate(np.unique(signer))}
        groups.append(np.array([source_groups[g] for g in signer], dtype=np.int32)); offset += len(source_groups)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups), labels, signature.hexdigest()


def stratified_split(y, rng, fraction=0.2):
    train, test = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < 3:
            raise ValueError(f"Class index {label} has only {len(indices)} examples; need at least 3 for separate splits")
        count = max(1, min(len(indices) - 2, round(len(indices) * fraction)))
        test.extend(indices[:count]); train.extend(indices[count:])
    return rng.permutation(train), rng.permutation(test)
