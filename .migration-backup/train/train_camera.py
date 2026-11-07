"""Train a depth-independent temporal ISL candidate; never overwrite deployment.

Split raw clips before augmentation. Group 0 is an inferred body cluster, NOT
a verified signer identity. Report that limitation and per-class test support.
The browser consumes the same setu-motion-2d-v1 features and exported weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STEPS, DIMS = 32, 184
BODY = [0, 11, 12, 13, 14, 15, 16]


def motion_features(anchored):
    """Input already shoulder-anchored and isotropic; z is deliberately unused."""
    a = np.asarray(anchored, np.float64)
    shape = a.shape[:-2]
    out = [np.clip(a[..., BODY, :2], -3, 3).reshape(*shape, 14) / 3]
    presence = []
    for base in [23, 44]:
        h = a[..., base:base+21, :2]
        palm = np.linalg.norm(h[..., 9, :] - h[..., 0, :], axis=-1)
        visible = palm > 0.025
        presence.append(visible.astype(float)[..., None])
        position = np.clip(h, -3, 3) / 3
        position *= visible[..., None, None]
        local = (h - h[..., :1, :]) / np.maximum(palm, .025)[..., None, None]
        local = np.clip(local, -3, 3) / 3
        local *= visible[..., None, None]
        out += [position.reshape(*shape, 42), local.reshape(*shape, 42)]
    return np.concatenate(out + presence, axis=-1).astype(np.float32)


def make_model(classes):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input((STEPS, DIMS)),
        tf.keras.layers.Conv1D(96, 5, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(.25),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(192, activation='relu'),
        tf.keras.layers.Dropout(.4),
        tf.keras.layers.Dense(classes, activation='softmax'),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--output', type=Path, default=ROOT / 'models/camera-candidate')
    args = ap.parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(47)
    rng = np.random.default_rng(47)
    path = ROOT / 'data/dataset.npz'
    d = np.load(path, allow_pickle=False)
    labels = d['labels'].tolist()
    X, y, group = motion_features(d['X']), d['y'], d['signer']
    test = np.flatnonzero(group == 0)
    pool = np.flatnonzero(group != 0)
    tr, va = [], []
    for label in range(len(labels)):
        ids = rng.permutation(pool[y[pool] == label])
        n = max(1, round(len(ids)*.15)) if len(ids) >= 3 else 0
        va.extend(ids[:n]); tr.extend(ids[n:])
    tr, va = np.asarray(tr), np.asarray(va)
    if len(np.unique(y[tr])) != len(labels): raise ValueError('Some labels have no training data')
    print(f'{len(labels)} signs; train {len(tr)}, validation {len(va)}, test {len(test)}', flush=True)
    # Training-only augmentation: temporal resampling and small landmark noise.
    batches = [X[tr]]
    for _ in range(3):
        a = X[tr].copy()
        for i in range(len(a)):
            lo, hi = rng.integers(0, 4), rng.integers(28, 32)
            idx = np.round(np.linspace(lo, hi, STEPS)).astype(int)
            a[i] = a[i, idx]
        a[..., :-2] += rng.normal(0, .008, a[..., :-2].shape)
        batches.append(a)
    xx, yy = np.concatenate(batches), np.tile(y[tr], len(batches))
    model = make_model(len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xx, yy, batch_size=64, epochs=args.epochs,
              validation_data=(X[va], y[va]), verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=.5)])
    # Fit confidence temperature using validation ONLY, then evaluate test once.
    vp = model.predict(X[va], verbose=0)
    def scaled(p, t):
        logits = np.log(np.maximum(p, 1e-12)) / t
        ex = np.exp(logits - logits.max(1, keepdims=True))
        return ex / ex.sum(1, keepdims=True)
    grid = np.linspace(1, 4, 31)
    temperature = float(min(grid, key=lambda t: -np.log(np.maximum(scaled(vp,t)[np.arange(len(va)),y[va]],1e-12)).mean()))
    probs = scaled(model.predict(X[test], verbose=0), temperature)
    correct = probs.argmax(1) == y[test]
    top5 = np.any(np.argsort(probs, axis=1)[:, -5:] == y[test, None], axis=1)
    metrics = {'format': 'setu-motion-2d-v1', 'classes': len(labels),
               'training_clips': len(tr), 'validation_clips': len(va), 'test_clips': len(test),
               'test_top1': float(correct.mean()), 'test_top5': float(top5.mean()),
               'temperature': temperature, 'live_camera_accuracy': None,
               'split': 'Held-out inferred body cluster 0; clusters are not verified signer IDs.',
               'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'per_class': {label: {'test_support': int((y[test]==i).sum()),
                                    'accuracy': float(correct[y[test]==i].mean()) if (y[test]==i).any() else None}
                             for i,label in enumerate(labels)}}
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / 'classifier.keras')
    weights = model.get_weights()
    data = b''.join(np.asarray(w, dtype='<f4').tobytes() for w in weights)
    (args.output / 'weights.bin').write_bytes(data)
    spec = {'format': 'setu-motion-2d-v1', 'labels': labels, 'steps': STEPS, 'dims': DIMS,
            'temperature': temperature, 'weights': [list(w.shape) for w in weights],
            'weights_sha256': hashlib.sha256(data).hexdigest(), 'metrics': metrics}
    (args.output / 'model.json').write_text(json.dumps(spec, indent=2))
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k != 'per_class'}, indent=2), flush=True)
    print('Candidate saved; not promoted without camera validation.', flush=True)


if __name__ == '__main__': main()
