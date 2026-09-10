#!/usr/bin/env python3
"""Train a non-deployed two-stream face-aware recognition candidate."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/dataset_face_motion.npz"
OUT = ROOT / "models/face-motion-candidate"
OWN = ROOT / "data/own.npz"
MIN_CLIPS = 10
MIN_GROUPS = 3


def relation_features(body: np.ndarray) -> np.ndarray:
    """Hand-to-mouth geometry and temporal motion in shoulder units."""
    mouth = (body[:, :, 9, :3] + body[:, :, 10, :3]) / 2
    joints = body[:, :, [23, 31, 35, 44, 52, 56], :3]
    relative = joints - mouth[:, :, None, :]
    velocity = np.diff(relative, axis=1, prepend=relative[:, :1])
    return np.concatenate([relative.reshape(len(body), 32, -1),
                           velocity.reshape(len(body), 32, -1)], axis=-1).astype(np.float32)


def build_model(classes: int, face_dims: int):
    import tensorflow as tf
    body = tf.keras.Input((32, 195), name="body")
    face = tf.keras.Input((32, face_dims), name="face")
    def branch(value, filters):
        value = tf.keras.layers.Conv1D(filters, 5, padding="same", activation="relu")(value)
        value = tf.keras.layers.Conv1D(filters, 3, padding="same", activation="relu")(value)
        return tf.keras.layers.GlobalAveragePooling1D()(value)
    joined = tf.keras.layers.Concatenate()([branch(body, 128), branch(face, 80)])
    embedding = tf.keras.layers.Dense(256, activation="relu", name="embedding")(joined)
    embedding = tf.keras.layers.Dropout(.35)(embedding)
    probabilities = tf.keras.layers.Dense(classes, activation="softmax", name="probabilities")(embedding)
    return tf.keras.Model([body, face], probabilities)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--personal", action="store_true",
                    help="train a one-signer candidate from data/own.npz")
    ap.add_argument("--promote", action="store_true",
                    help="refused here: evaluate and use a separate reviewed export command")
    args = ap.parse_args()
    if args.promote:
        raise SystemExit("This trainer never replaces the deployed model. Review candidate metrics first.")
    selected_data = OWN if args.personal else DATA
    if not selected_data.exists():
        raise SystemExit("Run train/ingest_recordings.py first." if args.personal
                         else "Run train/preprocess_face_motion.py first.")
    with np.load(selected_data, allow_pickle=False) as data:
        body = data["X"] if args.personal else data["X_body"]
        face, mask = data["X_face"], data["face_mask"]
        y, groups, labels = data["y"], data["signer"], data["labels"].tolist()
    if not args.personal and (ROOT / "data/dataset.npz").exists():
        with np.load(ROOT / "data/dataset.npz", allow_pickle=False) as include:
            include_body, include_y = include["X"], include["y"]
            include_groups, include_labels = include["signer"], include["labels"].tolist()
        merged_labels = sorted(set(labels) | set(include_labels), key=str.casefold)
        merged_ids = {label: index for index, label in enumerate(merged_labels)}
        y = np.concatenate([
            np.array([merged_ids[labels[index]] for index in y], dtype=np.int32),
            np.array([merged_ids[include_labels[index]] for index in include_y], dtype=np.int32),
        ])
        group_offset = (int(groups.max()) + 1) if len(groups) else 0
        groups = np.concatenate([groups, include_groups + group_offset])
        body = np.concatenate([body, include_body])
        face = np.concatenate([face, np.zeros((len(include_body), 32, 48, 3), dtype=np.float32)])
        mask = np.concatenate([mask, np.zeros((len(include_body), 32), dtype=np.float32)])
        labels = merged_labels
    required_groups = 1 if args.personal else MIN_GROUPS
    eligible = [i for i in range(len(labels))
                if (y == i).sum() >= MIN_CLIPS and len(np.unique(groups[y == i])) >= required_groups]
    if not eligible:
        raise SystemExit(f"No class has both {MIN_CLIPS}+ clips and {required_groups}+ source/signer groups.")
    remap = {old: new for new, old in enumerate(eligible)}
    keep = np.isin(y, eligible)
    body, face, mask, groups = body[keep], face[keep], mask[keep], groups[keep]
    y = np.array([remap[int(value)] for value in y[keep]], dtype=np.int32)
    labels = [labels[i] for i in eligible]
    face_input = np.concatenate([face.reshape(len(face), 32, -1),
                                 mask[..., None]], axis=-1).astype(np.float32)
    body_input = np.concatenate([body.reshape(len(body), 32, -1),
                                 relation_features(body)], axis=-1)
    # Keep the public body input at 195 by putting relational motion on the
    # face stream. This remains convertible by tensorflowjs_converter.
    face_input = np.concatenate([face_input, body_input[..., 195:]], axis=-1)
    body_input = body_input[..., :195]

    rng = np.random.default_rng(47)
    if args.personal:
        # One signer cannot support signer-disjoint evaluation. Keep a
        # stratified clip holdout and label the metric honestly as personal.
        test = np.zeros(len(y), dtype=bool)
        for label in np.unique(y):
            candidates = np.flatnonzero(y == label)
            test[rng.permutation(candidates)[:max(1, round(len(candidates) * .2))]] = True
        test_group = None
        train_pool = ~test
    else:
        test_group = int(np.bincount(groups).argmax())
        test = groups == test_group
        train_pool = ~test
    validation = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        candidates = np.flatnonzero(train_pool & (y == label))
        count = max(1, round(len(candidates) * .15))
        validation[rng.permutation(candidates)[:count]] = True
    train = train_pool & ~validation

    import tensorflow as tf
    tf.keras.utils.set_random_seed(47)
    model = build_model(len(labels), int(face_input.shape[-1]))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit({"body": body_input[train], "face": face_input[train]},
              y[train], validation_data=(
                  {"body": body_input[validation], "face": face_input[validation]},
                  y[validation]),
              epochs=args.epochs, batch_size=64, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    probabilities = model.predict({"body": body_input[test], "face": face_input[test]}, verbose=0)
    validation_probs = model.predict({"body": body_input[validation], "face": face_input[validation]}, verbose=0)
    temperatures = np.linspace(.75, 4, 66)
    def scale(probs, temperature):
        logits = np.log(np.maximum(probs, 1e-9)) / temperature
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    temperature = float(min(temperatures, key=lambda value:
        -np.log(np.maximum(scale(validation_probs, value)[np.arange(validation.sum()), y[validation]], 1e-9)).mean()))
    probabilities = scale(probabilities, temperature)
    predicted = probabilities.argmax(1)
    top5 = np.any(np.argsort(probabilities, axis=1)[:, -5:] == y[test, None], axis=1)
    hard_pairs = [("please", "pleased"), ("healthy", "how are you")]
    confusions = {}
    folded = {label.casefold(): i for i, label in enumerate(labels)}
    for first, second in hard_pairs:
        if first not in folded or second not in folded:
            confusions[f"{first}/{second}"] = {"available": False}
            continue
        a, b = folded[first], folded[second]
        relevant = np.isin(y[test], [a, b])
        confusions[f"{first}/{second}"] = {
            "available": True, "support": int(relevant.sum()),
            "cross_confusions": int(((y[test][relevant] == a) & (predicted[relevant] == b)).sum()
                                    + ((y[test][relevant] == b) & (predicted[relevant] == a)).sum()),
        }
    metrics = {
        "candidate": True, "deployed": False, "classes": len(labels),
        "labels": labels, "training_clips": int(train.sum()),
        "validation_clips": int(validation.sum()), "test_clips": int(test.sum()),
        "test_top1": float((predicted == y[test]).mean()),
        "test_top5": float(top5.mean()), "held_out_group": test_group,
        "temperature": temperature,
        "scope": "personal one-signer candidate" if args.personal else "general multi-source candidate",
        "per_class_recall": {
            label: (float((predicted[y[test] == index] == index).mean())
                    if np.any(y[test] == index) else None)
            for index, label in enumerate(labels)
        },
        "eligibility": {"minimum_clips": MIN_CLIPS, "minimum_groups": required_groups},
        "hard_negative_confusions": confusions,
        "input": {"body": [32, 195], "face": [32, int(face_input.shape[-1])]},
    }
    output = ROOT / "models/personal-candidate" if args.personal else OUT
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "classifier.keras")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "labels.json").write_text(json.dumps(labels))
    print(json.dumps(metrics, indent=2))
    print("Candidate saved. The deployed browser model was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
