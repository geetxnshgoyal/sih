"""
Fold your own recorded signs into the training set.

    .venv/bin/python train/ingest_recordings.py recordings/*.json

Reads the JSON the in-app recorder downloads and writes data/own.npz in the
same shape preprocess.py produces, so train.py can consume either or both.

Why this exists: INCLUDE is 7 signers, one room, one distance. Three rounds of
augmentation aimed at closing that gap each made held-out accuracy worse
(51.6% -> 48.0% -> 45.0%). Recording in the demo environment removes the gap
rather than modelling around it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "own.npz"
FORMATS = {"setu-recordings-v1", "setu-recordings-v2", "setu-recordings-v3"}
FACE_POINTS = 48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="setu-recordings-*.json from the app")
    ap.add_argument("--min-takes", type=int, default=10,
                    help="drop signs with fewer takes than this (default 10)")
    args = ap.parse_args()

    # The shell usually expands globs; Path.glob rejects absolute patterns, so
    # expand only when the shell did not and the argument is relative.
    paths: list[Path] = []
    for pattern in args.files:
        p = Path(pattern)
        if p.exists():
            paths.append(p)
        elif not p.is_absolute():
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.extend(sorted(Path(p.anchor).glob(str(p.relative_to(p.anchor)))))

    takes = []
    for path in paths:
        if not path.exists():
            print(f"missing: {path}")
            continue
        doc = json.loads(path.read_text())
        fmt = doc.get("format")
        if fmt not in FORMATS:
            print(f"skipping {path.name}: unexpected format {doc.get('format')!r}")
            continue
        batch = doc.get("takes", [])
        takes.extend(batch)
        print(f"{path.name}: {len(batch)} takes ({fmt})")

    if not takes:
        print("nothing to ingest")
        return 1

    valid = []
    seen = set()
    for take in takes:
        try:
            raw_body = take.get('body', take.get('frames'))
            seq = np.asarray(raw_body, dtype=np.float64)
            gloss = take['gloss'].strip()
            if not gloss or seq.ndim != 3 or seq.shape[1:] != (features.N_POINTS, 3) or len(seq) < 4 or not np.isfinite(seq).all():
                raise ValueError('invalid label or landmark shape')
            hands = np.any(seq[:, 23:65, :2] != 0, axis=(1, 2))
            pose = np.linalg.norm(seq[:, 11, :2] - seq[:, 12, :2], axis=1)
            if hands.mean() < .6 or (pose > .02).mean() < .8:
                raise ValueError('a signing hand and shoulders must be visible')
            # Re-importing the same take must not inflate its training count.
            import hashlib
            key = (gloss, hashlib.sha256(seq.tobytes()).hexdigest())
            if key in seen:
                continue
            seen.add(key); valid.append({**take, 'gloss': gloss})
        except (ValueError, KeyError, TypeError) as error:
            print(f"skipping invalid take: {error}")
    takes = valid
    counts: dict[str, int] = {}
    for t in takes:
        gloss = str(t.get("gloss", "")).strip()
        if not gloss:
            continue
        counts[gloss] = counts.get(gloss, 0) + 1

    thin = {g: n for g, n in counts.items() if n < args.min_takes}
    if thin:
        print(f"\ndropping {len(thin)} sign(s) with < {args.min_takes} takes:")
        for g, n in sorted(thin.items()):
            print(f"  {g}: {n}")
    keep = sorted(g for g, n in counts.items() if n >= args.min_takes)
    if not keep:
        print("no sign has enough takes yet")
        return 1

    label_id = {g: i for i, g in enumerate(keep)}
    X = np.zeros((len(takes), features.SEQ_LEN, features.N_POINTS, features.N_DIMS),
                 dtype=np.float32)
    X_face = np.zeros((len(takes), features.SEQ_LEN, FACE_POINTS, 3), dtype=np.float32)
    face_mask = np.zeros((len(takes), features.SEQ_LEN), dtype=np.float32)
    y = np.zeros(len(takes), dtype=np.int32)

    kept = 0
    for t in takes:
        gloss = str(t.get("gloss", "")).strip()
        if gloss not in label_id:
            continue
        seq = np.asarray(t.get("body", t.get("frames", [])), dtype=np.float64)   # (T, 65, 3) unit coords
        if seq.ndim != 3 or seq.shape[1] != features.N_POINTS:
            print(f"  ! bad shape {seq.shape} for {gloss}, skipped")
            continue
        # identical path to the INCLUDE pipeline from this point on.
        # `aspect` is the recording camera's width/height, written by
        # Recorder.tsx from v2 of the format. v1 files predate the field; they
        # were captured at the 1280x720 the recorder requests, so 16/9 is what
        # they were: but a v1 file recorded on a 4:3 camera is silently wrong
        # and should be re-recorded rather than trusted.
        aspect = float(t.get("aspect") or 16 / 9)
        raw_face = t.get("face")
        face = np.zeros((len(seq), FACE_POINTS, 3), dtype=np.float64)
        available = np.zeros(len(seq), dtype=np.float32)
        if isinstance(raw_face, list) and len(raw_face) == len(seq):
            for frame_i, frame in enumerate(raw_face):
                if frame is None:
                    continue
                candidate = np.asarray(frame, dtype=np.float64)
                if candidate.shape == (FACE_POINTS, 3) and np.isfinite(candidate).all():
                    face[frame_i] = candidate
                    available[frame_i] = 1
        # Anchor body and face together so hand-to-lip geometry shares one
        # coordinate system. Missing face frames are zeroed again after the
        # transform; otherwise subtracting the shoulder midpoint would turn
        # an absent face into plausible-looking points.
        combined = np.concatenate([seq, face], axis=1)
        arr = features.resample(features.anchor(features.isotropic(combined, aspect)))
        if not np.all(np.isfinite(arr)):
            continue
        indices = np.round(np.linspace(0, len(seq) - 1, features.SEQ_LEN)).astype(int)
        sampled_mask = available[indices]
        X[kept] = arr[:, :features.N_POINTS]
        X_face[kept] = arr[:, features.N_POINTS:] * sampled_mask[:, None, None]
        face_mask[kept] = sampled_mask
        y[kept] = label_id[gloss]
        kept += 1

    X, X_face, face_mask, y = X[:kept], X_face[:kept], face_mask[:kept], y[:kept]
    # every take is the same person in the same room, so a held-out group here
    # would measure nothing; signer is a single constant.
    signer = np.zeros(kept, dtype=np.int32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, X_face=X_face, face_mask=face_mask,
                        y=y, signer=signer, labels=np.array(keep),
                        capture_format=np.array("setu-recordings-v3"))
    print(f"\n{kept} takes across {len(keep)} signs -> {OUT.relative_to(ROOT)}")
    for g in keep:
        print(f"  {g:22s} {counts[g]}")
    print("\nNote: these all come from one signer in one room. Expect high")
    print("accuracy on yourself and little generalisation to anyone else , ")
    print("which is the right trade for a demo, and the wrong one for a claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
