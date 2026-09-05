"""
Build the 27-joint, 2-channel dataset that OpenHands' pretrained ST-GCN expects.

    .venv/bin/python train/preprocess_stgcn27.py

Reads  data/Pose_Signs/**/*.pkl, data/cislr_landmarks/**/*.npz, data/signer_index.json
Writes data/dataset_stgcn27.npz    X (N, 2, 120, 27), y, signer, corpus, labels

Why a second, parallel representation
-------------------------------------
AI4Bharat release a self-supervised ST-GCN (raw_dpc) pretrained across six sign
languages. It is the closest thing to a sign-language foundation model that is
actually downloadable, and testing it is the honest way to answer whether a big
pretrained model beats our 64.8%.

It cannot read our tensors. It wants a different everything:

    ours                          raw_dpc
    65 joints                     27 joints
    3 channels (x, y, z)          2 channels (x, y) -- no depth
    32 frames                     120 frames
    per-frame shoulder anchor     clip-level shoulder anchor

So this rebuilds from the raw sources rather than converting dataset_merged.npz,
which has already collapsed to 32 frames and cannot be un-collapsed.

Transforms mirror openhands/datasets/pose_transforms.py exactly -- PoseSelect,
CenterAndScaleNormalize (clip level, shoulders) and PoseUniformSubsampling --
because a pretrained encoder is only worth anything if fed what it was trained
on.

One deliberate departure: features.isotropic is applied first. MediaPipe's
coordinates carry the source video's aspect ratio, and INCLUDE is the one corpus
that is 16:9 rather than square (ARCHITECTURE.md 5.1). raw_dpc was pretrained
across six corpora of which five are already isotropic, so corrected geometry is
closer to what it saw, not further from it.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features
from preprocess import SafeUnpickler, clean_label

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "signer_index.json"
CISLR = ROOT / "data" / "cislr_landmarks"
MERGED = ROOT / "data" / "dataset_merged.npz"
OUT = ROOT / "data" / "dataset_stgcn27.npz"

# openhands PoseSelect preset "mediapipe_holistic_minimal_27", indexing the
# 75-point layout (33 pose + 21 left hand + 21 right hand).
KEEP_27 = [0, 2, 5, 11, 12, 13, 14,
           33, 37, 38, 41, 42, 45, 46, 49, 50, 53,
           54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
SHOULDERS = (3, 4)          # positions of pose 11/12 within KEEP_27
NUM_FRAMES = 120
INCLUDE_ASPECT = 1920 / 1080
PAD, MIN_ACTIVE = 3, 8      # CISLR trim, as in preprocess_cislr.py


def center_scale(x: np.ndarray) -> np.ndarray:
    """(T, 27, 2) -> centred on the shoulder midpoint, scaled by shoulder width.

    Clip level, not frame level: one centre and one scale for the whole clip,
    mirroring CenterAndScaleNormalize(frame_level=False). Note this differs from
    features.anchor, which is per frame -- matching the encoder matters more here
    than matching ourselves.
    """
    a, b = x[:, SHOULDERS[0], :], x[:, SHOULDERS[1], :]
    centre = ((a + b) / 2.0).mean(axis=0)
    dist = np.sqrt(((a - b) ** 2).sum(-1)).mean()
    if not np.isfinite(dist) or dist < 1e-8:
        return x - centre
    return (x - centre) / dist


def subsample(x: np.ndarray) -> np.ndarray:
    """(T, 27, 2) -> (120, 27, 2). torch.linspace(...).long() truncates."""
    t = x.shape[0]
    if t == 0:
        return np.zeros((NUM_FRAMES, len(KEEP_27), 2), dtype=np.float32)
    idx = np.linspace(0, t - 1, NUM_FRAMES).astype(np.int64)
    return x[np.clip(idx, 0, t - 1)]


def finish(kp75: np.ndarray) -> np.ndarray:
    """(T, 75, >=2) unit coords -> (2, 120, 27) ready for the encoder."""
    x = kp75[:, KEEP_27, :2].astype(np.float64)
    x = subsample(center_scale(x))
    return np.transpose(x, (2, 0, 1)).astype(np.float32)   # T,V,C -> C,T,V


def load_include(rel: str) -> np.ndarray | None:
    with open(ROOT / rel, "rb") as fh:
        obj = SafeUnpickler(fh).load()
    kp = np.asarray(obj["keypoints"], dtype=np.float64)
    kp = kp[0] if kp.ndim == 4 else kp
    w, h = float(obj["vid_shape"][0]), float(obj["vid_shape"][1])
    kp[..., 0] /= w
    kp[..., 1] /= h
    return features.isotropic(kp, features.aspect_of(obj["vid_shape"]))


def load_cislr(path: Path) -> np.ndarray | None:
    with np.load(path) as d:
        present = (np.abs(d["lh"]).sum(axis=(1, 2)) > 0) | \
                  (np.abs(d["rh"]).sum(axis=(1, 2)) > 0)
        idx = np.flatnonzero(present)
        if idx.size < MIN_ACTIVE:
            return None
        lo = max(int(idx[0]) - PAD, 0)
        hi = min(int(idx[-1]) + PAD + 1, len(present))
        kp = np.concatenate([d["pose"][lo:hi, :, :3], d["lh"][lo:hi], d["rh"][lo:hi]],
                            axis=1).astype(np.float64)
    return features.isotropic(kp, 1.0)      # CISLR is 300x300, verified


def main() -> int:
    labels = [str(s) for s in np.load(MERGED, allow_pickle=True)["labels"]]
    label_id = {n: i for i, n in enumerate(labels)}
    index = json.loads(INDEX.read_text())

    rows, ys, sg, cp = [], [], [], []
    bad = 0
    print(f"INCLUDE: {len(index)} clips")
    for i, (rel, meta) in enumerate(sorted(index.items())):
        if i % 1000 == 0 and i:
            print(f"  {i}/{len(index)}", flush=True)
        name = clean_label(meta["class"])
        if name not in label_id:
            continue
        try:
            kp = load_include(rel)
            arr = finish(kp)
            if not np.all(np.isfinite(arr)):
                bad += 1
                continue
            rows.append(arr); ys.append(label_id[name])
            sg.append(int(meta["signer"])); cp.append(0)
        except Exception:  # noqa: BLE001
            bad += 1

    # CISLR signer groups were recovered in preprocess_cislr.py; reuse them by
    # matching on clip id so the two datasets share one group numbering.
    m = np.load(MERGED, allow_pickle=True)
    cis_groups = m["signer"][m["corpus"] == 1]
    cis_files = sorted(CISLR.rglob("*.npz"))
    print(f"CISLR:   {len(cis_files)} clips")
    kept_c = 0
    for path in cis_files:
        gloss = path.parent.name
        if gloss not in label_id:
            continue
        try:
            kp = load_cislr(path)
            if kp is None:
                continue
            arr = finish(kp)
            if not np.all(np.isfinite(arr)):
                bad += 1
                continue
            rows.append(arr); ys.append(label_id[gloss])
            sg.append(int(cis_groups[kept_c]) if kept_c < len(cis_groups) else 3)
            cp.append(1)
            kept_c += 1
        except Exception:  # noqa: BLE001
            bad += 1

    X = np.stack(rows)
    y = np.asarray(ys, np.int32)
    signer = np.asarray(sg, np.int32)
    corpus = np.asarray(cp, np.int32)
    print(f"\nkept {len(X)}  (dropped {bad})")
    print(f"  INCLUDE {int((corpus == 0).sum())}  CISLR {int((corpus == 1).sum())}")
    print(f"  groups {np.bincount(signer).tolist()}")
    print(f"X {X.shape}  (C, T, V) = (2, {NUM_FRAMES}, {len(KEEP_27)})")

    np.savez_compressed(OUT, X=X, y=y, signer=signer, corpus=corpus,
                        labels=np.array(labels))
    print(f"\nwrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
