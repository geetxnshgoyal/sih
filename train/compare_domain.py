"""
Compare your camera's landmark geometry against INCLUDE's.

    .venv/bin/python train/compare_domain.py setu-recordings-*.json

Why this exists
---------------
The model collapses at laptop distance: 67.3% on held-out INCLUDE clips as
recorded, 2.1% after re-projecting them to close range. That re-projection is
a transform I wrote, not a measured camera model — MediaPipe's z is a relative
pseudo-depth in an unspecified scale, so `1 + z/d` approximates a pinhole
camera rather than reproducing one.

So the chain "your webcam fails BECAUSE of perspective distortion" has an
unverified link. This script closes it. Given real takes from your camera it
reports where your geometry actually differs from the training distribution,
and whether a perspective term explains the difference or something else does.

Twenty takes of any sign is enough.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features

ROOT = Path(__file__).resolve().parent.parent
POSE_DIR = ROOT / "data" / "Pose_Signs"

L_SH, R_SH = 11, 12
LEFT_HAND = slice(23, 44)
RIGHT_HAND = slice(44, 65)


class SafeUnpickler(pickle.Unpickler):
    ALLOW = {("numpy.core.multiarray", "_reconstruct"),
             ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, mod, name):
        if (mod, name) not in self.ALLOW:
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        return super().find_class(mod, name)


def stats(seqs: list[np.ndarray]) -> dict[str, tuple[float, float]]:
    """Scale-free descriptors of a set of unit-coordinate clips."""
    shoulder, zspread, hand_z, hand_span, torso = [], [], [], [], []
    for s in seqs:
        ls, rs = s[:, L_SH, :2], s[:, R_SH, :2]
        sw = np.linalg.norm(ls - rs, axis=1)
        sw = np.maximum(sw, 1e-6)
        shoulder.append(sw.mean())
        # everything below is normalised by shoulder width, so it is
        # comparable across cameras and body sizes
        zspread.append((s[..., 2].std() / sw.mean()))
        hands = np.concatenate([s[:, LEFT_HAND], s[:, RIGHT_HAND]], axis=1)
        live = np.abs(hands).sum(axis=2) > 0
        if live.any():
            hand_z.append(np.abs(hands[..., 2][live]).mean() / sw.mean())
            hand_span.append(
                (hands[..., :2][live].max() - hands[..., :2][live].min()) / sw.mean()
            )
        mid = (s[:, L_SH, :2] + s[:, R_SH, :2]) / 2
        torso.append(np.linalg.norm(s[:, 0, :2] - mid, axis=1).mean() / sw.mean())

    def ms(v):
        a = np.asarray(v, dtype=float)
        return (float(a.mean()), float(a.std())) if len(a) else (float("nan"),) * 2

    return {
        "shoulder span (of frame)": ms(shoulder),
        "z spread / shoulder": ms(zspread),
        "hand |z| / shoulder": ms(hand_z),
        "hand extent / shoulder": ms(hand_span),
        "nose-to-shoulder / shoulder": ms(torso),
    }


def load_include(n: int, rng) -> list[np.ndarray]:
    paths = sorted(POSE_DIR.rglob("*.pkl"))
    pick = [paths[i] for i in rng.choice(len(paths), min(n, len(paths)), replace=False)]
    out = []
    for p in pick:
        with open(p, "rb") as fh:
            o = SafeUnpickler(fh).load()
        out.append(features.to_unit(o["keypoints"], o["vid_shape"]))
    return out


def project(seq: np.ndarray, d_cam: float) -> np.ndarray:
    a = seq.copy()
    den = np.maximum(1.0 + a[..., 2] / d_cam, 0.25)
    a[..., 0] /= den
    a[..., 1] /= den
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--include-sample", type=int, default=300)
    args = ap.parse_args()

    yours = []
    for pattern in args.files:
        p = Path(pattern)
        paths = [p] if p.exists() else sorted(Path().glob(pattern))
        for path in paths:
            doc = json.loads(path.read_text())
            for t in doc.get("takes", []):
                yours.append(np.asarray(t["frames"], dtype=np.float64))
    if not yours:
        print("no takes found")
        return 1

    rng = np.random.default_rng(0)
    include = load_include(args.include_sample, rng)
    print(f"yours: {len(yours)} takes | INCLUDE: {len(include)} clips\n")

    a, b = stats(yours), stats(include)
    print(f"{'measure':30s} {'yours':>16s} {'INCLUDE':>16s} {'ratio':>8s}")
    for k in a:
        (am, asd), (bm, bsd) = a[k], b[k]
        ratio = am / bm if bm else float("nan")
        print(f"{k:30s} {am:8.3f}±{asd:5.3f} {bm:8.3f}±{bsd:5.3f} {ratio:7.2f}x")

    # An earlier version of this tried to infer a virtual camera distance by
    # matching these statistics. It does not work: every measure here is
    # normalised by shoulder width, and the perspective transform changes
    # shoulder width too, so the normalisation cancels most of the effect.
    # Fabricated takes at a known d=1.8 were fitted as d=3.0, and the statistic
    # moved only 11% across the whole plausible range.
    #
    # The direct test needs no geometry inference at all: run each candidate
    # model on your takes and compare. That is what --eval does.
    print("\nGeometry summary only. For the question that matters — does the")
    print("perspective-augmented model actually read YOUR signing — run:")
    print("  .venv-tf/bin/python train/eval_on_takes.py <recordings.json>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
