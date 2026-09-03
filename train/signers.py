"""
Recover signer identity from body proportions.

Why this exists
---------------
INCLUDE was recorded by deaf students at one school in Chennai, but the pose
release carries no signer field, and the official train/test split (Google
Drive) now 404s. Splitting clips randomly would put the same person in both
train and test, inflating accuracy — the exact flaw that makes most published
sign-language numbers untrustworthy.

Approach
--------
A person's skeleton proportions are stable across their own clips and differ
between people. Every feature below is a RATIO, so it is invariant to how far
the signer stood from the camera and where they stood in frame.

This is a proxy, not ground truth. It is validated in profile_signers.py by
checking that recovered groups are internally tight and that each group covers
many sign classes (a real signer signed most of the vocabulary, so a cluster
that maps to one class is a class artefact, not a person).
"""
import numpy as np

# MediaPipe pose indices (within the 0..22 pose block we keep)
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH = 11, 12
L_ELB, R_ELB = 13, 14
L_WR, R_WR = 15, 16

FEATURE_NAMES = [
    "head/shoulder",
    "upperarm/shoulder",
    "forearm/shoulder",
    "forearm/upperarm",
    "neck/shoulder",
    "facedepth/head",
]


def _d(a, b) -> float:
    return float(np.linalg.norm(a - b))


def proportions(seq_unit: np.ndarray) -> np.ndarray:
    """(T, 65, 3) unit coords -> 6 scale-invariant ratios, median over frames.

    The median rather than the mean, because MediaPipe occasionally throws a
    landmark far out of place on a blurred frame and a mean would follow it.
    """
    vals = []
    for f in seq_unit:
        sh = _d(f[L_SH, :2], f[R_SH, :2])
        if sh < 1e-4:
            continue
        head = _d(f[L_EAR, :2], f[R_EAR, :2])
        if head < 1e-4:
            continue
        mid = (f[L_SH, :2] + f[R_SH, :2]) / 2.0
        upper = (_d(f[L_SH, :2], f[L_ELB, :2]) + _d(f[R_SH, :2], f[R_ELB, :2])) / 2.0
        fore = (_d(f[L_ELB, :2], f[L_WR, :2]) + _d(f[R_ELB, :2], f[R_WR, :2])) / 2.0
        vals.append([
            head / sh,
            upper / sh,
            fore / sh,
            fore / max(upper, 1e-4),
            _d(f[NOSE, :2], mid) / sh,
            _d(f[NOSE, :2], f[L_EAR, :2]) / head,
        ])
    if not vals:
        return np.zeros(len(FEATURE_NAMES))
    return np.median(np.asarray(vals), axis=0)


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100):
    """Plain k-means. No sklearn dependency for six-dimensional data."""
    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centres = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / max(d2.sum(), 1e-12)
        centres.append(X[rng.choice(len(X), p=probs)])
    C = np.array(centres)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
    return labels, C


def inertia(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> float:
    return float(sum(((X[labels == j] - C[j]) ** 2).sum() for j in range(len(C))))
