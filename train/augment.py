"""
Geometric augmentation in landmark space.

The model saw 7 signers at one framing (shoulders spanning 0.12-0.15 of frame
width) in one room. A webcam user sits closer, may be mirrored, and gets z from
a slightly different depth estimate. These augmentations manufacture that
variety so the model stops depending on conditions it happened to be trained in.

Operates on the anchored (T, 65, 3) tensor, before standardisation.
"""
import numpy as np

# MediaPipe pose landmarks come in left/right pairs. Mirroring the body means
# swapping each pair, not just negating x, otherwise the left elbow ends up
# labelled as the right one and the skeleton is inconsistent.
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
              (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
POSE_KEEP = 23
LEFT_HAND = slice(23, 44)
RIGHT_HAND = slice(44, 65)


def mirror(seq: np.ndarray) -> np.ndarray:
    """Reflect the signer left-right.

    This is the augmentation that matters most for deployment. It makes the
    model handedness-agnostic, so a left-handed signer works, AND it removes
    the risk that MediaPipe's Left/Right labelling is inverted on a selfie
    camera relative to how the training data was extracted. Rather than guess
    which convention is right, train on both.
    """
    out = seq.copy()
    out[..., 0] *= -1.0
    for a, b in POSE_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    left = out[:, LEFT_HAND].copy()
    out[:, LEFT_HAND] = out[:, RIGHT_HAND]
    out[:, RIGHT_HAND] = left
    return out


def perspective(seq: np.ndarray, rng, d_lo=1.6, d_hi=9.0) -> np.ndarray:
    """Simulate camera distance by re-projecting through a pinhole model.

    This replaces an earlier scale_jitter that was a no-op: multiplying every
    coordinate by a constant is removed exactly by the unit-variance
    standardisation downstream (verified: max diff 4e-16). Distance is NOT a
    scale change, it is a projective one, points nearer the lens spread
    outward more than far ones, and that non-linearity does survive
    normalisation.

    d is the virtual camera distance in shoulder-widths. Small d is a laptop
    webcam at arm's length; large d approaches the orthographic view the
    INCLUDE signers were filmed at.
    """
    d = rng.uniform(d_lo, d_hi)
    out = seq.copy()
    denom = np.maximum(1.0 + out[..., 2] / d, 0.25)
    out[..., 0] /= denom
    out[..., 1] /= denom
    return out


def rotate(seq: np.ndarray, rng, max_deg=12.0) -> np.ndarray:
    """Small in-plane rotation: camera tilt, or a signer leaning."""
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(th), np.sin(th)
    out = seq.copy()
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def z_noise(seq: np.ndarray, rng, scale=0.35) -> np.ndarray:
    """Perturb depth.

    z is the least trustworthy channel across MediaPipe versions and camera
    setups, yet it carries about a third of the input signal (pose z std 1.32,
    hand z std 0.75). Adding noise stops the model leaning on depth detail that
    will not reproduce on someone else's webcam.
    """
    out = seq.copy()
    out[..., 2] += rng.normal(0, scale, out[..., 2].shape)
    return out


def time_mask(seq: np.ndarray, rng, max_width=5) -> np.ndarray:
    """Blank a few consecutive frames, occlusion, or a dropped detection."""
    out = seq.copy()
    w = int(rng.integers(2, max_width + 1))
    s = int(rng.integers(0, max(len(out) - w, 1)))
    out[s:s + w] = 0.0
    return out


# Perspective is back ON, and the earlier decision to disable it was wrong.
#
# It was judged against the far-camera held-out set, where it cost accuracy
# (51.6% -> 45.0%) and was reverted as a regression. But that set contains no
# close-range footage, so it could not see what perspective augmentation buys.
# Re-projecting the same held-out clips to laptop distance shows the real
# picture: the far-camera model scores 67.3% as recorded and 2.1% close up,
# against 0.4% chance. train.py now reports both and promotes on close.
#
# Mirror stays off: switching to HolisticLandmarker made the browser's
# handedness convention match the training extraction, so it is no longer
# guarding against anything.
def hand_dropout(seq: np.ndarray, rng, max_frac=0.35) -> np.ndarray:
    """Blank one hand for a run of frames, as MediaPipe does in the wild.

    Every other augmentation here perturbs geometry. None of them reproduce the
    failure that actually happens: the hand tracker losing a hand. Measured hand
    presence is 0.89 on INCLUDE, 0.79 on the ISL dictionary and as low as 0.44
    before trimming, and a zeroed hand is not noise, it is a specific point in
    feature space the model has never been trained to tolerate.

    A contiguous run rather than scattered frames, because that is the shape of
    a real dropout: the tracker loses the hand and takes a moment to find it.
    """
    seq = seq.copy()
    t = seq.shape[0]
    width = int(rng.integers(1, max(int(t * max_frac), 2)))
    start = int(rng.integers(0, max(t - width, 1)))
    hand = LEFT_HAND if rng.random() < 0.5 else RIGHT_HAND
    seq[start:start + width, hand, :] = 0.0
    return seq


USE_MIRROR = False
USE_PERSPECTIVE = True
USE_ROTATE = True
USE_Z_NOISE = False
USE_TIME_MASK = True
USE_HAND_DROPOUT = False   # switched on per experiment, see eval_robust.py


def augment_batch(X: np.ndarray, y: np.ndarray, rng, factor: int = 4):
    """X is (N, T, 65, 3) anchored. Returns the original plus `factor` variants."""
    Xs, ys = [X], [y]
    for i in range(factor):
        A = X.copy()
        # every second variant is mirrored, so both handedness conventions are
        # represented in roughly equal proportion
        if USE_MIRROR and i % 2 == 1:
            A = mirror(A.reshape(-1, *A.shape[2:])).reshape(A.shape) \
                if A.ndim == 4 else mirror(A)
        for j in range(len(A)):
            a = A[j]
            if USE_PERSPECTIVE:
                a = perspective(a, rng)
            if USE_ROTATE:
                a = rotate(a, rng)
            if USE_Z_NOISE:
                a = z_noise(a, rng)
            if USE_TIME_MASK and rng.random() < 0.5:
                a = time_mask(a, rng)
            if USE_HAND_DROPOUT and rng.random() < 0.5:
                a = hand_dropout(a, rng)
            A[j] = a
        Xs.append(A)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)
