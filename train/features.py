"""
Feature extraction — Python side of the parity contract.

MUST stay byte-for-byte equivalent to app/src/lib/features.ts.
Any change here requires the same change there, and `make parity` must pass.

Dataset layout (OpenHands INCLUDE poses, verified):
    dict{'keypoints': (T, 75, 3) float64, 'confidences': (T,75), 'vid_shape': (W,H)}
    75 points = 33 pose + 21 left hand + 21 right hand   (MediaPipe Holistic)
    x,y are PIXEL coords -> divide by vid_shape. z is already relative.
    Pose 23-32 are legs: extrapolated outside frame, mean confidence 0.25 -> dropped.
"""
import numpy as np

from face import face_block_size

SEQ_LEN = 32
POSE_KEEP = 23          # pose landmarks 0..22 — includes the 11 head/face points
FACE_MODE = "HEAD_ONLY" # INCLUDE poses carry no face mesh; see train/face.py
N_FACE = face_block_size(FACE_MODE)
N_POINTS = POSE_KEEP + 21 + 21 + N_FACE   # 65 head-only, 113 with full face
N_DIMS = 3
FEATURE_SIZE = SEQ_LEN * N_POINTS * N_DIMS

# Head motion survives normalisation because we anchor at the SHOULDER midpoint,
# not the head — so nod/shake/tilt remain visible to the model.

# indices into the raw 75-point array
POSE = slice(0, POSE_KEEP)
LEFT = slice(33, 54)
RIGHT = slice(54, 75)

# anchor + scale reference, in the 65-point space
L_SHOULDER, R_SHOULDER = 11, 12


def select_points(kp: np.ndarray) -> np.ndarray:
    """(T,75,3) -> (T,65,3), dropping the leg landmarks."""
    return np.concatenate([kp[:, POSE], kp[:, LEFT], kp[:, RIGHT]], axis=1)


def anchor(seq: np.ndarray) -> np.ndarray:
    """Body-anchored and scale-invariant. Input already in unit coordinates."""
    seq = seq.astype(np.float64).copy()
    # anchor at the shoulder midpoint so the signer's position does not matter
    mid = (seq[:, L_SHOULDER, :] + seq[:, R_SHOULDER, :]) / 2.0
    seq -= mid[:, None, :]

    # scale by shoulder width so distance from camera does not matter
    span = np.linalg.norm(seq[:, L_SHOULDER, :2] - seq[:, R_SHOULDER, :2], axis=1)
    span = np.maximum(span, 1e-6)[:, None, None]
    return seq / span


def resample(seq: np.ndarray) -> np.ndarray:
    """Any number of frames -> exactly SEQ_LEN, by nearest-index striding."""
    t = seq.shape[0]
    if t == 0:
        return np.zeros((SEQ_LEN, seq.shape[1], N_DIMS))
    idx = np.round(np.arange(SEQ_LEN) * (t - 1) / max(SEQ_LEN - 1, 1)).astype(int)
    return seq[np.minimum(idx, t - 1)]


def standardise(v: np.ndarray) -> np.ndarray:
    m = v.mean()
    s = max(float(v.std()), 1e-6)
    return (v - m) / s


def to_unit(kp: np.ndarray, vid_shape) -> np.ndarray:
    """Dataset-only step: raw pixel (T,75,3) -> unit-square (T,65,3).

    The browser's MediaPipe already returns unit coordinates, so this step
    exists only to bring the dataset into the same space. Everything after
    this point is the shared contract tested by train/test_parity.py.
    """
    if kp.ndim == 4:
        kp = kp[0]
    seq = select_points(kp).astype(np.float64).copy()
    seq[..., 0] /= float(vid_shape[0])
    seq[..., 1] /= float(vid_shape[1])
    return seq


def extract(seq: np.ndarray) -> np.ndarray:
    """SHARED CONTRACT. unit-coordinate (T,65,3) -> flat float32 features.

    Mirrored exactly by extractFeatures() in app/src/lib/features.ts.
    """
    seq = anchor(np.asarray(seq, dtype=np.float64))
    seq = resample(seq)
    return standardise(seq.reshape(-1)).astype(np.float32)


def extract_from_raw(kp: np.ndarray, vid_shape) -> np.ndarray:
    return extract(to_unit(kp, vid_shape))
