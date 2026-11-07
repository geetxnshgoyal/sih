"""
Feature extraction: Python side of the parity contract.

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
POSE_KEEP = 23          # pose landmarks 0..22, includes the 11 head/face points
FACE_MODE = "HEAD_ONLY" # INCLUDE poses carry no face mesh; see train/face.py
N_FACE = face_block_size(FACE_MODE)
N_POINTS = POSE_KEEP + 21 + 21 + N_FACE   # 65 head-only, 113 with full face
N_DIMS = 3
FEATURE_SIZE = SEQ_LEN * N_POINTS * N_DIMS

# Head motion survives normalisation because we anchor at the SHOULDER midpoint,
# not the head: so nod/shake/tilt remain visible to the model.

# indices into the raw 75-point array
POSE = slice(0, POSE_KEEP)
LEFT = slice(33, 54)
RIGHT = slice(54, 75)

# anchor + scale reference, in the 65-point space
L_SHOULDER, R_SHOULDER = 11, 12


def select_points(kp: np.ndarray) -> np.ndarray:
    """(T,75,3) -> (T,65,3), dropping the leg landmarks."""
    return np.concatenate([kp[:, POSE], kp[:, LEFT], kp[:, RIGHT]], axis=1)


def isotropic(seq: np.ndarray, aspect: float) -> np.ndarray:
    """Undo MediaPipe's aspect-dependent normalisation. (T,N,3) -> (T,N,3).

    MediaPipe divides x by the frame WIDTH and y by the frame HEIGHT, so its
    "normalised" coordinates are stretched by the frame's aspect ratio. The same
    skeleton shot at 16:9 and at 1:1 comes out geometrically different:

        source            frame        nose above shoulders
        INCLUDE           1920x1080    1.027 shoulder-widths
        CISLR              300x300     0.552 shoulder-widths
        true anatomy       --          ~0.55-0.70

    1.027 / (1920/1080) = 0.578. The square-format corpus is the correct one;
    INCLUDE is stretched vertically by 1.78 and the model spent its whole life
    learning that stretch as if it were part of the sign. Measured cost: a model
    trained on INCLUDE scores 2.1% on CISLR, barely above the 0.38% chance rate
    (ARCHITECTURE.md 5.1).

    Dividing y by the aspect ratio puts every source, either corpus, and any
    webcam the app runs on: into one isotropic space where a shoulder-width
    means the same thing horizontally and vertically.

    z is left alone: MediaPipe already reports it on roughly the x scale.
    """
    seq = np.asarray(seq, dtype=np.float64).copy()
    seq[..., 1] /= float(aspect)
    return seq


# Nose above the shoulder line, in shoulder widths, for a correctly-scaled
# human. Measured across four independently-produced corpora that are already
# isotropic: CISLR 0.552, MS-ASL 0.559, WLASL 0.572, AUTSL 0.583.
ANATOMY_RATIO = 0.578
ANATOMY_BAND = (0.42, 0.80)


def check_isotropy(anchored: np.ndarray, name: str = "corpus") -> float:
    """Warn if an anchored batch is not in plausible human proportions.

    Accepts (T, N, 3) or (B, T, N, 3) already through anchor().

    This exists because the same bug has now happened twice. MediaPipe's
    coordinates carry the source video's aspect ratio, so a corpus whose aspect
    is guessed wrong comes out stretched, and NOTHING downstream complains: the
    model trains happily and simply fails on anything shaped differently. It
    cost 2.1% cross-corpus accuracy on INCLUDE, and was about to be repeated on
    the ISL dictionary, which was assumed square and is in fact 16:9.

    A ratio near 1.0 means the y axis is still stretched by roughly the frame's
    aspect. Returns the measured ratio so callers can log it.
    """
    a = np.asarray(anchored, dtype=np.float64)
    if a.ndim == 4:
        a = a.reshape(-1, a.shape[-2], a.shape[-1])
    span = np.abs(a[:, L_SHOULDER, 0] - a[:, R_SHOULDER, 0])
    nose = np.abs(a[:, 0, 1] - (a[:, L_SHOULDER, 1] + a[:, R_SHOULDER, 1]) / 2.0)
    ok = span > 1e-6
    if not ok.any():
        return float("nan")
    ratio = float(np.median(nose[ok] / span[ok]))
    lo, hi = ANATOMY_BAND
    if not (lo <= ratio <= hi):
        print(f"  !! {name}: nose/shoulder = {ratio:.3f}, outside {lo}-{hi}. "
              f"The aspect ratio is probably wrong "
              f"(implied {ratio / ANATOMY_RATIO:.2f}x). See features.isotropic.")
    return ratio


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


def extract(seq: np.ndarray, aspect: float) -> np.ndarray:
    """SHARED CONTRACT. unit-coordinate (T,65,3) -> flat float32 features.

    `aspect` is the SOURCE FRAME's width / height. It is required, not
    defaulted: a wrong aspect is silent, the model still returns a confident
    answer, it is just answering about a differently-shaped body. Making every
    caller state it is the only way to keep that from happening again.

    Mirrored exactly by extractFeatures() in app/src/lib/features.ts.
    """
    seq = isotropic(np.asarray(seq, dtype=np.float64), aspect)
    seq = anchor(seq)
    seq = resample(seq)
    return standardise(seq.reshape(-1)).astype(np.float32)


def aspect_of(vid_shape) -> float:
    """Frame width / height for an INCLUDE pose-release clip.

    `vid_shape` is stored (H, W), (1080, 1920) for every clip in the release,
    i.e. LANDSCAPE 1920x1080. The docstring at the top of this file calls it
    (W,H); that is the upstream naming, and to_unit's divisor order matches the
    release's own pre-multiplication, so both are left as they are. Verified:
    to_unit output agrees with MediaPipe run directly on the source video to
    three decimals (1.021 vs 1.027).
    """
    h, w = float(vid_shape[0]), float(vid_shape[1])
    return w / max(h, 1e-9)


def extract_from_raw(kp: np.ndarray, vid_shape) -> np.ndarray:
    return extract(to_unit(kp, vid_shape), aspect_of(vid_shape))
