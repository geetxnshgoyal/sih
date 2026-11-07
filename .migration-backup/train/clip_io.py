"""
The ONE way to read a landmark clip, so a wrong aspect cannot be silent.

Every consumer of data/*_landmarks used to write this:

    aspect = float(npz["aspect"]) if "aspect" in npz else 1.0

and that `else 1.0` was a bug in four places at once. extract_islgov.py predates
the isotropic fix and stores no aspect field, so all 13,662 Government
dictionary clips fell through to 1.0. They are 1920x1080, measured: five random
clips pulled from the source repository are all exactly 1.7778.

The damage was real and quiet. features.isotropic divides y by the aspect, so
those clips were processed as though a 16:9 body were square, stretching every
skeleton vertically by 1.78x. Measured with features.check_isotropy, the
nose-to-shoulder ratio came out at 0.972 against an anatomical band of
0.42-0.80, where NCERT reads 0.557, ISH Shiksha 0.565 and INCLUDE 0.568.

That is the same class of bug that cost 2.1 points across corpora once before,
and then recurred in preprocess_ssl.py with a hardcoded ASPECT = 1.0 for the
same dictionary videos. Third time: no default at all. A corpus whose aspect is
neither stored nor registered here raises.
"""
from pathlib import Path

import numpy as np

# Corpora whose clips carry no aspect field, with the measured value.
# Add to this ONLY with a measurement, never a guess.
KNOWN_ASPECT = {
    # 1920x1080. Measured: five clips pulled at random from the source
    # repository are all exactly 1.7778.
    "islgov_landmarks": 16 / 9,
    # 300x300. Measured for all 612 clips and recorded in
    # data/cislr/_aspect.json, every one of them 1.0.
    "cislr_landmarks": 1.0,
}


class UnknownAspect(Exception):
    """The clip has no aspect and its corpus is not registered. Do not guess."""


def clip_aspect(path: Path, npz) -> float:
    """The source frame's width / height for this clip.

    Prefers what the extractor stored. Falls back only to a MEASURED value
    registered above, keyed on the corpus directory. Otherwise raises, because
    a wrong aspect does not throw: the model still returns a confident answer,
    it is just answering about a differently-shaped body.
    """
    if "aspect" in npz:
        a = float(npz["aspect"])
        if 0.2 < a < 5.0:
            return a
        raise UnknownAspect(f"{path}: stored aspect {a} is not plausible")
    for part in path.parts:
        if part in KNOWN_ASPECT:
            return KNOWN_ASPECT[part]
    raise UnknownAspect(
        f"{path}: no aspect stored and corpus not in clip_io.KNOWN_ASPECT. "
        f"Measure the source video and register it; do not default."
    )


def load_points(path: Path, pose_keep: int, min_active: int = 8, pad: int = 2):
    """One landmark npz -> ((T, 65, 3) unit coordinates, aspect), or (None, None).

    Trimmed to the span where a hand is visible, which is what every
    preprocessor here does and what the training clips were cut to.
    """
    try:
        with np.load(path) as npz:
            if "pose" not in npz or "lh" not in npz or "rh" not in npz:
                return None, None
            present = (np.abs(npz["lh"]).sum(axis=(1, 2)) > 0) | \
                      (np.abs(npz["rh"]).sum(axis=(1, 2)) > 0)
            idx = np.flatnonzero(present)
            if idx.size < min_active:
                return None, None
            lo = max(int(idx[0]) - pad, 0)
            hi = min(int(idx[-1]) + pad + 1, len(present))
            pts = np.concatenate([npz["pose"][:, :pose_keep, :3],
                                  npz["lh"][:, :, :3], npz["rh"][:, :, :3]],
                                 axis=1).astype(np.float64)[lo:hi]
            aspect = clip_aspect(path, npz)
    except UnknownAspect:
        raise
    except Exception:
        return None, None
    if pts.shape[0] < min_active:
        return None, None
    return pts, aspect
