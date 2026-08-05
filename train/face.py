"""
Non-manual markers — the grammar ISL carries on the face.

Why this file exists
--------------------
Sign languages mark grammar non-manually. In ISL:
  * raised eyebrows      -> yes/no question
  * furrowed eyebrows    -> wh-question (who/what/where)
  * head shake           -> negation, scoped over the manual sign
  * mouth morphemes      -> distinguish otherwise identical handshapes
A hands-only model translates a question as a statement. That is not a
polish issue; it is a correctness issue.

What we can and cannot do with each data source
-----------------------------------------------
OpenHands INCLUDE poses (what we train Phase 1 on):
    33 pose + 42 hand points. Face coverage is 11 coarse pose landmarks
    (nose, 6 eye, 2 ear, 2 mouth-corner), confidence 1.00.
    -> head tilt/shake/nod IS recoverable.   HEAD_ONLY
    -> eyebrows and mouth shape are NOT present at all.

Our own recordings + any re-extraction from INCLUDE raw video:
    full MediaPipe FaceLandmarker, 468 points.
    -> everything above.                     FULL_FACE

Using all 468 face points would swamp the 65 body points, so FULL_FACE keeps
a curated subset: eyebrows, eye aperture, and the lip outline.
"""

# MediaPipe FaceLandmarker (468-point canonical mesh) indices.
LEFT_EYEBROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LEFT_EYE      = [33, 160, 158, 133, 153, 144]      # aperture (EAR-style)
RIGHT_EYE     = [362, 385, 387, 263, 373, 380]
LIPS_OUTER    = [61, 37, 0, 267, 291, 314, 17, 84]
LIPS_INNER    = [78, 81, 13, 311, 308, 402, 14, 178]

FACE_SUBSET = (LEFT_EYEBROW + RIGHT_EYEBROW +
               LEFT_EYE + RIGHT_EYE +
               LIPS_OUTER + LIPS_INNER)
N_FACE = len(FACE_SUBSET)          # 48

# Head-pose landmarks already present in the pose block (indices into the 33).
HEAD_POSE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

MODES = {
    "HEAD_ONLY": 0,      # INCLUDE poses: head movement only, no face block
    "FULL_FACE": N_FACE, # own recordings: + eyebrows, eyes, lips
}


def face_block_size(mode: str) -> int:
    if mode not in MODES:
        raise ValueError(f"unknown face mode {mode!r}; expected one of {list(MODES)}")
    return MODES[mode]
