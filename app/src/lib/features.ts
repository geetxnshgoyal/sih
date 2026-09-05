/**
 * Feature extraction — THE PARITY-CRITICAL FILE.
 *
 * This is the exact mirror of train/features.py. If the two drift, the model
 * scores well offline and fails on camera, with almost no visible symptom.
 * train/test_parity.py runs both on identical input and asserts equality.
 * Change one side, change the other, re-run the test. No exceptions.
 *
 * SHARED CONTRACT
 *   input : unit-coordinate sequence, (T, N_POINTS, 3), plus the source
 *           frame's aspect ratio (width / height)
 *   isotropic: y /= aspect, undoing MediaPipe's aspect-dependent normalisation
 *   anchor: shoulder midpoint origin, shoulder-width scale
 *   resample: nearest-index stride to exactly SEQ_LEN frames
 *   standardise: zero mean, unit std over the whole flattened vector
 *   output: Float32Array(SEQ_LEN * N_POINTS * 3)
 *
 * Point layout (must match train/features.py select_points):
 *   [ 0..22]  pose 0-22   — head, shoulders, arms (legs dropped: extrapolated,
 *                           mean confidence 0.25 in the source data)
 *   [23..43]  left hand   — 21 MediaPipe hand landmarks
 *   [44..64]  right hand  — 21 MediaPipe hand landmarks
 *   [65..]    face subset — only when FACE_MODE is FULL_FACE (see train/face.py)
 */

export const SEQ_LEN = 32;
export const POSE_KEEP = 23;
export const N_FACE = 0;                 // HEAD_ONLY — mirrors FACE_MODE in features.py
export const N_POINTS = POSE_KEEP + 21 + 21 + N_FACE;   // 65
export const N_DIMS = 3;
export const FEATURE_SIZE = SEQ_LEN * N_POINTS * N_DIMS;

/** Indices within the assembled point array. */
export const L_SHOULDER = 11;
export const R_SHOULDER = 12;

export type Landmark = { x: number; y: number; z: number };
/** One frame: exactly N_POINTS landmarks, already in unit coordinates. */
export type PointFrame = Landmark[];

const ZERO: Landmark = { x: 0, y: 0, z: 0 };

/**
 * Assemble MediaPipe Tasks output into the contract's point order.
 * Missing hands are zero-filled, matching how the source dataset stores them.
 */
export function assembleFrame(
  pose: Landmark[] | null,
  left: Landmark[] | null,
  right: Landmark[] | null
): PointFrame {
  const out: PointFrame = new Array(N_POINTS);
  for (let i = 0; i < POSE_KEEP; i++) out[i] = pose?.[i] ?? ZERO;
  for (let i = 0; i < 21; i++) out[POSE_KEEP + i] = left?.[i] ?? ZERO;
  for (let i = 0; i < 21; i++) out[POSE_KEEP + 21 + i] = right?.[i] ?? ZERO;
  return out;
}

/**
 * Shoulder-midpoint origin, shoulder-width scale, in ISOTROPIC coordinates.
 * Mirrors isotropic() + anchor() in Python, fused into one pass.
 *
 * `aspect` is the source frame's width / height. MediaPipe divides x by the
 * frame width and y by the frame height, so its "normalised" output is
 * stretched by the aspect ratio — the same skeleton at 16:9 and at 1:1 is
 * geometrically different. Dividing y by the aspect undoes that.
 *
 * This is not cosmetic. A model trained on 16:9 INCLUDE video scored 2.1% on
 * square CISLR video against a 0.38% chance rate, purely because of this
 * (ARCHITECTURE.md 5.1). The app is subject to the same failure whenever the
 * webcam's shape differs from the training corpus — so the aspect must come
 * from the live video element, never a constant.
 *
 * Division rather than multiplying by a reciprocal: Python does `y /= aspect`
 * element-wise, and `y * (1/aspect)` rounds differently. test_parity.py asserts
 * equality to 0.000e+00, so that difference would surface as a failure.
 */
function anchor(seq: PointFrame[], aspect: number): Float64Array {
  const per = N_POINTS * N_DIMS;
  const out = new Float64Array(seq.length * per);

  for (let t = 0; t < seq.length; t++) {
    const f = seq[t];
    const ls = f[L_SHOULDER], rs = f[R_SHOULDER];
    const lsy = ls.y / aspect, rsy = rs.y / aspect;
    const mx = (ls.x + rs.x) / 2, my = (lsy + rsy) / 2, mz = (ls.z + rs.z) / 2;

    // shoulder width uses x,y only — matches np.linalg.norm(..., :2) in Python
    const dx = ls.x - rs.x, dy = lsy - rsy;
    const span = Math.max(Math.hypot(dx, dy), 1e-6);

    const base = t * per;
    for (let i = 0; i < N_POINTS; i++) {
      const p = f[i];
      out[base + i * 3 + 0] = (p.x - mx) / span;
      out[base + i * 3 + 1] = (p.y / aspect - my) / span;
      out[base + i * 3 + 2] = (p.z - mz) / span;
    }
  }
  return out;
}

/** Nearest-index striding to exactly SEQ_LEN frames. Mirrors resample(). */
function resampleIndices(t: number): number[] {
  const idx: number[] = new Array(SEQ_LEN);
  const denom = Math.max(SEQ_LEN - 1, 1);
  for (let i = 0; i < SEQ_LEN; i++) {
    // Unreachable at SEQ_LEN=32 (denominator 31 is prime, so i*(t-1)/31
    // never lands on .5) but kept: changing SEQ_LEN makes it reachable, and
    // numpy rounds half-to-even while Math.round rounds half-up.
    const raw = (i * (t - 1)) / denom;
    idx[i] = Math.min(roundHalfToEven(raw), t - 1);
  }
  return idx;
}

/** numpy's round-half-to-even, so index selection cannot diverge from Python. */
function roundHalfToEven(v: number): number {
  const floor = Math.floor(v);
  const diff = v - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Zero mean, unit std over the flattened vector. Mirrors standardise(). */
function standardise(v: Float64Array): Float32Array {
  let mean = 0;
  for (let i = 0; i < v.length; i++) mean += v[i];
  mean /= v.length;

  let acc = 0;
  for (let i = 0; i < v.length; i++) acc += (v[i] - mean) ** 2;
  const std = Math.max(Math.sqrt(acc / v.length), 1e-6);

  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = (v[i] - mean) / std;
  return out;
}

/**
 * SHARED CONTRACT entry point.
 *
 * `aspect` is the source frame's width / height — for live camera,
 * `video.videoWidth / video.videoHeight`. It is required, not defaulted: a
 * wrong aspect is silent. The model still returns a confident answer, it is
 * just answering about a differently-shaped body.
 */
export function extractFeatures(seq: PointFrame[], aspect: number): Float32Array {
  const per = N_POINTS * N_DIMS;
  if (seq.length === 0) return new Float32Array(FEATURE_SIZE);

  const anchored = anchor(seq, aspect);
  const idx = resampleIndices(seq.length);

  const picked = new Float64Array(FEATURE_SIZE);
  for (let i = 0; i < SEQ_LEN; i++) {
    picked.set(anchored.subarray(idx[i] * per, idx[i] * per + per), i * per);
  }
  return standardise(picked);
}
