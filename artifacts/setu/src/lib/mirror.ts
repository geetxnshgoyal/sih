/**
 * Left-right reflection of an anchored clip, identical to train/augment.py
 * mirror(): negate x, swap each left/right pose landmark pair, swap the hands.
 *
 * The model was trained on both orientations (every second augmented variant
 * is mirrored), so a mirrored clip is in-distribution. Averaging the model's
 * answer over the clip and its reflection is free test-time augmentation: it
 * cancels whatever handedness bias one orientation carries, and removes the
 * dependency on whether MediaPipe's Left/Right labelling on a selfie camera
 * matches how the training corpus was extracted.
 *
 * Operates on the (SEQ_LEN, 65, 3) anchored tensor from extractAnchored(),
 * before standardisation, exactly where the Python augmentation runs.
 */
import { N_DIMS, N_POINTS, POSE_KEEP, SEQ_LEN } from "./features";

const POSE_PAIRS: [number, number][] = [
  [1, 4], [2, 5], [3, 6], [7, 8], [9, 10],
  [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22],
];
const LEFT_HAND = POSE_KEEP;          // 23..43
const RIGHT_HAND = POSE_KEEP + 21;    // 44..64

export function mirrorAnchored(v: Float64Array): Float64Array {
  const per = N_POINTS * N_DIMS;
  const out = new Float64Array(v.length);
  const frames = Math.min(SEQ_LEN, Math.floor(v.length / per));
  for (let t = 0; t < frames; t++) {
    const base = t * per;
    // build the point permutation for this frame
    for (let i = 0; i < N_POINTS; i++) {
      let src = i;
      if (i >= RIGHT_HAND) src = i - 21;
      else if (i >= LEFT_HAND) src = i + 21;
      else for (const [a, b] of POSE_PAIRS) {
        if (i === a) { src = b; break; }
        if (i === b) { src = a; break; }
      }
      out[base + i * 3 + 0] = -v[base + src * 3 + 0];
      out[base + i * 3 + 1] = v[base + src * 3 + 1];
      out[base + i * 3 + 2] = v[base + src * 3 + 2];
    }
  }
  return out;
}
