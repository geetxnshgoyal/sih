/**
 * Temperature scaling — make a stated confidence mean what it says.
 *
 * The problem, measured
 * ---------------------
 * The exported model is a softmax classifier, and softmax confidence on a model
 * this size is not a probability — it is systematically inflated. Measured on
 * 1,566 clips from a signer group the model never trained on, at close range:
 *
 *   says 0.90  ->  right 48% of the time
 *   says 0.99  ->  right 70% of the time
 *   Expected Calibration Error: 34.1pp
 *
 * At the old gate (FLOOR 0.75) that meant 54% of segments were spoken aloud and
 * only 57.9% of those were correct — so 42% of everything the app said was
 * wrong, stated confidently, to a patient.
 *
 * The fix
 * -------
 * Divide the logits by a single scalar T before softmax. T is fitted once,
 * offline, by minimising negative log-likelihood on held-out data. It cannot
 * change which class wins — dividing every logit by the same positive number
 * preserves their order — so accuracy is untouched at 40.4%. All it changes is
 * how confident the model claims to be:
 *
 *   ECE 34.1pp -> 5.0pp        says 0.90 -> right 91%
 *
 * That is the whole point. The model is no better; it now admits it, and the
 * gate can be set from a number that means something.
 *
 * Fitted by the block in run/calib.log, held-out group 0. Refit after any
 * retrain: T is a property of the trained weights, not of the architecture.
 */

/** Fitted on held-out group 0, close range. See the header. */
export const TEMPERATURE = 2.69;

/**
 * Re-apply softmax at temperature T to already-softmaxed probabilities.
 *
 * The shipped graph model has softmax baked into its final layer, so the raw
 * logits are not available at runtime. log(p) recovers them up to a constant,
 * and softmax is invariant to that constant — so this is exactly equivalent to
 * scaling the true logits, without needing to re-export the model.
 */
export function calibrate(probs: ArrayLike<number>, T = TEMPERATURE): Float32Array {
  const n = probs.length;
  const scaled = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    // log(0) would be -Infinity; the epsilon keeps it finite and orders survive
    scaled[i] = Math.log(Math.max(probs[i] as number, 1e-12)) / T;
    if (scaled[i] > max) max = scaled[i];
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    scaled[i] = Math.exp(scaled[i] - max);
    sum += scaled[i];
  }
  for (let i = 0; i < n; i++) scaled[i] /= sum;
  return scaled;
}

export type Certainty = "confident" | "uncertain" | "unusable";

/**
 * How a calibrated confidence should be presented.
 *
 * Measured trade-off on held-out data, after calibration:
 *
 *   floor   speaks    and is right
 *    0.30     49%        60.2%
 *    0.50     23%        75.8%
 *    0.70     10%        84.3%
 *    0.90      2%        97.4%
 *
 * A single hard floor forces a bad choice: high enough to be trustworthy and
 * the app is silent nine times in ten; low enough to be responsive and it is
 * wrong two times in five. Three bands avoid that. Above 0.70 it speaks. Between
 * 0.40 and 0.70 it shows the reading and asks the clinician to confirm rather
 * than announcing it. Below 0.40 it says nothing, because a 1-in-3 guess is
 * worse than silence in a room where symptoms are being recorded.
 */
export const CONFIDENT = 0.70;
export const UNCERTAIN = 0.40;

export function certainty(conf: number): Certainty {
  if (conf >= CONFIDENT) return "confident";
  if (conf >= UNCERTAIN) return "uncertain";
  return "unusable";
}
