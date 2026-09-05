/**
 * Temperature scaling: make a stated confidence mean what it says.
 *
 * The problem, measured
 * ---------------------
 * The exported model is a softmax classifier, and softmax confidence on a model
 * this size is not a probability, it is systematically inflated. Left alone it
 * says 0.90 while being right far less often, and the gate then speaks a
 * confident wrong reading aloud to a patient.
 *
 * The fix
 * -------
 * Divide the logits by a single scalar T before softmax. It cannot change which
 * class wins: dividing every logit by the same positive number preserves their
 * order: so accuracy is untouched. All it changes is how confident the model
 * claims to be:
 *
 *   ECE 23.5pp -> 6.5pp
 *
 * The model is no better; it now admits what it is, and the gate can be set
 * from a number that means something.
 *
 * How T is fitted (train/train_production.py)
 * -------------------------------------------
 * On OUT-OF-FOLD predictions: leave-one-signer-group-out over all 7 groups, so
 * every prediction used to fit T came from a model that had never seen that
 * signer. Fitting on a group the model trained on would produce a T that
 * flatters it. 4,894 predictions, scored at close range because that is the
 * condition the app runs in.
 *
 * REFIT AFTER ANY RETRAIN. T is a property of the trained weights, not of the
 * architecture: train_production.py prints the new value.
 */

/** Fitted out-of-fold over 7 signer groups, close range. See the header. */
export const TEMPERATURE = 2.34;

/**
 * Re-apply softmax at temperature T to already-softmaxed probabilities.
 *
 * The shipped graph model has softmax baked into its final layer, so the raw
 * logits are not available at runtime. log(p) recovers them up to a constant,
 * and softmax is invariant to that constant, so this is exactly equivalent to
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
 *    0.30    72.8%       69.9%
 *    0.40    61.0%       74.9%
 *    0.50    49.4%       80.6%
 *    0.70    30.6%       87.8%
 *    0.90    11.9%       93.4%
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
