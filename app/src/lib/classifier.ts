/**
 * Gloss classifier: the Phase 1 replacement for Phase 0's rule-based classify().
 *
 * Same contract as before: frames in, {gloss, conf} out. Everything around it
 * (gate, UI, speech) is unchanged, which is the whole reason Phase 0 was built
 * with this seam in it.
 */
import * as tf from "@tensorflow/tfjs";
import { extractFeatures, SEQ_LEN, N_POINTS, N_DIMS, type PointFrame } from "./features";
import type { Prediction } from "./gate";
import { calibrate, TEMPERATURE } from "./calibrate";

export class GlossClassifier {
  private model: tf.GraphModel | null = null;
  private labels: string[] = [];
  // Temperature belongs to the WEIGHTS, not the architecture, so it has to
  // travel with whichever model was loaded. Two ship, and using one model's
  // temperature on the other silently misreports every confidence.
  private temperature = TEMPERATURE;

  get ready() { return this.model !== null; }
  get vocabulary() { return [...this.labels]; }

  async load(modelUrl: string, labelsUrl: string, temperature = TEMPERATURE) {
    this.temperature = temperature;
    const [model, labels] = await Promise.all([
      tf.loadGraphModel(modelUrl),
      fetch(labelsUrl).then(r => r.json() as Promise<string[]>),
    ]);
    this.model = model;
    this.labels = labels;

    // Warm up: the first predict() compiles shaders and can take ~100ms.
    // Doing it now means the first real sign is not the slow one.
    tf.tidy(() => {
      (this.model as tf.GraphModel).predict(
        tf.zeros([1, SEQ_LEN, N_POINTS * N_DIMS])
      ) as tf.Tensor;
    });
  }

  /**
   * Top-k predictions, for the diagnostics panel.
   *
   * `aspect` is the source frame's width / height: see extractFeatures.
   * Passing the wrong one does not throw; it silently classifies a
   * differently-shaped body.
   */
  predictTop(frames: PointFrame[], aspect: number, k = 3): { gloss: string; conf: number }[] {
    if (!this.model || frames.length === 0) return [];
    const probs = tf.tidy(() => {
      const feats = extractFeatures(frames, aspect);
      const input = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
      return (this.model!.predict(input) as tf.Tensor).dataSync();
    });
    return Array.from(calibrate(probs, this.temperature))
      .map((conf, i) => ({ gloss: this.labels[i], conf }))
      .sort((a, b) => b.conf - a.conf)
      .slice(0, k);
  }

  /** Runs the model over a rolling buffer of frames. `aspect` = width / height. */
  predict(frames: PointFrame[], aspect: number): Prediction {
    if (!this.model || frames.length === 0) return { gloss: null, conf: 0 };

    const probs = tf.tidy(() => {
      const feats = extractFeatures(frames, aspect);
      const input = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
      return (this.model!.predict(input) as tf.Tensor).dataSync();
    });

    // Calibrate before reading a confidence off this. Raw softmax here is not a
    // probability: measured on a held-out signer group it said 0.90 while being
    // right 48% of the time (ECE 34.1pp). Temperature scaling brings that to
    // 5.0pp without touching which class wins. See lib/calibrate.ts.
    const cal = calibrate(probs, this.temperature);

    let bestIdx = 0;
    for (let i = 1; i < cal.length; i++) if (cal[i] > cal[bestIdx]) bestIdx = i;
    const conf = cal[bestIdx];

    // No hard floor here any more. The old `conf < 0.5 -> null` was applied to
    // UNCALIBRATED confidence, where 0.5 meant roughly 23% correct, so it let
    // most wrong answers through. Banding now happens in calibrate.certainty()
    // and is surfaced in the UI, so an uncertain read is shown as uncertain
    // rather than silently discarded or confidently announced.
    return { gloss: this.labels[bestIdx], conf };
  }

  dispose() { this.model?.dispose(); this.model = null; }
}
