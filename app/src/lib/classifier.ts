/**
 * Gloss classifier — the Phase 1 replacement for Phase 0's rule-based classify().
 *
 * Same contract as before: frames in, {gloss, conf} out. Everything around it
 * (gate, UI, speech) is unchanged, which is the whole reason Phase 0 was built
 * with this seam in it.
 */
import * as tf from "@tensorflow/tfjs";
import { extractFeatures, SEQ_LEN, N_POINTS, N_DIMS, type PointFrame } from "./features";
import type { Prediction } from "./gate";

export class GlossClassifier {
  private model: tf.GraphModel | null = null;
  private labels: string[] = [];

  get ready() { return this.model !== null; }
  get vocabulary() { return [...this.labels]; }

  async load(modelUrl: string, labelsUrl: string) {
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

  /** Top-k predictions, for the diagnostics panel. */
  predictTop(frames: PointFrame[], k = 3): { gloss: string; conf: number }[] {
    if (!this.model || frames.length === 0) return [];
    const probs = tf.tidy(() => {
      const feats = extractFeatures(frames);
      const input = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
      return (this.model!.predict(input) as tf.Tensor).dataSync();
    });
    return Array.from(probs)
      .map((conf, i) => ({ gloss: this.labels[i], conf }))
      .sort((a, b) => b.conf - a.conf)
      .slice(0, k);
  }

  /** Runs the model over a rolling buffer of frames. */
  predict(frames: PointFrame[]): Prediction {
    if (!this.model || frames.length === 0) return { gloss: null, conf: 0 };

    const probs = tf.tidy(() => {
      const feats = extractFeatures(frames);
      const input = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
      return (this.model!.predict(input) as tf.Tensor).dataSync();
    });

    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
    const conf = probs[bestIdx];

    // Same floor as Phase 0: an uncertain guess is reported as no sign at all.
    return conf < 0.5
      ? { gloss: null, conf }
      : { gloss: this.labels[bestIdx], conf };
  }

  dispose() { this.model?.dispose(); this.model = null; }
}
