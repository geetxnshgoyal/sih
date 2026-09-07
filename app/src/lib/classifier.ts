/**
 * Gloss classifier: the Phase 1 replacement for Phase 0's rule-based classify().
 *
 * Same contract as before: frames in, {gloss, conf} out. Everything around it
 * (gate, UI, speech) is unchanged, which is the whole reason Phase 0 was built
 * with this seam in it.
 */
import * as tf from "@tensorflow/tfjs";
import {
  extractFeatures, anchorableCount, SEQ_LEN, N_POINTS, N_DIMS, type PointFrame,
} from "./features";
import type { Prediction } from "./gate";
import { calibrate, TEMPERATURE } from "./calibrate";

// The exported graph has two outputs and tfjs does not promise their order, so
// they are executed by name. Identity is the 83-way softmax, Identity_1 the
// 256-d embedding feeding it. See train/export_tfjs.py.
const OUT_PROBS = "Identity";
const OUT_EMBED = "Identity_1";

/**
 * How much of a clip must carry a usable pose before it is worth classifying.
 *
 * Everything downstream is anchored on the shoulders. When MediaPipe finds the
 * hands but loses the pose, both shoulders arrive as (0,0), anchor()'s 1e-6
 * clamp multiplies every coordinate by a million, and standardise() renormalises
 * the result into a vector that looks entirely ordinary. Measured on exactly
 * that input the model answered "alive" at 99.6% confidence, in the "confident"
 * band, which the app speaks aloud.
 *
 * A majority is the bar: below it the sign is being read off frames that mostly
 * had no body in them, and no answer is the correct answer.
 */
const MIN_ANCHORED = 0.5;

export class GlossClassifier {
  private model: tf.GraphModel | null = null;
  private labels: string[] = [];
  // One model ships, so one temperature. It still lives here rather than being
  // imported at each call site: if a second model ever returns, the compiler
  // points at load() instead of letting a stale constant misreport confidence.
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
      const warm = (this.model as tf.GraphModel).execute(
        tf.zeros([1, SEQ_LEN, N_POINTS * N_DIMS]),
        [OUT_PROBS, OUT_EMBED]
      ) as tf.Tensor[];
      warm.forEach(t => t.dispose());
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
    if (!this.usable(frames, aspect)) return [];
    const probs = tf.tidy(() => {
      const feats = extractFeatures(frames, aspect);
      const input = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
      const [p] = this.model!.execute(input, [OUT_PROBS, OUT_EMBED]) as tf.Tensor[];
      return p.dataSync();
    });
    return Array.from(calibrate(probs, this.temperature))
      .map((conf, i) => ({ gloss: this.labels[i], conf }))
      .sort((a, b) => b.conf - a.conf)
      .slice(0, k);
  }

  /** Runs the model over a rolling buffer of frames. `aspect` = width / height. */
  predict(frames: PointFrame[], aspect: number): Prediction {
    if (!this.model || frames.length === 0) return { gloss: null, conf: 0 };
    if (!this.usable(frames, aspect)) return { gloss: null, conf: 0 };

    const probs = tf.tidy(() => {
      const feats = extractFeatures(frames, aspect);
      const input = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
      const [p] = this.model!.execute(input, [OUT_PROBS, OUT_EMBED]) as tf.Tensor[];
      return p.dataSync();
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

  /**
   * The 256-d embedding for a clip, L2-normalised.
   *
   * This is what the classifier sees just before it commits to one of its 83
   * answers. Two clips of the same sign land near each other here even when
   * the sign is not one of the 83, which is what lets lib/bank.ts recognise
   * words the softmax has no output for. `aspect` = width / height, as ever.
   */
  embed(frames: PointFrame[], aspect: number): Float32Array | null {
    if (!this.model || frames.length === 0) return null;
    if (!this.usable(frames, aspect)) return null;
    // not tf.tidy: it can only return tensors and containers, and this returns
    // a plain array or null, so the two tensors are disposed by hand
    const feats = extractFeatures(frames, aspect);
    const input = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
    let outs: tf.Tensor[] | null = null;
    try {
      outs = this.model.execute(input, [OUT_PROBS, OUT_EMBED]) as tf.Tensor[];
      const v = outs[1].dataSync() as Float32Array;
      let n = 0;
      for (let i = 0; i < v.length; i++) n += v[i] * v[i];
      n = Math.sqrt(n);
      if (!(n > 1e-9)) return null;
      const out = new Float32Array(v.length);
      for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
      return out;
    } finally {
      input.dispose();
      outs?.forEach(t => t.dispose());
    }
  }

  /** Are enough frames anchorable for the answer to mean anything? */
  usable(frames: PointFrame[], aspect: number): boolean {
    if (frames.length === 0) return false;
    return anchorableCount(frames, aspect) / frames.length >= MIN_ANCHORED;
  }

  dispose() { this.model?.dispose(); this.model = null; }
}
