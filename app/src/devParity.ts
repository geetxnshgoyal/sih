/**
 * Dev-only check: does TF.js in the browser reproduce Python's prediction?
 *
 * The feature extractors are already proven equal (train/test_parity.py).
 * This closes the remaining gap: that the exported weights and architecture
 * survived the Keras -> H5 -> TF.js round trip intact.
 *
 * Runs on load in dev, logs to the console, ships nowhere near production.
 */
import * as tf from "@tensorflow/tfjs";
import { extractFeatures, SEQ_LEN, N_POINTS, N_DIMS, type PointFrame } from "./lib/features";
import { asset } from "./lib/assetUrl";

export async function checkModelParity() {
  try {
    const [ref, labels, model] = await Promise.all([
      fetch(asset("/model/_ref.json")).then((r) => r.json()),
      fetch(asset("/model/labels.json")).then((r) => r.json()) as Promise<string[]>,
      tf.loadGraphModel("/model/model.json"),
    ]);

    const rows: number[][] = ref.input;
    const flat = rows.flat();
    const input = tf.tensor(flat, [1, rows.length, rows[0].length]);
    const probs = Array.from((model.predict(input) as tf.Tensor).dataSync());
    input.dispose();

    let maxDiff = 0;
    for (let i = 0; i < probs.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(probs[i] - ref.probs[i]));
    }
    const top = probs
      .map((p, i) => [p, i] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 3);

    const agree = labels[top[0][1]] === ref.top1;
    console.log("[parity] backend      :", tf.getBackend());
    console.log("[parity] classes      :", labels.length);
    console.log("[parity] python top-1 :", ref.top1);
    console.log("[parity] tfjs   top-1 :", labels[top[0][1]]);
    console.log("[parity] tfjs   top-3 :", top.map(([p, i]) => `${labels[i]} ${p.toFixed(5)}`));
    console.log("[parity] max prob diff:", maxDiff.toExponential(3));
    console.log(agree && maxDiff < 1e-3 ? "[parity] PASS" : "[parity] FAIL");
  } catch (e) {
    console.error("[parity] error", e);
  }
}

/** End-to-end: raw unit frames -> extractFeatures -> model, vs Python. */
export async function checkClipParity() {
  try {
    const [clips, labels, model] = await Promise.all([
      fetch(asset("/model/_demo.json")).then((r) => r.json()),
      fetch(asset("/model/labels.json")).then((r) => r.json()) as Promise<string[]>,
      tf.loadGraphModel("/model/model.json"),
    ]);
    console.log("[clip] name              py-pred          ts-pred          py-conf ts-conf");
    for (const c of clips) {
      const frames: PointFrame[] = c.frames.map((f: number[][]) =>
        f.map(([x, y, z]) => ({ x, y, z }))
      );
      const feats = extractFeatures(frames);
      const t = tf.tensor(feats, [1, SEQ_LEN, N_POINTS * N_DIMS]);
      const probs = Array.from((model.predict(t) as tf.Tensor).dataSync());
      t.dispose();
      let bi = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bi]) bi = i;
      const match = labels[bi] === c.pred ? "  " : "!!";
      console.log(
        `[clip]${match} ${c.true.padEnd(16)} ${String(c.pred).padEnd(16)} ` +
        `${labels[bi].padEnd(16)} ${c.conf.toFixed(2)}    ${probs[bi].toFixed(2)}`
      );
    }
  } catch (e) {
    console.error("[clip] error", e);
  }
}
