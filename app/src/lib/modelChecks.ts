import * as tf from '@tensorflow/tfjs';
import { GlossClassifier } from './classifier';
import { fetchJson } from './assets';
import { loadSignLibrary } from './signLibrary';
import { segmentQuality } from './segment';
import type { PointFrame } from './features';

export type Clip = { true: string; frames: number[][][] };
export function parseRecordings(data: unknown): Clip[] {
  const doc = data as { format?: string; takes?: { gloss: string; frames: number[][][] }[] };
  if (doc?.format !== 'setu-recordings-v1' || !Array.isArray(doc.takes) || !doc.takes.length) throw new Error('Choose a nonempty Setu recordings JSON file.');
  return doc.takes.map(t => {
    if (!t || typeof t.gloss !== 'string' || !t.gloss.trim() || !Array.isArray(t.frames) || t.frames.length < 4 || t.frames.length > 2000 || t.frames.some(f => !Array.isArray(f) || f.length !== 65 || f.some(p => !Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite)))) {
      throw new Error('Invalid recording: each take needs 4–2000 frames of 65 finite 3D landmarks.');
    }
    return { true: t.gloss, frames: t.frames };
  });
}
export async function evaluateClips(clips: Clip[]): Promise<{ expected: string; predicted: string; score: number; accepted: boolean; reason: string | null }[]> {
  const classifier = new GlossClassifier();
  try {
    await classifier.load();
    const results = [];
    for (const clip of clips) {
      const frames: PointFrame[] = clip.frames.map(f => f.map(([x,y,z]) => ({x,y,z})));
      const reason = segmentQuality(frames);
      const [top] = reason && reason !== 'one-hand' ? [] : classifier.predictTop(frames, 1);
      results.push({ expected: clip.true, predicted: top?.gloss ?? 'Not classified', score: top?.conf ?? 0,
        accepted: !reason && (top?.conf ?? 0) >= 0.75, reason: reason ?? (classifier.vocabulary.includes(clip.true) ? null : 'Label is not in the trained vocabulary') });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    return results;
  } finally { classifier.dispose(); }
}
export async function checkModel(): Promise<string[]> {
  const classifier = new GlossClassifier();
  try {
    const [, library] = await Promise.all([classifier.load(), loadSignLibrary()]);
    const ref = await fetchJson<{ input: number[][]; probs: number[]; top1: string }>('/model/_ref.json');
    const model = await tf.loadGraphModel('/model/model.json');
    try {
      const probs = tf.tidy(() => Array.from((model.predict(tf.tensor(ref.input.flat(), [1,32,195])) as tf.Tensor).dataSync()));
      if (probs.length !== ref.probs.length) throw new Error('Reference belongs to a different model. Re-export it.');
      const diff = Math.max(...probs.map((p,i) => Math.abs(p-ref.probs[i])));
      if (!Number.isFinite(diff) || diff > 0.001) throw new Error(`Model parity failed: max probability difference ${diff}`);
      return [
        `Recognition model loaded and inference passed: ${classifier.vocabulary.length} labels on ${tf.getBackend()}`,
        `Python/export parity passed: max probability difference ${diff.toExponential(2)}`,
        `${Object.keys(library).length} validated playback recordings`,
        `${classifier.vocabulary.filter(g => !library[g]).length} recognition labels have no playback recording`,
      ];
    } finally { model.dispose(); }
  } finally { classifier.dispose(); }
}
