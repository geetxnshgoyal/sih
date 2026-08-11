/**
 * Stability gate. Silence beats a wrong word.
 *
 * A prediction only fires when it has held steady across a window of frames
 * AND cleared the confidence floor. Carried over unchanged from Phase 0 —
 * this logic is model-independent by design.
 */

export const FLOOR = 0.75;      // below this, say nothing
export const WINDOW = 14;       // frames considered
export const NEEDED = 10;       // of WINDOW that must agree
export const COOLDOWN = 1800;   // ms before the same gloss may repeat

export type Prediction = { gloss: string | null; conf: number };
export type GateResult = { fire: string | null; conf: number; progress: number };

export class StabilityGate {
  /**
   * Judge one prediction made on an already-segmented sign.
   *
   * The N-of-M window below exists to reject flicker when classifying a rolling
   * buffer many times a second. Once a segmenter hands over a single trimmed
   * sign, that job is already done — requiring ten agreeing frames would mean
   * nothing ever fires, since there is only one prediction per sign. The
   * confidence floor and the repeat cooldown still apply.
   */
  once({ gloss, conf }: Prediction, now = performance.now()): GateResult {
    if (!gloss || conf < FLOOR) return { fire: null, conf, progress: 0 };
    if (this.last.gloss === gloss && now - this.last.t < COOLDOWN) {
      return { fire: null, conf, progress: 1 };
    }
    this.last = { gloss, t: now };
    return { fire: gloss, conf, progress: 1 };
  }

  private window: Prediction[] = [];
  protected last: { gloss: string | null; t: number } = { gloss: null, t: 0 };

  reset() {
    this.window = [];
    this.last = { gloss: null, t: 0 };
  }

  push({ gloss, conf }: Prediction, now = performance.now()): GateResult {
    this.window.push({ gloss, conf });
    if (this.window.length > WINDOW) this.window.shift();

    const bins = new Map<string, number[]>();
    for (const p of this.window) {
      if (!p.gloss) continue;
      const arr = bins.get(p.gloss) ?? [];
      arr.push(p.conf);
      bins.set(p.gloss, arr);
    }

    let best: string | null = null, count = 0, meanConf = 0;
    for (const [gloss, confs] of bins) {
      if (confs.length > count) {
        best = gloss;
        count = confs.length;
        meanConf = confs.reduce((a, b) => a + b, 0) / confs.length;
      }
    }

    const progress = best
      ? Math.min(1, count / NEEDED) * (meanConf >= FLOOR ? 1 : 0.55)
      : 0;

    const ready = best !== null && count >= NEEDED && meanConf >= FLOOR;
    const blocked = this.last.gloss === best && now - this.last.t < COOLDOWN;

    if (ready && !blocked) {
      this.last = { gloss: best, t: now };
      this.window = [];
      return { fire: best, conf: meanConf, progress: 1 };
    }
    return { fire: null, conf: meanConf, progress };
  }
}
