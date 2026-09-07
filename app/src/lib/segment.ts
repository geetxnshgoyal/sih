import type { PointFrame } from './features';

export function motionEnergy(a: PointFrame, b: PointFrame): number {
  const shoulder = Math.max(Math.hypot(b[11].x - b[12].x, b[11].y - b[12].y), 1e-4);
  let sum = 0, live = 0;
  for (let i = 23; i < 65; i++) {
    const p = a[i], q = b[i];
    if ((!p.x && !p.y) || (!q.x && !q.y)) continue;
    sum += Math.hypot(q.x - p.x, q.y - p.y); live++;
  }
  return live ? sum / live / shoulder : 0;
}
export type SegState = 'idle' | 'signing' | 'settling';
export type SegmentRejection = 'no-hands' | 'one-hand' | 'no-pose' | 'too-short' | 'too-long';

/** Apply the trained model's input requirements to automatic AND manual capture. */
export function segmentQuality(frames: PointFrame[]): SegmentRejection | null {
  if (frames.length < 4) return 'too-short';
  let hands = 0, both = 0, poses = 0;
  for (const f of frames) {
    if (f.length !== 65 || f.some(p => ![p.x, p.y, p.z].every(Number.isFinite))) return 'no-pose';
    const left = f.slice(23, 44).some(p => p.x !== 0 || p.y !== 0);
    const right = f.slice(44, 65).some(p => p.x !== 0 || p.y !== 0);
    if (left || right) hands++;
    if (left && right) both++;
    if (Math.hypot(f[11].x - f[12].x, f[11].y - f[12].y) > 0.02) poses++;
  }
  if (poses / frames.length < 0.8) return 'no-pose';
  if (hands / frames.length < 0.6) return 'no-hands';
  if (both / frames.length < 0.6) return 'one-hand';
  return null;
}

type Sample = { frame: PointFrame; at: number };
/** Wall-clock segmentation: behavior must not depend on the laptop's inference FPS. */
export class SignSegmenter {
  static START = 0.012;
  static STOP = 0.006;
  static MIN_FRAMES = 4;
  static MIN_MS = 350;
  // Internal holds are part of moving signs. A 350ms boundary split "How are
  // you" in two and confidently classified its suffix as "healthy". Allow a
  // deliberate end pause, while keeping the whole movement in one sequence.
  static QUIET_MS = 900;
  static MAX_MS = 10000;
  static HAND_GAP_MS = 450;
  private state: SegState = 'idle';
  private samples: Sample[] = [];
  private preroll: Sample[] = [];
  private prev: Sample | null = null;
  private quietAt: number | null = null;
  private gapAt: number | null = null;
  private energy = 0;
  private rejected: SegmentRejection | null = null;
  get current() { return this.state; }
  get length() { return this.samples.length; }
  get progress() { return this.state === 'idle' ? 0 : Math.min(1, this.samples.length / 8); }
  get lastEnergy() { return this.energy; }
  get lastReject() { return this.rejected; }
  reset() {
    this.state = 'idle'; this.samples = []; this.preroll = []; this.prev = null;
    this.quietAt = null; this.gapAt = null; this.energy = 0; this.rejected = null;
  }
  private finish(samples: Sample[]): PointFrame[] | null {
    const frames = samples.map(s => s.frame);
    this.rejected = segmentQuality(frames);
    if (!this.rejected && samples.at(-1)!.at - samples[0].at < SignSegmenter.MIN_MS) this.rejected = 'too-short';
    this.samples = []; this.quietAt = null; this.gapAt = null;
    return this.rejected && this.rejected !== 'one-hand' ? null : frames;
  }
  push(frame: PointFrame, handsVisible: boolean, at = performance.now()): PointFrame[] | null {
    this.rejected = null;
    if (this.prev && at <= this.prev.at) return null;
    // A tab suspension or camera stall is not continuous signing.
    if (this.prev && at - this.prev.at > 1000) this.reset();
    const previous = this.prev;
    this.prev = { frame, at };
    this.energy = previous ? motionEnergy(previous.frame, frame) * (1000 / 30) / Math.max(at - previous.at, 1) : 0;
    if (!handsVisible) {
      this.gapAt ??= at;
      if (this.state === 'signing' && at - this.gapAt >= SignSegmenter.HAND_GAP_MS) {
        this.state = 'idle'; this.preroll = [];
        return this.finish(this.samples);
      }
      if (this.state === 'settling' && at - this.gapAt >= SignSegmenter.HAND_GAP_MS) this.reset();
      return null;
    }
    this.gapAt = null;
    if (this.state === 'settling') {
      // A long unbroken movement must not create a new prediction every N frames.
      if (this.energy <= SignSegmenter.STOP) this.quietAt ??= at;
      else this.quietAt = null;
      if (this.quietAt !== null && at - this.quietAt >= SignSegmenter.QUIET_MS) this.reset();
      return null;
    }
    if (this.state === 'idle') {
      this.preroll.push({ frame, at });
      this.preroll = this.preroll.filter(s => at - s.at <= 350);
      if (this.energy >= SignSegmenter.START) {
        this.state = 'signing'; this.samples = this.preroll.slice(); this.preroll = [];
      }
      return null;
    }
    this.samples.push({ frame, at });
    if (this.energy <= SignSegmenter.STOP) this.quietAt ??= at;
    else this.quietAt = null;
    if (at - this.samples[0].at >= SignSegmenter.MAX_MS) {
      this.state = 'settling'; this.samples = []; this.rejected = 'too-long'; return null;
    }
    if (this.quietAt !== null && at - this.quietAt >= SignSegmenter.QUIET_MS) {
      // Keep the final held shape, but trim the rest of the pause.
      const end = this.quietAt + 100;
      this.state = 'idle';
      return this.finish(this.samples.filter(s => s.at <= end));
    }
    return null;
  }
}
