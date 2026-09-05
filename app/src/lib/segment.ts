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
  static MIN_HAND_FRACTION = 0.6;

  /** True when at least one hand has real (non-zero) landmarks in this frame. */
  private static frameHasHand(f: PointFrame): boolean {
    for (let i = 23; i < f.length; i++) {
      const p = f[i];
      if (p.x !== 0 || p.y !== 0) return true;
    }
    return false;
  }

  /**
   * Which hands are actually present. Measured separately because ONE hand is
   * not good enough, and treating it as good enough is a real bug we hit.
   */
  private static handsPresent(f: PointFrame): { left: boolean; right: boolean } {
    let left = false, right = false;
    for (let i = 23; i < 23 + 21; i++) {
      if (f[i].x !== 0 || f[i].y !== 0) { left = true; break; }
    }
    for (let i = 23 + 21; i < f.length; i++) {
      if (f[i].x !== 0 || f[i].y !== 0) { right = true; break; }
    }
    return { left, right };
  }

  /**
   * Reject a window that does not hold enough real hand data to be a sign.
   * This is the guard that stops a confident label being produced from nothing.
   */
  static hasEnoughHands(frames: PointFrame[]): boolean {
    if (frames.length === 0) return false;
    let withHands = 0;
    for (const f of frames) if (SignSegmenter.frameHasHand(f)) withHands++;
    return withHands / frames.length >= SignSegmenter.MIN_HAND_FRACTION;
  }

  /**
   * Detect a window where only ONE hand is visible.
   *
   * One-handed windows are usable only with human confirmation. The segmenter
   * marks them via lastReject and still returns the segment; SignBridge then
   * shows candidates but blocks automatic speech.
   */
  static hasBothHands(frames: PointFrame[]): boolean {
    if (frames.length === 0) return false;
    let bothCount = 0;
    for (const f of frames) {
      const { left, right } = SignSegmenter.handsPresent(f);
      if (left && right) bothCount++;
    }
    return bothCount / frames.length >= SignSegmenter.MIN_HAND_FRACTION;
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

/**
 * Split a RECORDING into signs, offline.
 *
 * The live segmenter has to decide "has the sign ended?" from the frames it has
 * seen so far, with no view of what comes next. That is why it is brittle: a
 * 0.4 s hold inside a compound sign looks identical, in the moment, to the end
 * of a sign. Measured on five real clips of Thank you and Good Morning it
 * emitted nothing at all, while classifying the same clips whole gave
 * Thank you 99% and Morning 93%.
 *
 * A recording has no such problem. The whole sequence is in hand, so the
 * threshold can be set FROM the recording rather than guessed ahead of it, and
 * a pause is only a boundary if it is quiet relative to the rest of this
 * particular recording by this particular signer.
 *
 * Returns one entry per sign, in order. A recording holding a single sign
 * returns a single entry, which is the case that matters most: it is then
 * exactly the shape of a training clip.
 */
/**
 * How long a pause must be, in frames at 15 fps, before it separates two signs
 * rather than being a hold inside one.
 *
 * Swept against five real clips of Thank you and Good Morning, plus the two of
 * them concatenated with a rest between:
 *
 *     gap     single signs stay whole   the pair splits into
 *     0.33s   no, 2 of 5 over-split     4
 *     0.80s   no, 1 of 5 over-split     3
 *     1.00s   yes                       3
 *     1.20s   yes                       2   correct
 *     1.47s   yes                       1   boundary missed
 *
 * 1.2 s is the only value tested that gets both right, and it is a rule a
 * person can be told: pause about a second between signs. Below it lies the
 * hold inside a compound sign, which is what broke the live segmenter.
 */
export const SIGN_GAP_FRAMES = 18;
export function splitRecording(frames: PointFrame[], minGap = SIGN_GAP_FRAMES, quietFrac = 0.35): PointFrame[][] {
  if (frames.length < SignSegmenter.MIN_FRAMES) return [];

  const energy: number[] = [0];
  for (let i = 1; i < frames.length; i++) {
    energy.push(motionEnergy(frames[i - 1], frames[i]));
  }
  // three-frame mean: MediaPipe jitter puts single-frame spikes in the middle
  // of a genuine pause, and a spike is enough to stop a run being a boundary
  const smooth = energy.map((_, i) => {
    const a = energy[Math.max(0, i - 1)], b = energy[i];
    const c = energy[Math.min(energy.length - 1, i + 1)];
    return (a + b + c) / 3;
  });

  // Adaptive: quiet MEANS quiet for this signer, in this recording. A fixed
  // threshold cannot serve both someone signing briskly and someone deliberate.
  const sorted = [...smooth].filter(v => v > 0).sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length * 0.5)] : 0;
  const quiet = Math.max(SignSegmenter.STOP, median * quietFrac);

  /** a pause must last this long to be a boundary rather than a hold */
  const MIN_GAP = minGap;       // frames, at the 15 fps everything is sampled at

  const runs: [number, number][] = [];
  let s = -1;
  for (let i = 0; i < smooth.length; i++) {
    if (smooth[i] <= quiet) { if (s < 0) s = i; }
    else { if (s >= 0 && i - s >= MIN_GAP) runs.push([s, i]); s = -1; }
  }
  if (s >= 0 && smooth.length - s >= MIN_GAP) runs.push([s, smooth.length]);

  // cut at the middle of each interior pause; leading and trailing pauses are
  // rest, not boundaries, and are trimmed rather than split on
  const cuts = runs
    .filter(([a, b]) => a > 0 && b < smooth.length)
    .map(([a, b]) => Math.floor((a + b) / 2));

  const bounds = [0, ...cuts, frames.length];
  const out: PointFrame[][] = [];
  for (let i = 0; i < bounds.length - 1; i++) {
    let lo = bounds[i], hi = bounds[i + 1];
    while (lo < hi && smooth[lo] <= quiet) lo++;          // trim leading rest
    while (hi > lo && smooth[hi - 1] <= quiet) hi--;      // trim trailing rest
    const piece = frames.slice(lo, hi);
    if (piece.length >= SignSegmenter.MIN_FRAMES &&
        SignSegmenter.hasEnoughHands(piece)) out.push(piece);
  }
  return out;
}
