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
  /**
   * How long the cooldown after a discarded over-long sign may last.
   *
   * `settling` used to end only when the energy fell below the quiet level for
   * QUIET_MS. When that level is unreachable, so is the exit: the segmenter sat
   * in `settling` forever and every later sign was ignored, with the hands
   * still in frame and the tracker still running. Recognition was dead until
   * the signer happened to drop their hands out of shot for HAND_GAP_MS.
   *
   * A cooldown is meant to stop one long movement producing a prediction every
   * few frames. A fixed ceiling does that and cannot deadlock.
   */
  static SETTLE_MS = 2500;
  /**
   * Quiet, relative to how hard this signer is moving right now.
   *
   * STOP alone is an absolute floor. It is roughly the right order of
   * magnitude -- measured over real motionless rest taken from
   * data/islgov_landmarks (both hands tracked, hand centroid wandering under
   * 5% of shoulder width, so a person genuinely holding still), at the 15 fps
   * this captures at and through the scaling push() applies:
   *
   *     rest energy   p50 0.0025   p75 0.0048   p90 0.0085   p95 0.0128
   *
   * so 82% of truly still frames fall under STOP = 0.006. But QUIET_MS wants
   * 0.9 s of them UNBROKEN, and the ~18% that spike above it are scattered:
   * 12 of 40 real rest runs never deliver a clean 0.9 s at all. Every spike
   * restarts the count, so a sign that has plainly ended can keep looking
   * unfinished until it hits MAX_MS and is discarded.
   *
   * Judging quiet against the sign's own median instead of a fixed number
   * absorbs most of that: a signer whose rest jitters at 0.008 is still
   * obviously at rest next to their own 0.05 while signing. splitRecording
   * solved the offline half of this problem the same way and with the same
   * constant -- quiet MEANS quiet for this signer, in this recording.
   *
   * Measured end to end, with SETTLE_MS in place. On 250 real clips, clips that
   * produced no segment at all fell 33.2% -> 22.0% and usable segments rose
   * 212 -> 323. On 400 trials of a real trimmed sign followed by real
   * motionless rest, read-nothing fell 37.5% -> 17.5% and exactly-one-segment
   * rose 35.0% -> 51.5%.
   *
   * It is not free: over-splitting rose 27.5% -> 31.0% on that second set. A
   * split sign still produces a reading and a shortlist, where the old
   * behaviour produced silence, so the trade is worth taking -- but raising the
   * fraction further buys recall mostly by cutting more signs in half, which is
   * why it stays at splitRecording's measured 0.35.
   */
  static QUIET_FRACTION = 0.35;
  private state: SegState = 'idle';
  private samples: Sample[] = [];
  private preroll: Sample[] = [];
  private prev: Sample | null = null;
  private quietAt: number | null = null;
  private gapAt: number | null = null;
  private settleAt: number | null = null;
  private energy = 0;
  /** Every energy seen in the current sign, for the adaptive quiet level. */
  private energies: number[] = [];
  /** The adaptive quiet level, which only rises within a sign. */
  private quietBar = 0;
  private rejected: SegmentRejection | null = null;
  get current() { return this.state; }
  get length() { return this.samples.length; }
  get progress() { return this.state === 'idle' ? 0 : Math.min(1, this.samples.length / 8); }
  get lastEnergy() { return this.energy; }
  get lastReject() { return this.rejected; }
  /** Finish the current live sign when the signer chooses its boundary. */
  flush(): PointFrame[] | null {
    if (this.state !== 'signing' || !this.samples.length) return null;
    this.state = 'idle';
    return this.finish(this.samples.slice());
  }
  reset() {
    this.state = 'idle'; this.samples = []; this.preroll = []; this.prev = null;
    this.quietAt = null; this.gapAt = null; this.settleAt = null;
    this.energy = 0; this.energies = []; this.quietBar = 0; this.rejected = null;
  }

  /**
   * The energy below which this sign counts as over.
   *
   * STOP is a floor, not the answer: see QUIET_FRACTION. Until enough of the
   * sign has been seen to have an opinion about it, the floor is all there is.
   *
   * The bar only ever rises within a sign. Taking the median live, without the
   * ratchet, is self-defeating: the still frames at the end of the sign are
   * themselves added to the sample, so a long enough pause drags the median --
   * and with it the bar -- below the very energy that pause is running at, and
   * the sign is never allowed to end. The ratchet keeps the bar where the
   * MOVEMENT put it. A signer who has demonstrated 0.04 of motion in this sign
   * does not get to redefine 0.011 as busy by holding still for long enough.
   */
  private quietLevel(): number {
    if (this.energies.length >= 5) {
      const sorted = [...this.energies].sort((a, b) => a - b);
      const median = sorted[sorted.length >> 1];
      this.quietBar = Math.max(this.quietBar, median * SignSegmenter.QUIET_FRACTION);
    }
    return Math.max(SignSegmenter.STOP, this.quietBar);
  }
  private finish(samples: Sample[]): PointFrame[] | null {
    const frames = samples.map(s => s.frame);
    this.rejected = segmentQuality(frames);
    if (!this.rejected && samples.at(-1)!.at - samples[0].at < SignSegmenter.MIN_MS) this.rejected = 'too-short';
    this.samples = []; this.quietAt = null; this.gapAt = null;
    this.energies = []; this.quietBar = 0;
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
      if (this.energy <= this.quietLevel()) this.quietAt ??= at;
      else this.quietAt = null;
      if (this.quietAt !== null && at - this.quietAt >= SignSegmenter.QUIET_MS) this.reset();
      // ...but the cooldown has to end either way. Waiting only on quiet meant
      // waiting forever whenever quiet was unreachable. See SETTLE_MS.
      else if (this.settleAt !== null && at - this.settleAt >= SignSegmenter.SETTLE_MS) this.reset();
      return null;
    }
    if (this.state === 'idle') {
      this.preroll.push({ frame, at });
      this.preroll = this.preroll.filter(s => at - s.at <= 350);
      if (this.energy >= SignSegmenter.START) {
        this.state = 'signing'; this.samples = this.preroll.slice(); this.preroll = [];
        this.energies = [this.energy]; this.quietBar = 0;
      }
      return null;
    }
    this.samples.push({ frame, at });
    this.energies.push(this.energy);
    if (this.energy <= this.quietLevel()) this.quietAt ??= at;
    else this.quietAt = null;
    if (at - this.samples[0].at >= SignSegmenter.MAX_MS) {
      this.state = 'settling'; this.samples = []; this.settleAt = at;
      this.rejected = 'too-long'; return null;
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

/**
 * What to tell the signer when a live window was rejected, or cleared.
 *
 * Pulled out of SignBridge and made an exhaustive switch on purpose: a reject
 * reason handled in the type but missing a case here fails to compile instead
 * of silently falling through. `no-pose` was exactly that silent case --
 * SegmentRejection has always listed it, but SignBridge's `if/else` chain had
 * no branch for it, so a segment discarded for missing shoulders (reachable by
 * leaning in close: hands stay visible while shoulders leave the frame) said
 * nothing, and the `hasHands` fallback then actively cleared whatever notice
 * was already showing.
 *
 * `hasHands` is passed separately, not read off `reject`, because "no reject
 * and hands are visible" is a real, distinct state (clear the notice) from
 * "no reject and hands are not visible" (say nothing new; the frame simply
 * has not moved enough to matter yet).
 */
export type Notice = { message: string | null; holdMs: number } | 'unchanged';
export function noticeForReject(reject: SegmentRejection | null, hasHands: boolean): Notice {
  switch (reject) {
    case 'no-hands':
      return { message: 'no hands detected. Step back so both hands are in frame', holdMs: 0 };
    case 'one-hand':
      return { message: 'only one hand visible. Confirm the sign below or bring both hands into frame', holdMs: 0 };
    case 'no-pose':
      return { message: 'your shoulders are not fully in frame. Move back a little', holdMs: 4000 };
    case 'too-long':
      return { message: 'that kept moving for 10 seconds, so it was not read. Sign one sign, then pause for a moment.', holdMs: 5000 };
    case 'too-short':
      return { message: 'that was too quick to read. Sign a little slower, then pause.', holdMs: 4000 };
    case null:
      // No rejection this frame. Hands visible: whatever was showing no longer
      // applies, clear it. No hands: this frame simply has not moved enough to
      // become anything yet, which is not news, leave the last notice alone.
      return hasHands ? { message: null, holdMs: 0 } : 'unchanged';
  }
}
