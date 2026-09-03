import type { PointFrame } from "./features";

/**
 * Decide where a sign starts and stops.
 *
 * Why this is necessary
 * ---------------------
 * Every training clip is trimmed to exactly one sign. Feeding the model a
 * continuous rolling window instead — which is what the app did — mixes rest
 * position and transitions into the same 32 resampled frames, and accuracy
 * falls off a cliff:
 *
 *     trimmed clip, as trained ................ 66.0%
 *     rolling window, 10 rest frames .......... 54.8%
 *     rolling window, 20 rest frames .......... 40.4%
 *     rolling window, 40 rest frames .......... 23.2%
 *
 * Live, the buffer is mostly idle, so it sat at the bottom of that range. No
 * amount of model work fixes it; the input has to be segmented the same way
 * the training data was.
 *
 * Motion energy is measured on the hands only, normalised by shoulder width so
 * it does not depend on how close the signer sits.
 */

const L_SH = 11;
const R_SH = 12;
const HANDS_FROM = 23;

/** Mean per-frame hand displacement, in shoulder-widths. */
export function motionEnergy(a: PointFrame, b: PointFrame): number {
  const ls = b[L_SH], rs = b[R_SH];
  const shoulder = Math.max(Math.hypot(ls.x - rs.x, ls.y - rs.y), 1e-4);
  let sum = 0;
  let live = 0;
  for (let i = HANDS_FROM; i < a.length; i++) {
    const p = a[i], q = b[i];
    // zero means the hand was not detected in that frame
    if ((p.x === 0 && p.y === 0) || (q.x === 0 && q.y === 0)) continue;
    sum += Math.hypot(q.x - p.x, q.y - p.y);
    live++;
  }
  return live ? sum / live / shoulder : 0;
}

export type SegState = "idle" | "signing" | "settling";

/**
 * Watches motion and emits a trimmed segment when a sign completes.
 *
 * Thresholds are deliberately asymmetric: it takes a clear burst of movement
 * to start, but only a short lull to stop, so a sign is cut at its natural end
 * rather than running into the next one.
 */
export class SignSegmenter {
  private state: SegState = "idle";
  private frames: PointFrame[] = [];
  private quiet = 0;
  /** consecutive frames with no hand detected, inside a segment */
  private gap = 0;
  private prev: PointFrame | null = null;
  private energy: number[] = [];

  /** movement above this (shoulder-widths/frame) means signing has begun */
  static START = 0.012;
  /** below this counts as still */
  static STOP = 0.006;
  /** consecutive still frames that end a sign */
  static QUIET_FRAMES = 6;
  /** ignore blips too short to be a sign */
  static MIN_FRAMES = 12;
  /** stop runaway segments if someone never goes still */
  static MAX_FRAMES = 90;
  /**
   * Least fraction of a segment's frames that must actually contain a hand.
   *
   * MediaPipe drops hands intermittently — a flicker is enough to start a
   * segment, and `handsVisible` only describes the CURRENT frame, so a buffer
   * can accumulate that is mostly hand-less. Classifying that is not a
   * near-miss, it is meaningless: zero-filled hands anchor to a single fixed
   * point far outside the training manifold, the softmax saturates, and the
   * model returns an arbitrary class at ~100%. Measured directly against the
   * trained model — an all-zero-hands input returns 'Temple' at 100.0%
   * regardless of torso pose or scale.
   *
   * A sign is made with the hands. If most of the window has none, there is no
   * sign in it, and saying nothing is the correct output.
   */
  static MIN_HAND_FRACTION = 0.6;
  /**
   * Consecutive hand-less frames tolerated inside a sign before abandoning it.
   *
   * MediaPipe loses the hands for a frame or two mid-sign routinely — on motion
   * blur, on self-occlusion, when a hand crosses the face. Abandoning on the
   * first miss throws away good signs. Tolerating a short gap and letting
   * MIN_HAND_FRACTION judge the finished window is both more forgiving of real
   * tracking and stricter about what actually gets classified.
   */
  static HAND_GAP_TOLERANCE = 5;

  get current(): SegState { return this.state; }
  get length(): number { return this.frames.length; }
  /** 0..1, how much of a plausible sign has been captured — drives the UI ring */
  get progress(): number {
    if (this.state === "idle") return 0;
    return Math.min(1, this.frames.length / SignSegmenter.MIN_FRAMES);
  }
  get lastEnergy(): number { return this.energy.at(-1) ?? 0; }

  reset() {
    this.state = "idle";
    this.frames = [];
    this.quiet = 0;
    this.gap = 0;
    this.prev = null;
    this.energy = [];
  }

  /** True when at least one hand has real (non-zero) landmarks in this frame. */
  private static frameHasHand(f: PointFrame): boolean {
    for (let i = HANDS_FROM; i < f.length; i++) {
      const p = f[i];
      if (p.x !== 0 || p.y !== 0) return true;
    }
    return false;
  }

  /**
   * Which hands are actually present. Measured separately because ONE hand is
   * not good enough, and treating it as good enough is a real bug we hit.
   *
   * Measured on 400 held-out clips, zeroing one hand's 21 landmarks:
   *
   *   both hands      92.8% top-1, 208/264 classes predicted, conf 0.96
   *   left missing    22.8% top-1,  86/264 classes,           conf 0.77
   *   right missing   12.8% top-1,  57/264 classes,           conf 0.75
   *
   * Accuracy collapses and predictions pile onto a few attractors — Car, Truck,
   * Mouse, Today, Religion. Crucially the CONFIDENCE stays around 0.75-0.77,
   * i.e. above gate.FLOOR, so these get announced as if they were real reads.
   * That is the "it always says Truck" symptom: at close range one hand drifts
   * out of frame, the old one-hand-is-enough check passed, and half the input
   * was zeros.
   */
  private static handsPresent(f: PointFrame): { left: boolean; right: boolean } {
    let left = false, right = false;
    for (let i = HANDS_FROM; i < HANDS_FROM + 21; i++) {
      if (f[i].x !== 0 || f[i].y !== 0) { left = true; break; }
    }
    for (let i = HANDS_FROM + 21; i < f.length; i++) {
      if (f[i].x !== 0 || f[i].y !== 0) { right = true; break; }
    }
    return { left, right };
  }

  /**
   * Reject a window that does not hold enough real hand data to be a sign.
   * This is the guard that stops a confident label being produced from nothing.
   */
  private static hasEnoughHands(frames: PointFrame[]): boolean {
    if (frames.length === 0) return false;
    let withHands = 0;
    for (const f of frames) if (SignSegmenter.frameHasHand(f)) withHands++;
    return withHands / frames.length >= SignSegmenter.MIN_HAND_FRACTION;
  }

  /**
   * Reject a window where only ONE hand is visible.
   *
   * Not an arbitrary strictness: ALL 4,284 INCLUDE training clips have both
   * hands present for the majority of frames — measured, zero exceptions. The
   * model has therefore never seen a one-handed input, so feeding it one is
   * fully out of distribution. It does not fail loudly; it falls onto a few
   * attractor classes (Car, Truck, Mouse, Today) at ~0.76 confidence, which is
   * above gate.FLOOR and so gets spoken aloud as if it were a real reading.
   *
   * This does mean genuinely one-handed signing is refused rather than guessed
   * at. That is the correct trade for a medical kiosk: "move your other hand
   * into frame" is recoverable, "you have a fever" when the patient signed
   * something else is not. When one-handed training data exists, relax this.
   */
  private static hasBothHands(frames: PointFrame[]): boolean {
    if (frames.length === 0) return false;
    let bothCount = 0;
    for (const f of frames) {
      const { left, right } = SignSegmenter.handsPresent(f);
      if (left && right) bothCount++;
    }
    return bothCount / frames.length >= SignSegmenter.MIN_HAND_FRACTION;
  }

  /** Why the last window was discarded — surfaced in the UI, not swallowed. */
  private _lastReject: "no-hands" | "one-hand" | null = null;
  get lastReject() { return this._lastReject; }

  /** Feed one frame. Returns a trimmed segment when a sign has just ended. */
  push(frame: PointFrame, handsVisible: boolean): PointFrame[] | null {
    this._lastReject = null;

    if (!handsVisible) {
      if (this.state !== "signing") {
        this.reset();
        return null;
      }

      // Mid-sign dropout. Keep the frame (it carries the body, and dropping it
      // would distort timing) and keep going, up to the tolerance.
      this.gap++;
      this.frames.push(frame);

      if (this.gap <= SignSegmenter.HAND_GAP_TOLERANCE) return null;

      // Gap too long — the hands are genuinely gone. Emit only if what we have
      // is mostly real hand data; `handsVisible` describes one frame, so the
      // buffer behind it can be almost entirely empty.
      const inProgress = this.frames.slice(0, this.frames.length - this.gap);
      this.reset();
      if (inProgress.length < SignSegmenter.MIN_FRAMES) return null;
      if (!SignSegmenter.hasEnoughHands(inProgress)) {
        this._lastReject = "no-hands";
        return null;
      }
      if (!SignSegmenter.hasBothHands(inProgress)) {
        this._lastReject = "one-hand";
        return null;
      }
      return inProgress;
    }
    this.gap = 0;

    const e = this.prev ? motionEnergy(this.prev, frame) : 0;
    this.prev = frame;
    this.energy.push(e);
    if (this.energy.length > 120) this.energy.shift();

    if (this.state === "idle") {
      if (e >= SignSegmenter.START) {
        this.state = "signing";
        this.frames = [frame];
        this.quiet = 0;
      }
      return null;
    }

    this.frames.push(frame);
    this.quiet = e <= SignSegmenter.STOP ? this.quiet + 1 : 0;

    const ended = this.quiet >= SignSegmenter.QUIET_FRAMES;
    const tooLong = this.frames.length >= SignSegmenter.MAX_FRAMES;

    if (ended || tooLong) {
      // drop the trailing still frames — they are rest, not sign
      const trimmed = ended
        ? this.frames.slice(0, Math.max(1, this.frames.length - SignSegmenter.QUIET_FRAMES))
        : this.frames.slice();
      this.state = "idle";
      this.frames = [];
      this.quiet = 0;
      if (trimmed.length < SignSegmenter.MIN_FRAMES) return null;
      // Same guard as the hands-gone path: a window can reach its natural end
      // having lost the hands partway through, and that is equally unclassifiable.
      if (!SignSegmenter.hasEnoughHands(trimmed)) {
        this._lastReject = "no-hands";
        return null;
      }
      if (!SignSegmenter.hasBothHands(trimmed)) {
        this._lastReject = "one-hand";
        return null;
      }
      return trimmed;
    }
    return null;
  }
}
