import { useEffect, useMemo, useRef } from "react";
import type { SignFaceFrame, SignFrame } from "../lib/reverse";
import { playbackBounds, trackedHand } from "../lib/signGeometry";

const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
] as const;
const PALM = [0, 5, 9, 13, 17] as const;
const LEFT_BASE = 23;
const RIGHT_BASE = 44;
const LOGICAL_WIDTH = 480;
const LOGICAL_HEIGHT = 420;
const RENDER_SCALE = 2;
type XY = readonly [number, number];

function line(ctx: CanvasRenderingContext2D, points: XY[], close = false) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (const p of points.slice(1)) ctx.lineTo(...p);
  if (close) ctx.closePath();
}

function drawAvatar(
  canvas: HTMLCanvasElement,
  frame: SignFrame,
  face: SignFaceFrame | null,
  bounds: NonNullable<ReturnType<typeof playbackBounds>>,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = LOGICAL_WIDTH, h = LOGICAL_HEIGHT;
  ctx.setTransform(RENDER_SCALE, 0, 0, RENDER_SCALE, 0, 0);
  const { minX, maxX, minY, maxY } = bounds;
  const scale = Math.min(w / Math.max(maxX - minX, .001), h / Math.max(maxY - minY, .001)) * .82;
  const ox = w / 2 - ((minX + maxX) / 2) * scale;
  const oy = h / 2 - ((minY + maxY) / 2) * scale + 12;
  const point = (n: number): XY => [frame[n][0] * scale + ox, frame[n][1] * scale + oy];
  const ls = point(11), rs = point(12), nose = point(0);
  const shoulders = Math.max(30, Math.hypot(ls[0] - rs[0], ls[1] - rs[1]));
  const headRadius = Math.max(28, shoulders * .25);
  const head: XY = [nose[0], nose[1] + headRadius * .16];

  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#f7fbff");
  bg.addColorStop(1, "#e9f3f5");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const top = (ls[1] + rs[1]) / 2 - 4;
  const bottom = Math.min(h + 12, top + shoulders * 1.25);
  ctx.fillStyle = "#165a72";
  ctx.strokeStyle = "#073d52";
  ctx.lineWidth = 3;
  line(ctx, [
    [ls[0] - shoulders * .12, top], [rs[0] + shoulders * .12, top],
    [rs[0] + shoulders * .30, bottom], [ls[0] - shoulders * .30, bottom],
  ], true);
  ctx.fill();
  ctx.stroke();

  const drawArm = (a: number, b: number, c: number) => {
    const points = [point(a), point(b), point(c)];
    ctx.strokeStyle = "#59351f"; ctx.lineWidth = 18; line(ctx, points); ctx.stroke();
    ctx.strokeStyle = "#d99a72"; ctx.lineWidth = 12; line(ctx, points); ctx.stroke();
  };
  drawArm(11, 13, 15);
  drawArm(12, 14, 16);

  ctx.fillStyle = "#d99a72";
  ctx.strokeStyle = "#59351f";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(head[0], head[1], headRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = "#2b201b";
  ctx.lineWidth = 2.6;
  if (face?.length === 48) {
    const facePoint = (n: number): XY => [face[n][0] * scale + ox, face[n][1] * scale + oy];
    for (const [start, length, close] of [[0, 10, false], [10, 10, false], [20, 6, true],
                                          [26, 6, true], [32, 8, true], [40, 8, true]] as const) {
      line(ctx, Array.from({length}, (_, i) => facePoint(start + i)), close);
      ctx.stroke();
    }
  } else {
    for (const eye of [point(2), point(5)]) {
      ctx.beginPath(); ctx.moveTo(eye[0] - 5, eye[1]); ctx.lineTo(eye[0] + 5, eye[1]); ctx.stroke();
    }
    line(ctx, [point(9), point(10)]);
    ctx.stroke();
  }

  const depth = (base: number) => {
    const values = frame.slice(base, base + 21).map(p => p[2]).filter(Number.isFinite);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  };
  const handsByDepth = [LEFT_BASE, RIGHT_BASE]
    .filter(base => trackedHand(frame, base))
    .sort((a, b) => depth(b) - depth(a));

  const drawHand = (base: number) => {
    const p = (n: number) => point(base + n);
    // A local light outline separates hands from clothing, skin and the face
    // while retaining the original landmark positions and hand proportions.
    ctx.strokeStyle = "rgba(247, 253, 255, .96)";
    ctx.lineWidth = 7;
    line(ctx, PALM.map(p), true);
    ctx.stroke();
    for (const [a, b] of HAND_BONES) {
      line(ctx, [p(a), p(b)]);
      ctx.stroke();
    }
    ctx.fillStyle = "#e7ad86";
    ctx.strokeStyle = "#342015";
    ctx.lineWidth = 2;
    line(ctx, PALM.map(p), true);
    ctx.fill();
    ctx.stroke();
    for (const [a, b] of HAND_BONES) {
      ctx.strokeStyle = "#342015"; ctx.lineWidth = 4.5; line(ctx, [p(a), p(b)]); ctx.stroke();
      ctx.strokeStyle = "#f0b58d"; ctx.lineWidth = 2.2; line(ctx, [p(a), p(b)]); ctx.stroke();
    }
    for (const joint of [4, 8, 12, 16, 20]) {
      const q = p(joint);
      ctx.fillStyle = "#fffaf6";
      ctx.strokeStyle = "#342015";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(q[0], q[1], 2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
  };
  for (const base of handsByDepth) drawHand(base);
}

/** A readable illustrated signer driven directly by exported landmarks. */
export default function SignPlayer({
  frames, faceFrames, fps = 14, onComplete, label = "Sign demonstration",
}: {
  frames: SignFrame[] | null;
  faceFrames?: (SignFaceFrame | null)[];
  fps?: number;
  onComplete?: () => void;
  label?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const completeRef = useRef(onComplete);
  const bounds = useMemo(() => playbackBounds(frames ?? []), [frames]);

  useEffect(() => { completeRef.current = onComplete; }, [onComplete]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frames?.length || !bounds) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const playbackFps = reduced ? Math.min(fps, 8) : fps;
    const start = performance.now();
    let lastFrame = -1;
    let request = 0;
    let complete = false;
    const render = (now: number) => {
      const frameIndex = Math.min(frames.length - 1,
        Math.floor((now - start) * Math.max(1, playbackFps) / 1000));
      if (frameIndex !== lastFrame) {
        drawAvatar(canvas, frames[frameIndex], faceFrames?.[frameIndex] ?? null, bounds);
        canvas.dataset.frame = String(frameIndex);
        lastFrame = frameIndex;
      }
      if (frameIndex < frames.length - 1) request = window.requestAnimationFrame(render);
      else if (!complete) { complete = true; completeRef.current?.(); }
    };
    drawAvatar(canvas, frames[0], faceFrames?.[0] ?? null, bounds);
    canvas.dataset.frame = "0";
    request = window.requestAnimationFrame(render);
    return () => window.cancelAnimationFrame(request);
  }, [frames, faceFrames, fps, bounds]);

  return <canvas ref={canvasRef} width={LOGICAL_WIDTH * RENDER_SCALE} height={LOGICAL_HEIGHT * RENDER_SCALE}
    className="signplayer sign-avatar" role="img" aria-label={label} data-frame="0"
    data-renderer="request-animation-frame" />;
}
