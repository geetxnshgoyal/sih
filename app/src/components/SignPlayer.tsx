import { useEffect, useMemo, useRef, useState } from "react";
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
type XY = readonly [number, number];

function line(ctx: CanvasRenderingContext2D, points: XY[], close = false) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (const p of points.slice(1)) ctx.lineTo(...p);
  if (close) ctx.closePath();
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
  const [index, setIndex] = useState(0);
  const completeRef = useRef(onComplete);
  const bounds = useMemo(() => playbackBounds(frames ?? []), [frames]);

  useEffect(() => { completeRef.current = onComplete; }, [onComplete]);
  useEffect(() => {
    setIndex(0);
    if (!frames?.length) return;
    let frame = 0;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const playbackFps = reduced ? Math.min(fps, 8) : fps;
    const timer = window.setInterval(() => {
      frame += 1;
      if (frame >= frames.length) {
        window.clearInterval(timer);
        completeRef.current?.();
      } else setIndex(frame);
    }, 1000 / Math.max(1, playbackFps));
    return () => window.clearInterval(timer);
  }, [frames, fps]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const frame = frames?.[Math.min(index, (frames?.length ?? 1) - 1)];
    const face = faceFrames?.[Math.min(index, (faceFrames?.length ?? 1) - 1)] ?? null;
    if (!canvas || !ctx || !frame || !bounds) return;

    const w = canvas.width, h = canvas.height;
    const { minX, maxX, minY, maxY } = bounds;
    const scale = Math.min(w / Math.max(maxX - minX, .001), h / Math.max(maxY - minY, .001)) * .70;
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

    // Filled clothing gives the body a stable silhouette.
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

    // A neutral but readable face for legacy clips. Future face-aware clips can
    // extend the same player without changing the body frame contract.
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
    const hands = [LEFT_BASE, RIGHT_BASE]
      .filter(base => trackedHand(frame, base))
      .sort((a, b) => depth(b) - depth(a));

    const drawHand = (base: number, map: (n: number) => XY = point, width = 1) => {
      const p = (n: number) => map(base + n);
      ctx.fillStyle = "#e7ad86";
      ctx.strokeStyle = "#432719";
      ctx.lineWidth = 2.5 * width;
      line(ctx, PALM.map(p), true);
      ctx.fill();
      ctx.stroke();
      for (const [a, b] of HAND_BONES) {
        ctx.strokeStyle = "#432719"; ctx.lineWidth = 7 * width; line(ctx, [p(a), p(b)]); ctx.stroke();
        ctx.strokeStyle = "#e7ad86"; ctx.lineWidth = 4.2 * width; line(ctx, [p(a), p(b)]); ctx.stroke();
      }
      ctx.fillStyle = "#fff8f2";
      for (const tip of [4, 8, 12, 16, 20]) {
        const q = p(tip); ctx.beginPath(); ctx.arc(q[0], q[1], 2.1 * width, 0, Math.PI * 2); ctx.fill();
      }
    };
    for (const base of hands) drawHand(base);

    // Keep the real face-overlap in the avatar and duplicate only the hand in
    // a magnifier. This clarifies fingers without changing the sign geometry.
    const nearFace = hands.filter(base => frame.slice(base, base + 21).some((_, i) => {
      const p = point(base + i);
      return Math.hypot(p[0] - head[0], p[1] - head[1]) < headRadius * 1.18;
    }));
    nearFace.slice(0, 2).forEach((base, inset) => {
      const cx = inset ? 72 : w - 72, cy = 74, radius = 57;
      const raw = frame.slice(base, base + 21);
      const xs = raw.map(p => p[0]), ys = raw.map(p => p[1]);
      const minHX = Math.min(...xs), maxHX = Math.max(...xs);
      const minHY = Math.min(...ys), maxHY = Math.max(...ys);
      const detailScale = Math.min(radius * 1.4 / Math.max(maxHX - minHX, .001),
                                   radius * 1.4 / Math.max(maxHY - minHY, .001));
      const map = (n: number): XY => {
        const p = frame[n];
        return [(p[0] - (minHX + maxHX) / 2) * detailScale + cx,
                (p[1] - (minHY + maxHY) / 2) * detailScale + cy];
      };
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,.97)";
      ctx.strokeStyle = "#0f667a";
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.clip();
      drawHand(base, map, 1.25);
      ctx.restore();
    });
  }, [frames, faceFrames, index, bounds]);

  return <canvas ref={canvasRef} width={420} height={380} className="signplayer sign-avatar"
    role="img" aria-label={label} data-frame={index} />;
}
