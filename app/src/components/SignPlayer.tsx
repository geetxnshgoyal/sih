import { useEffect, useMemo, useRef, useState } from "react";
import type { SignFrame } from "../lib/reverse";
import { playbackBounds, trackedHand } from "../lib/signGeometry";

const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const BODY_BONES = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12], [0, 11], [0, 12]];
const LEFT_BASE = 23;
const RIGHT_BASE = 44;

/**
 * Draws a sign as an animated skeleton.
 *
 * The frame index lives in state and an interval advances it, rather than a
 * requestAnimationFrame closure. The RAF version drew nothing in practice —
 * the callback's captured canvas context went stale across the re-renders that
 * the parent triggers while stepping through a sentence. Driving from state
 * means every frame change is an ordinary render and the draw always runs
 * against the live canvas.
 *
 * Frames are shoulder-anchored and shoulder-scaled, so signs from different
 * signers arrive in the same coordinate space and play back consistently.
 */
export default function SignPlayer({
  frames,
  fps = 14,
  onComplete,
  label = "Sign demonstration",
}: {
  frames: SignFrame[] | null;
  fps?: number;
  onComplete?: () => void;
  label?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [i, setI] = useState(0);

  const completeRef = useRef(onComplete);
  useEffect(() => { completeRef.current = onComplete; }, [onComplete]);
  // One viewport for the entire clip preserves motion and hand/body proportions.
  const bounds = useMemo(() => playbackBounds(frames ?? []), [frames]);
  useEffect(() => {
    setI(0);
    if (!frames?.length) return;
    let frame = 0;
    const id = window.setInterval(() => {
      frame++;
      if (frame >= frames.length) {
        window.clearInterval(id);
        completeRef.current?.();
      } else setI(frame);
    }, 1000 / Math.max(1, fps));
    return () => window.clearInterval(id);
  }, [frames, fps]);

  useEffect(() => {
    const cv = canvasRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);

    const frame = frames?.[Math.min(i, (frames?.length ?? 1) - 1)];
    if (!frame) return;

    if (!bounds) return;
    const { minX, maxX, minY, maxY } = bounds;
    const sc = Math.min(
      w / Math.max(maxX - minX, 1e-3),
      h / Math.max(maxY - minY, 1e-3)
    ) * 0.82;
    const ox = w / 2 - ((minX + maxX) / 2) * sc;
    const oy = h / 2 - ((minY + maxY) / 2) * sc;
    const P = (n: number) => [frame[n][0] * sc + ox, frame[n][1] * sc + oy] as const;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.strokeStyle = "rgba(15,95,117,.65)";
    ctx.lineWidth = 6;
    for (const [a, b] of BODY_BONES) {
      const [ax, ay] = P(a), [bx, by] = P(b);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }

    for (const base of [LEFT_BASE, RIGHT_BASE]) {
      // an undetected hand is stored as zeros — do not draw a collapsed claw
      const present = trackedHand(frame, base);
      if (!present) continue;
      ctx.strokeStyle = "rgba(15,95,117,.95)";
      ctx.lineWidth = 3.5;
      for (const [a, b] of HAND_BONES) {
        const [ax, ay] = P(base + a), [bx, by] = P(base + b);
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      }
      ctx.fillStyle = "#0f766e";
      for (let n = 0; n < 21; n++) {
        const [px, py] = P(base + n);
        ctx.beginPath(); ctx.arc(px, py, 3.2, 0, Math.PI * 2); ctx.fill();
      }
    }
  }, [frames, i, bounds]);

  return <canvas ref={canvasRef} width={420} height={380} className="signplayer" role="img" aria-label={label} data-frame={i} />;
}
