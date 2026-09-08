import type { SignFrame } from './reverse';
/** Missing hands become a repeated nonzero point AFTER shoulder anchoring. */
export function trackedHand(frame: SignFrame, base: number): boolean {
  const wrist = frame[base];
  return frame.slice(base + 1, base + 21).some(p => Math.hypot(p[0]-wrist[0], p[1]-wrist[1]) > 0.00001);
}
export function playbackBounds(frames: SignFrame[]) {
  const points = frames.flatMap(frame => [
    ...[0,11,12,13,14,15,16].map(i => frame[i]),
    ...(trackedHand(frame,23) ? frame.slice(23,44) : []),
    ...(trackedHand(frame,44) ? frame.slice(44,65) : []),
  ]);
  if (!points.length) return null;
  return {
    minX: Math.min(...points.map(p=>p[0])), maxX: Math.max(...points.map(p=>p[0])),
    minY: Math.min(...points.map(p=>p[1])), maxY: Math.max(...points.map(p=>p[1])),
  };
}
