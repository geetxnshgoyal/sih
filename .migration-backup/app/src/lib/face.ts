import type { Landmark } from './features';

// Mirrors train/face.py exactly. Keep this small subset instead of all 468
// landmarks so facial motion helps without overwhelming the hand signal.
export const FACE_SUBSET = [
  70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
  300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
  33, 160, 158, 133, 153, 144,
  362, 385, 387, 263, 373, 380,
  61, 37, 0, 267, 291, 314, 17, 84,
  78, 81, 13, 311, 308, 402, 14, 178,
] as const;

export type FaceFrame = Landmark[];

export function selectFace(face: Landmark[] | null): FaceFrame | null {
  if (!face || face.length < 468) return null;
  const selected = FACE_SUBSET.map(index => face[index]);
  return selected.every(point => point && [point.x, point.y, point.z].every(Number.isFinite))
    ? selected
    : null;
}
