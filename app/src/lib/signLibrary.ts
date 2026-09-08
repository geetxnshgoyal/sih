import { fetchJson } from './assets';
import type { SignLibrary } from './reverse';

export function validateSignLibrary(data: unknown): SignLibrary {
  if (!data || typeof data !== 'object' || Array.isArray(data) || !Object.keys(data).length) {
    throw new Error('The sign playback library is empty or invalid.');
  }
  for (const [gloss, frames] of Object.entries(data)) {
    if (!Array.isArray(frames) || !frames.length || frames.some(frame =>
      !Array.isArray(frame) || frame.length !== 65 || frame.some(point =>
        !Array.isArray(point) || (point.length !== 2 && point.length !== 3) || !point.every(Number.isFinite)))) {
      throw new Error(`Invalid playback frames for ${gloss}. Re-export the sign library.`);
    }
  }
  return data as SignLibrary;
}
let pending: Promise<SignLibrary> | null = null;
export function loadSignLibrary(): Promise<SignLibrary> {
  pending ??= fetchJson<unknown>('/model/_signs.json').then(validateSignLibrary).catch(error => {
    pending = null;
    throw error;
  });
  return pending;
}
