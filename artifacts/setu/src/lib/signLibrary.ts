import { fetchJson } from './assets';
import { asset } from './assetUrl';
import type { SignClip, SignLibrary } from './reverse';

type SignIndex = {
  version: 1;
  count: number;
  glosses: Record<string, string>;
  shards: string[];
};

export function validateSignLibrary(data: unknown): SignLibrary {
  if (!data || typeof data !== 'object' || Array.isArray(data) || !Object.keys(data).length) {
    throw new Error('The sign playback library is empty or invalid.');
  }
  const normalized: SignLibrary = {};
  for (const [gloss, value] of Object.entries(data)) {
    const candidate = value as Partial<SignClip>;
    const frames = Array.isArray(value) ? value : candidate.body;
    if (!Array.isArray(frames) || !frames.length || frames.some(frame =>
      !Array.isArray(frame) || frame.length !== 65 || frame.some(point =>
        !Array.isArray(point) || (point.length !== 2 && point.length !== 3) || !point.every(Number.isFinite)))) {
      throw new Error(`Invalid playback frames for ${gloss}. Re-export the sign library.`);
    }
    if (!Array.isArray(value) && candidate.face && (
      !Array.isArray(candidate.face) || candidate.face.length !== frames.length ||
      candidate.face.some(face => face !== null && (
        !Array.isArray(face) || face.length !== 48 ||
        face.some(point => !Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite))
      ))
    )) throw new Error(`Invalid playback face frames for ${gloss}.`);
    normalized[gloss] = {
      version: 2,
      fps: !Array.isArray(value) && Number.isFinite(candidate.fps) ? candidate.fps! : 14,
      body: frames,
      ...(!Array.isArray(value) && candidate.face ? { face: candidate.face } : {}),
    };
  }
  return normalized;
}

function validateIndex(value: unknown): SignIndex {
  const index = value as Partial<SignIndex>;
  if (index?.version !== 1 || !index.glosses || typeof index.glosses !== 'object' ||
      !Array.isArray(index.shards) || Object.keys(index.glosses).length !== index.count) {
    throw new Error('The sharded sign index is invalid.');
  }
  const allowed = new Set(index.shards);
  if (Object.entries(index.glosses).some(([gloss, shard]) =>
    !gloss.trim() || typeof shard !== 'string' || !allowed.has(shard) || shard.includes('/') || shard.includes('..'))) {
    throw new Error('The sharded sign index contains an invalid entry.');
  }
  return index as SignIndex;
}

let indexPending: Promise<SignIndex | null> | null = null;
let legacyPending: Promise<SignLibrary> | null = null;
const shardPending = new Map<string, Promise<SignLibrary>>();
const clipCache = new Map<string, SignClip>();

async function loadIndex(): Promise<SignIndex | null> {
  indexPending ??= fetchJson<unknown>(asset('/signs/index.json')).then(validateIndex).catch(() => null);
  return indexPending;
}

async function loadLegacy(): Promise<SignLibrary> {
  legacyPending ??= fetchJson<unknown>(asset('/model/_signs.json')).then(validateSignLibrary).catch(error => {
    legacyPending = null;
    throw error;
  });
  return legacyPending;
}

/** Load names only. Frames remain in their shard until playback requests one. */
export async function loadSignCatalog(): Promise<string[]> {
  const index = await loadIndex();
  if (index) return Object.keys(index.glosses);
  return Object.keys(await loadLegacy());
}

export async function loadSignClip(gloss: string): Promise<SignClip | null> {
  if (clipCache.has(gloss)) return clipCache.get(gloss)!;
  const index = await loadIndex();
  if (!index) return (await loadLegacy())[gloss] ?? null;
  const shard = index.glosses[gloss];
  if (!shard) return null;
  let pending = shardPending.get(shard);
  if (!pending) {
    pending = fetchJson<unknown>(asset(`/signs/${shard}`)).then(validateSignLibrary);
    shardPending.set(shard, pending);
  }
  const library = await pending;
  for (const [name, clip] of Object.entries(library)) clipCache.set(name, clip);
  return library[gloss] ?? null;
}

/** Diagnostics may intentionally verify the complete playback export. */
export async function loadSignLibrary(): Promise<SignLibrary> {
  const names = await loadSignCatalog();
  const clips = await Promise.all(names.map(async name => [name, await loadSignClip(name)] as const));
  return Object.fromEntries(clips.filter((entry): entry is readonly [string, SignClip] => !!entry[1]));
}
