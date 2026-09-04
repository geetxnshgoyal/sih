/**
 * Gloss sequence -> spoken sentence, with the provenance attached.
 *
 * The honesty contract
 * --------------------
 * `sentence.ts` deliberately refused to reorder glosses, on the grounds that
 * inventing plausible word order produces confident mistranslation. That
 * reasoning still holds — what changes here is that a real reordering now
 * exists for the sequences we precomputed, so we can be accurate where we have
 * an answer and honest where we do not.
 *
 * Every result carries a `source` saying which of the three it is, and the UI
 * shows it. A precomputed sentence may be presented as a translation. A
 * gloss-order fallback may not.
 *
 * The table is generated offline by app/tools/buildGlossTable.ts — no API key
 * and no network call ever reaches the browser. See that file for why.
 */
import { phraseFor, type LangCode } from "./speech";
import { asset } from "./assetUrl";

export type TranslationSource =
  /** Precomputed reordering. Real translation — safe to present as one. */
  | "reordered"
  /** Single known sign from the bundled phrase table. */
  | "phrasebook"
  /** No entry: glosses read out in signed order. NOT a translation. */
  | "gloss-order";

export interface Translation {
  text: string;
  source: TranslationSource;
  /** True when the spoken order actually differs from the signed order. */
  reordered: boolean;
  register?: "statement" | "question" | "request";
}

interface Entry {
  glosses: string[];
  domain: string;
  reordered: boolean;
  register: "statement" | "question" | "request";
  [lang: string]: unknown;
}

interface Table {
  generated: string;
  model: string;
  entries: Record<string, Entry>;
}

let table: Table | null = null;
let loadFailed = false;

/**
 * Load the precomputed table. Safe to call repeatedly; safe to never call —
 * every lookup degrades to the phrasebook path if the table is missing, which
 * is exactly the pre-existing behaviour.
 */
// asset() is required here, not a plain absolute path. Under a subpath deploy
// (GitHub Pages serves from /<repo>/) "/model/..." resolves to the domain root
// and 404s — the table silently never loads and every phrase falls back to
// gloss order, which looks like the generator failed rather than the path.
export async function loadGlossTable(url = asset("/model/_utterances.json")): Promise<boolean> {
  if (table) return true;
  if (loadFailed) return false;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status}`);
    // A dev server with SPA fallback answers a MISSING file with index.html and
    // HTTP 200, so `res.ok` alone does not mean the table exists. Check the shape
    // rather than trusting the status, or a missing table silently becomes a
    // parse error at the first lookup instead of a clean degrade here.
    const body: unknown = await res.json();
    if (
      typeof body !== "object" ||
      body === null ||
      typeof (body as Table).entries !== "object" ||
      (body as Table).entries === null
    ) {
      throw new Error("not a gloss table");
    }
    table = body as Table;
    return true;
  } catch {
    // A missing table is a supported state, not an error: the app runs exactly
    // as it did before this feature existed.
    loadFailed = true;
    return false;
  }
}

export function tableSize(): number {
  return table ? Object.keys(table.entries).length : 0;
}

export function tableGeneratedAt(): string | null {
  return table?.generated ?? null;
}

function keyOf(glosses: string[]): string {
  return glosses.join("|");
}

/**
 * Translate a completed utterance.
 *
 * Lookup order:
 *   1. exact gloss sequence in the precomputed table  -> real reordering
 *   2. single gloss in the bundled phrase table       -> phrasebook
 *   3. anything else                                  -> signed order, labelled
 */
export function translateGlosses(glosses: string[], lang: LangCode): Translation {
  if (glosses.length === 0) {
    return { text: "", source: "gloss-order", reordered: false };
  }

  const entry = table?.entries[keyOf(glosses)];
  if (entry) {
    const text = entry[lang];
    if (typeof text === "string" && text.length > 0) {
      return {
        text,
        source: "reordered",
        reordered: entry.reordered,
        register: entry.register,
      };
    }
  }

  if (glosses.length === 1) {
    return { text: phraseFor(glosses[0], lang), source: "phrasebook", reordered: false };
  }

  return {
    text: glosses.map((g) => phraseFor(g, lang)).join(" "),
    source: "gloss-order",
    reordered: false,
  };
}

/** Short label for the UI, so the screen never overstates what happened. */
export function sourceLabel(source: TranslationSource): string {
  switch (source) {
    case "reordered":
      return "translated";
    case "phrasebook":
      return "phrase";
    case "gloss-order":
      return "gloss order — not translated";
  }
}
