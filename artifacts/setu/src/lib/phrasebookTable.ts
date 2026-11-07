import { asset } from "./assetUrl";
import type { Phrase } from "./phrasebook";
import type { LangCode } from "./speech";

/**
 * Translations for the phrase board.
 *
 * Generated offline by train/translate_phrasebook.py (NLLB-200, local, free)
 * and committed, so the board works with no network, no API key and no account.
 * That is the point of this whole path: a hospital kiosk with a flaky
 * connection still speaks.
 */

interface Table {
  generated: string;
  model: string;
  entries: Record<string, Record<string, string>>;
}

let table: Table | null = null;
let failed = false;

export async function loadPhrasebook(
  url = asset("/model/_phrasebook.json")
): Promise<boolean> {
  if (table) return true;
  if (failed) return false;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(String(res.status));
    const body: unknown = await res.json();
    // A dev server with SPA fallback answers a missing file with index.html and
    // HTTP 200, so check the shape rather than the status, otherwise a missing
    // table becomes a parse error at the first tap instead of a clean degrade.
    if (
      typeof body !== "object" || body === null ||
      typeof (body as Table).entries !== "object" || (body as Table).entries === null
    ) {
      throw new Error("not a phrasebook");
    }
    table = body as Table;
    return true;
  } catch {
    failed = true;
    return false;
  }
}

/**
 * The text to speak for a phrase.
 *
 * Falls back to the English written in phrasebook.ts when the table is missing
 * or a language is absent. English spoken aloud is still useful in an Indian
 * hospital: silence is not, so the degrade is to a worse language, never to
 * nothing.
 */
export function phraseText(phrase: Phrase, lang: LangCode): string {
  return table?.entries[phrase.id]?.[lang] || phrase.en;
}

export function phrasebookSize(): number {
  return table ? Object.keys(table.entries).length : 0;
}
