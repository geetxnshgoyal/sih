import { phrases } from "./phrases";
import type { Phrase } from "./phrasebook";
import type { LangCode } from "./speech";
/** Written English/Hindi service phrases ship with the app. */
export async function loadPhrasebook(): Promise<boolean> {
  return true;
}
export function phraseText(phrase: Phrase, lang: LangCode): string {
  return lang === "hi-IN"
    ? (phrases.find((item) => item.id === phrase.id)?.hi ?? phrase.en)
    : phrase.en;
}
export function phrasebookSize(): number {
  return phrases.length;
}
