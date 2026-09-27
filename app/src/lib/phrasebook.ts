/** Written service phrases. Selection is not sign recognition or translation. */
import { phrases } from "./phrases";
export type PhraseCategory = string;
export interface Phrase {
  id: string;
  en: string;
  short?: string;
  category: PhraseCategory;
  urgent?: boolean;
  glosses?: string[];
}
export const SERVICE_PHRASES: Phrase[] = phrases.map((phrase) => ({
  ...phrase,
  category: phrase.category as PhraseCategory,
}));
export const SERVICE_ORDER: PhraseCategory[] = [
  "Account charges",
  "Communication",
  "Everyday help",
];
export const CATEGORY_LABEL: Record<PhraseCategory, string> = {
  "Account charges": "Account charges",
  Communication: "Communication",
  "Everyday help": "Everyday help",
};
