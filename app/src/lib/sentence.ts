import { phraseFor, type LangCode } from "./speech";
import { translateGlosses, type Translation } from "./glossTranslate";

/**
 * Assemble recognised signs into an utterance.
 *
 * ISL word order is not spoken word order, "STATION GO WHERE" rather than
 * "where do I go for the station": so a faithful system reorders glosses
 * before speaking. That reordering is a language task, and the honest options
 * are an LLM call or a linguist-written grammar.
 *
 * It is now done, but OFFLINE: `app/tools/buildGlossTable.ts` generates the
 * reorderings ahead of time into `public/model/_utterances.json`, which the app
 * reads with no network call and no API key in the browser. See
 * `glossTranslate.ts` for the lookup and the provenance contract.
 *
 * The original caution still governs everything outside that table. A sequence
 * with no precomputed entry is NOT reordered, it falls back to the phrasebook,
 * or to glosses in signed order, and reports which of the three happened so the
 * UI can label it. Inventing plausible word order at runtime would produce
 * confident mistranslation, which remains worse than none.
 */

/** A pause longer than this closes the utterance. */
export const UTTERANCE_GAP_MS = 2600;

export type Utterance = {
  glosses: string[];
  text: string;
  at: string;
  /**
   * Confidence of the LEAST certain sign in the utterance.
   *
   * Minimum, not mean: an utterance is only as trustworthy as its weakest
   * component. "I need PARACETAMOL" where the drug name was a 35% guess is a
   * 35% utterance, however sure the model was about "I" and "need". Averaging
   * would hide exactly the sign that matters most.
   */
  conf: number;
};

/**
 * Sign pairs that read naturally when run together. Kept small and explicit , 
 * this is a phrasebook, not a grammar, and it should not pretend otherwise.
 */
const PHRASE_PAIRS: Record<string, Record<LangCode, string>> = {
  "I|sick": {
    "hi-IN": "मैं बीमार हूँ", "ta-IN": "நான் நோய்வாய்ப்பட்டிருக்கிறேன்",
    "te-IN": "నేను అనారోగ్యంగా ఉన్నాను", "bn-IN": "আমি অসুস্থ",
    "mr-IN": "मी आजारी आहे", "en-IN": "I am sick",
  },
  "I|Doctor": {
    "hi-IN": "मुझे डॉक्टर चाहिए", "ta-IN": "எனக்கு மருத்துவர் வேண்டும்",
    "te-IN": "నాకు డాక్టర్ కావాలి", "bn-IN": "আমার ডাক্তার দরকার",
    "mr-IN": "मला डॉक्टर हवा आहे", "en-IN": "I need a doctor",
  },
  "Price|What": {
    "hi-IN": "कीमत क्या है", "ta-IN": "விலை என்ன", "te-IN": "ధర ఎంత",
    "bn-IN": "দাম কত", "mr-IN": "किंमत काय आहे", "en-IN": "What is the price",
  },
};

/**
 * Build the spoken form of a completed utterance, with its provenance.
 *
 * Order of preference:
 *   1. precomputed LLM reordering  (real translation)
 *   2. hand-written PHRASE_PAIRS   (phrasebook)
 *   3. glosses in signed order     (explicitly not a translation)
 */
export function assembleWithSource(glosses: string[], lang: LangCode): Translation {
  if (glosses.length === 0) return { text: "", source: "gloss-order", reordered: false };

  // 1. the precomputed table, which also covers the single-gloss case
  const translated = translateGlosses(glosses, lang);
  if (translated.source === "reordered") return translated;
  if (glosses.length === 1) return translated;

  // 2. the original pairwise phrasebook, kept because it is hand-verified
  const parts: string[] = [];
  let usedPair = false;
  for (let i = 0; i < glosses.length; i++) {
    const pair = `${glosses[i]}|${glosses[i + 1] ?? ""}`;
    const known = PHRASE_PAIRS[pair]?.[lang];
    if (known) {
      parts.push(known);
      usedPair = true;
      i += 1;
      continue;
    }
    parts.push(phraseFor(glosses[i], lang));
  }

  // 3. no pair matched either, this is signed order, and says so
  return {
    text: parts.join(" "),
    source: usedPair ? "phrasebook" : "gloss-order",
    reordered: false,
  };
}

/** Text-only form, for callers that do not display provenance. */
export function assemble(glosses: string[], lang: LangCode): string {
  return assembleWithSource(glosses, lang).text;
}

/**
 * Collects signs until a pause, then emits the utterance.
 * Time is passed in rather than read from the clock so it can be tested.
 */
export class UtteranceBuilder {
  private glosses: string[] = [];
  private confs: number[] = [];
  private lastAt = 0;

  get pending(): string[] { return [...this.glosses]; }

  /**
   * Returns a completed utterance if this sign started a new one.
   *
   * `conf` is required. It used to be dropped here, and the closing path then
   * emitted a hardcoded 1: so every finished utterance displayed 100% no
   * matter how uncertain its signs were, while the live readout showed the
   * honest number. That is the worst possible combination: the transient view
   * was truthful and the permanent record was not.
   */
  add(gloss: string, now: number, conf: number): Utterance | null {
    let finished: Utterance | null = null;
    if (this.glosses.length && now - this.lastAt > UTTERANCE_GAP_MS) {
      finished = this.flushAt(new Date(this.lastAt));
    }
    this.glosses.push(gloss);
    this.confs.push(conf);
    this.lastAt = now;
    return finished;
  }

  /** Close the current utterance if the pause has elapsed. */
  tick(now: number): Utterance | null {
    if (!this.glosses.length) return null;
    if (now - this.lastAt <= UTTERANCE_GAP_MS) return null;
    return this.flushAt(new Date(this.lastAt));
  }

  private flushAt(when: Date): Utterance {
    const glosses = this.glosses;
    const conf = this.confs.length ? Math.min(...this.confs) : 0;
    this.glosses = [];
    this.confs = [];
    return { glosses, text: "", at: when.toLocaleTimeString(), conf };
  }

  reset() {
    this.glosses = [];
    this.confs = [];
    this.lastAt = 0;
  }
}
