import { PHRASES } from "./speech";
/**
 * Direction B: spoken language back into ISL.
 *
 * The hearing person speaks; the deaf person needs to see it signed. We have
 * 96 signs available as pose sequences, played back as an animated skeleton.
 *
 * Why skeletons rather than video: the pose release is 662 MB and already on
 * disk, where the raw video is 56.8 GB. A skeleton also sidesteps the awkward
 * cut between clips of different signers. Real video clips can replace this
 * later without touching the matching logic.
 */

// 65 points x [x, y], shoulder-anchored and shoulder-scaled. Depth is not
// stored: SignPlayer reads p[0] and p[1] only, so the z column was a third of
// a 3 MB file doing nothing. See train/export_signs.py.
export type SignFrame = number[][];
export type SignLibrary = Record<string, SignFrame[]>;

/**
 * Spoken words that should map onto a gloss we can actually play.
 *
 * This map is the setting-neutral core. Words specific to a deployment, "ward"
 * and "prescription", or "platform" and "fare": live on the domain in
 * `domains.ts` and are layered on top at call time, so the same recogniser
 * serves a hospital desk and a station enquiry counter without either one
 * carrying the other's vocabulary.
 */
const SYNONYMS: Record<string, string> = {
  hi: "Hello", hey: "Hello", namaste: "Hello", नमस्ते: "Hello",
  thanks: "Thank you", shukriya: "Thank you", धन्यवाद: "Thank you",
  ok: "Alright", okay: "Alright", fine: "Alright", ठीक: "Alright",
  physician: "Doctor", डॉक्टर: "Doctor", clinic: "Hospital", अस्पताल: "Hospital",
  medicine: "Medicine", dawai: "Medicine", दवा: "Medicine",
  cost: "Price", rate: "Price", कीमत: "Price", पैसा: "Money", paisa: "Money",
  rupees: "Money", cash: "Money", ticket: "train ticket", टिकट: "train ticket",
  station: "Train Station", स्टेशन: "Train Station",
  shop: "Store or Shop", store: "Store or Shop", दुकान: "Store or Shop",
  road: "Street or Road", street: "Street or Road", सड़क: "Street or Road",
  toilet: "Bathroom", washroom: "Bathroom", restroom: "Bathroom",
  phone: "Telephone", mobile: "Cell phone",
  home: "House", घर: "House",
  police: "Police", पुलिस: "Police",
  cheap: "cheap", costly: "expensive", big: "big large", small: "small little",
  hot: "hot", cold: "cold", quick: "fast", slow: "slow",
  ill: "sick", unwell: "sick", bimar: "sick",
   today: "Today", tomorrow: "Tomorrow",
  yesterday: "Yesterday", morning: "Morning", night: "Night",
  me: "I", my: "I", mine: "I", your: "you", us: "we", them: "they",
};

/**
 * Turn a spoken sentence into a playable sign sequence.
 *
 * This is word matching, not translation. ISL has its own grammar and word
 * order, so a faithful system would reorder glosses before playback, that is
 * the LLM step in the full design. Here we keep spoken order and are explicit
 * about it, rather than pretending a lookup is translation.
 */
export function textToGlosses(
  text: string,
  library: SignLibrary,
  /** Setting-specific words, layered over the shared map. See domains.ts. */
  extraSynonyms: Record<string, string> = {}
): {
  matched: string[];
  skipped: string[];
} {
  const normalize = (value: string) => value.normalize('NFC').toLowerCase()
    .replace(/[^\p{L}\p{N}\p{M}\s]/gu, ' ').trim().replace(/\s+/g, ' ');
  const known = new Map<string, string>();
  for (const g of Object.keys(library)) known.set(normalize(g), g);
  // Match the longest complete phrase, including aliases containing spaces.
  // Resolve alias targets without case sensitivity; never substitute an absent clip.
  const lookup = new Map(known);
  for (const [gloss, translations] of Object.entries(PHRASES)) {
    const target = known.get(normalize(gloss));
    if (target) for (const phrase of Object.values(translations)) if (phrase) lookup.set(normalize(phrase), target);
  }
  for (const [alias, target] of Object.entries({ ...SYNONYMS, ...extraSynonyms })) {
    const gloss = known.get(normalize(target));
    if (gloss && !known.has(normalize(alias))) lookup.set(normalize(alias), gloss);
  }
  const words = normalize(text).split(' ').filter(Boolean);
  const maxWords = Math.max(1, ...Array.from(lookup.keys(), key => key.split(' ').length));
  const matched: string[] = [];
  const skipped: string[] = [];
  for (let i = 0; i < words.length;) {
    let size = Math.min(maxWords, words.length - i);
    for (; size > 0; size--) {
      const gloss = lookup.get(words.slice(i, i + size).join(' '));
      if (gloss) { matched.push(gloss); break; }
    }
    if (size) i += size;
    else skipped.push(words[i++]);
  }
  return { matched, skipped };
}

/** Speech recognition, where the browser supports it. */
export function createRecogniser(lang: string): SpeechRecognition | null {
  const w = window as unknown as {
    SpeechRecognition?: { new (): SpeechRecognition };
    webkitSpeechRecognition?: { new (): SpeechRecognition };
  };
  const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
  if (!Ctor) return null;
  const r = new Ctor();
  r.lang = lang;
  r.continuous = false;
  r.interimResults = true;
  r.maxAlternatives = 1;
  return r;
}
