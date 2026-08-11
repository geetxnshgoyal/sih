/**
 * Direction B — spoken language back into ISL.
 *
 * The hearing person speaks; the deaf person needs to see it signed. We have
 * 96 signs available as pose sequences, played back as an animated skeleton.
 *
 * Why skeletons rather than video: the pose release is 662 MB and already on
 * disk, where the raw video is 56.8 GB. A skeleton also sidesteps the awkward
 * cut between clips of different signers. Real video clips can replace this
 * later without touching the matching logic.
 */

export type SignFrame = number[][];        // 65 points x 3
export type SignLibrary = Record<string, SignFrame[]>;

/**
 * Spoken words that should map onto a gloss we can actually play.
 *
 * This map is the setting-neutral core. Words specific to a deployment — "ward"
 * and "prescription", or "platform" and "fare" — live on the domain in
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
  home: "House", घर: "House", food: "Restaurant", khana: "Restaurant",
  police: "Police", पुलिस: "Police", help: "Police",
  cheap: "cheap", costly: "expensive", big: "big large", small: "small little",
  hot: "hot", cold: "cold", quick: "fast", slow: "slow",
  ill: "sick", unwell: "sick", bimar: "sick",
  now: "Time", when: "Time", today: "Today", tomorrow: "Tomorrow",
  yesterday: "Yesterday", morning: "Morning", night: "Night",
  me: "I", my: "I", mine: "I", your: "you", us: "we", them: "they",
};

/**
 * Turn a spoken sentence into a playable sign sequence.
 *
 * This is word matching, not translation. ISL has its own grammar and word
 * order, so a faithful system would reorder glosses before playback — that is
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
  // Domain entries win on collision: "station" means Train Station at an enquiry
  // desk even though the shared map has other ideas.
  const synonyms = { ...SYNONYMS, ...extraSynonyms };

  const known = new Map<string, string>();
  for (const g of Object.keys(library)) known.set(g.toLowerCase(), g);

  const words = text
    .toLowerCase()
    // \p{M} matters: every Indic vowel sign is a combining mark, so stripping
    // marks turns नमस्ते into "नमस त" and no Devanagari input ever matches.
    .replace(/[^\p{L}\p{N}\p{M}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);

  const matched: string[] = [];
  const skipped: string[] = [];

  for (let i = 0; i < words.length; i++) {
    // try a two-word phrase first ("thank you", "how are you" style)
    const two = `${words[i]} ${words[i + 1] ?? ""}`.trim();
    const three = `${two} ${words[i + 2] ?? ""}`.trim();

    if (known.has(three)) { matched.push(known.get(three)!); i += 2; continue; }
    if (known.has(two))   { matched.push(known.get(two)!);   i += 1; continue; }

    const w = words[i];
    if (known.has(w)) { matched.push(known.get(w)!); continue; }
    // The library guard matters: a synonym may point at a gloss the classifier
    // knows but which has no bundled pose frames (Location, Temple, Paper).
    // Matching it would queue a sign that renders as an empty player, so it is
    // treated as unmatched and reported in `skipped` instead.
    if (synonyms[w] && library[synonyms[w]]) { matched.push(synonyms[w]); continue; }
    skipped.push(w);
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
