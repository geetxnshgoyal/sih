/**
 * The gloss sequences we precompute translations for.
 *
 * Every entry uses ONLY glosses that exist in models/labels.json — INCLUDE is a
 * lexicon of 264 isolated signs, so an utterance can only be built from signs the
 * classifier can actually recognise. Notably absent, and worth remembering before
 * you add anything: there is no YES, NO, PLEASE, HELP, WHERE, WHAT, or NOT.
 *
 * The workarounds that vocabulary forces:
 *   "Location" carries WHERE   — "Bathroom Location" = "where is the bathroom"
 *   "Time"     carries WHEN    — "Doctor Time" = "when will the doctor come"
 *   "Price"    carries HOW MUCH
 *   "Mean"     carries WHAT DOES ... MEAN
 *
 * buildGlossTable.ts validates every gloss here against labels.json and refuses
 * to run if any is unknown, so this file cannot silently drift from the model.
 */

export interface GlossSeq {
  /** Signs in the order they are produced, i.e. ISL order, not spoken order. */
  glosses: string[];
  /** Where this utterance is expected — steers register and word choice. */
  domain: "medical" | "travel" | "civic" | "social";
  /** Free-text hint for anything the glosses alone cannot convey. */
  note?: string;
}

/**
 * Utterances a patient plainly needs and INCLUDE's 264 signs CANNOT express.
 *
 * Each of these was attempted and removed because the gloss does not exist in
 * `models/labels.json` — the classifier can never emit it, so no amount of
 * translation work makes the utterance reachable. This is the concrete,
 * checkable form of HANDOFF.md's "INCLUDE is a lexicon, not a conversation
 * vocabulary", and it is the argument for recording our own signs.
 *
 * Closing these is a data problem, not a model problem.
 */
export const UNREACHABLE: Array<{ want: string; missing: string }> = [
  { want: "I need water", missing: "Water" },
  { want: "I need help", missing: "Help" },
  { want: "yes / no", missing: "Yes, No" },
  { want: "where is X (as a question word)", missing: "Where — 'Location' is the stand-in" },
  { want: "my name is X", missing: "Name" },
  { want: "I need a hotel", missing: "Hotel" },
  { want: "it hurts here", missing: "Pain, Hurt" },
  { want: "please", missing: "Please" },
];

export const GLOSS_CORPUS: GlossSeq[] = [
  // ---------------------------------------------------------------- medical
  { glosses: ["I", "sick"], domain: "medical" },
  { glosses: ["I", "Doctor"], domain: "medical", note: "a request, not a statement of fact" },
  { glosses: ["I", "Medicine"], domain: "medical", note: "requesting medicine" },
  { glosses: ["I", "Medicine", "Time"], domain: "medical", note: "asking when to take the medicine" },
  { glosses: ["Doctor", "Location"], domain: "medical" },
  { glosses: ["Hospital", "Location"], domain: "medical" },
  { glosses: ["Doctor", "Time"], domain: "medical", note: "when will the doctor arrive" },
  { glosses: ["Medicine", "Price"], domain: "medical" },
  { glosses: ["I", "Doctor", "Today"], domain: "medical", note: "wants to see a doctor today" },
  { glosses: ["I", "Doctor", "Tomorrow"], domain: "medical" },
  { glosses: ["Patient", "I"], domain: "medical", note: "identifying oneself as the patient" },
  { glosses: ["I", "Deaf"], domain: "medical", note: "core self-identification; must be dignified" },
  { glosses: ["I", "Deaf", "Sign"], domain: "medical", note: "I am deaf, I use sign language" },
  { glosses: ["I", "sick", "Today"], domain: "medical" },
  { glosses: ["I", "hot"], domain: "medical", note: "fever, not ambient temperature" },
  { glosses: ["I", "cold"], domain: "medical", note: "feeling cold / chills" },
  { glosses: ["I", "weak"], domain: "medical" },
  { glosses: ["I", "healthy"], domain: "medical" },
  { glosses: ["Baby", "sick"], domain: "medical" },
  { glosses: ["Child", "sick"], domain: "medical" },
  { glosses: ["Mother", "sick"], domain: "medical" },
  { glosses: ["Father", "sick"], domain: "medical" },
  { glosses: ["Medicine", "Time"], domain: "medical", note: "at what time is the medicine due" },
  { glosses: ["I", "Bathroom"], domain: "medical", note: "needs to use the bathroom" },
  { glosses: ["Bathroom", "Location"], domain: "medical" },
  { glosses: ["Doctor", "good"], domain: "medical" },
  { glosses: ["I", "Hospital", "Today"], domain: "medical" },

  // ----------------------------------------------------------------- travel
  { glosses: ["Train Station", "Location"], domain: "travel" },
  { glosses: ["I", "train ticket"], domain: "travel" },
  { glosses: ["train ticket", "Price"], domain: "travel" },
  { glosses: ["Train", "Time"], domain: "travel", note: "when does the train leave" },
  { glosses: ["Bus", "Location"], domain: "travel" },
  { glosses: ["Bus", "Time"], domain: "travel" },
  { glosses: ["I", "Bus"], domain: "travel", note: "wants to take the bus" },
  { glosses: ["Restaurant", "Location"], domain: "travel" },
  { glosses: ["Market", "Location"], domain: "travel" },
  { glosses: ["Bank", "Location"], domain: "travel" },
  { glosses: ["Street or Road", "Location"], domain: "travel" },
  { glosses: ["I", "Money"], domain: "travel", note: "needs money / needs to withdraw" },
  { glosses: ["Price"], domain: "travel", note: "how much does this cost" },
  { glosses: ["cheap"], domain: "travel" },
  { glosses: ["expensive"], domain: "travel" },
  { glosses: ["Train", "Time", "Tomorrow"], domain: "travel" },
  { glosses: ["I", "Car"], domain: "travel", note: "needs a car / taxi" },
  { glosses: ["Plane", "Time"], domain: "travel" },
  { glosses: ["Temple", "Location"], domain: "travel" },
  { glosses: ["Park", "Location"], domain: "travel" },
  { glosses: ["Library", "Location"], domain: "travel" },

  // ------------------------------------------------------------------ civic
  { glosses: ["Police", "Location"], domain: "civic" },
  { glosses: ["I", "Police"], domain: "civic", note: "needs the police" },
  { glosses: ["School", "Location"], domain: "civic" },
  { glosses: ["Office", "Location"], domain: "civic" },
  { glosses: ["I", "Student"], domain: "civic" },
  { glosses: ["I", "Teacher"], domain: "civic" },
  { glosses: ["I", "Job"], domain: "civic", note: "looking for work" },
  { glosses: ["Bill", "Price"], domain: "civic" },

  // ----------------------------------------------------------------- social
  { glosses: ["Hello"], domain: "social" },
  { glosses: ["Thank you"], domain: "social" },
  { glosses: ["Good Morning"], domain: "social" },
  { glosses: ["Good afternoon"], domain: "social" },
  { glosses: ["Good evening"], domain: "social" },
  { glosses: ["Good night"], domain: "social" },
  { glosses: ["How are you"], domain: "social" },
  { glosses: ["I", "happy"], domain: "social" },
  { glosses: ["I", "sad"], domain: "social" },
  { glosses: ["Pleased"], domain: "social" },
  { glosses: ["I", "Friend"], domain: "social" },
  { glosses: ["I", "Family"], domain: "social" },
  { glosses: ["Mother", "Father"], domain: "social", note: "my parents" },
  { glosses: ["you", "good"], domain: "social" },
  { glosses: ["Alright"], domain: "social" },
  { glosses: ["I", "Sign"], domain: "social", note: "I sign / I use sign language" },
  { glosses: ["I", "Deaf", "you", "Sign"], domain: "social", note: "I am deaf — can you sign?" },
];
