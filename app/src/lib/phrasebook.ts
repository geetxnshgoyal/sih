/**
 * The phrase board — the part that always works.
 *
 * Why this is not the gloss corpus
 * -------------------------------
 * Tap-to-say never runs the classifier. That frees it completely from the 264
 * INCLUDE labels, and the freedom matters more than it sounds: the eight things
 * a patient most needs to say — Water, Help, Yes, No, Pain, Name, Please,
 * Hotel — have no gloss in INCLUDE, so the recognition path can NEVER produce
 * them. Here they are just phrases, and they work.
 *
 * The design position
 * -------------------
 * Recognition is 40.4% correct on a signer it has not seen. A board is 100%.
 * So the board is the product and recognition is a shortcut on top of it, not
 * the other way round. A Deaf patient tonight is better served by sixty phrases
 * that always work than by 264 signs that are right two times in five.
 *
 * Ordering follows a real triage conversation, not the alphabet: identity first
 * (nothing else can happen until "I am Deaf" is established), then the presenting
 * complaint, then history, then needs. `urgent` entries surface first in the UI.
 *
 * Register
 * --------
 * These are spoken aloud BY the patient TO a clinician, often frightened and in
 * pain. Complete sentences, plain words, no telegraphic shorthand. "I am Deaf"
 * is capitalised as cultural identity, not a deficit.
 *
 * WHAT THIS FILE STILL NEEDS: review by Deaf ISL users, and by native speakers
 * of each output language. Two separate reviews, for two separate risks.
 *
 * The English here is a hearing engineer's guess at what a Deaf patient wants
 * to say. The translations are machine output from NLLB-200 and are NOT
 * verified — automated checks caught empties, untranslated passthrough, Latin
 * script leakage and self-duplication, but no automated check catches GRAMMAR.
 * One that slipped through: "I do not understand" rendered in Hindi as
 * "मैं समझ में नहीं आता", which is wrong (roughly "I am not understood").
 *
 * A wrong phrase spoken confidently to a clinician is the same failure the
 * calibration work removed from the recognition path. Until a native speaker
 * has read every line, treat this table as a draft that happens to be wired up.
 */

export type PhraseCategory =
  | "identity"
  | "emergency"
  | "pain"
  | "symptoms"
  | "history"
  | "needs"
  | "understanding"
  | "logistics"
  | "directions"
  | "transport"
  | "money"
  | "courtesy";

export interface Phrase {
  /** Stable key. Also the lookup key into the translation table. */
  id: string;
  /** English source. Translations are generated from exactly this string. */
  en: string;
  /** Short label for the button face; falls back to `en` when absent. */
  short?: string;
  category: PhraseCategory;
  /** Surfaces first, and is styled to be findable at a glance. */
  urgent?: boolean;
  /**
   * The gloss sequence this corresponds to, WHERE ONE EXISTS.
   * Optional on purpose — most useful phrases have no INCLUDE gloss, and
   * requiring one would cut exactly the vocabulary a patient needs.
   */
  glosses?: string[];
}

export const CATEGORY_LABEL: Record<PhraseCategory, string> = {
  identity: "About me",
  emergency: "Emergency",
  pain: "Pain",
  symptoms: "Symptoms",
  history: "History",
  needs: "I need",
  understanding: "Understanding",
  logistics: "Practical",
  directions: "Directions",
  transport: "Travel",
  money: "Money",
  courtesy: "Courtesy",
};

/* ------------------------------------------------------------------ health */

export const HEALTH_PHRASES: Phrase[] = [
  // Identity leads. Until this is established nothing else in the conversation
  // can proceed, and it is the phrase most likely to be needed by every user.
  { id: "deaf", en: "I am Deaf.", short: "I am Deaf", category: "identity", urgent: true, glosses: ["I", "Deaf"] },
  { id: "deaf-sign", en: "I am Deaf and I use Indian Sign Language.", short: "I am Deaf — I sign ISL", category: "identity", glosses: ["I", "Deaf", "Sign"] },
  { id: "need-interpreter", en: "I need a sign language interpreter.", short: "I need an interpreter", category: "identity", urgent: true },
  { id: "cannot-hear", en: "I cannot hear you. Please write it down.", short: "Please write it down", category: "identity" },
  { id: "face-me", en: "Please look at me when you speak so I can read your lips.", short: "Please face me", category: "identity" },
  { id: "write-please", en: "Please write your question and I will answer.", short: "Write your question", category: "identity" },
  { id: "patient-me", en: "I am the patient.", short: "I am the patient", category: "identity", glosses: ["Patient", "I"] },
  { id: "with-family", en: "I am here with my family.", short: "I am with family", category: "identity" },

  // Emergency. These must be reachable in one tap, never behind a category.
  { id: "emergency", en: "This is an emergency.", short: "Emergency", category: "emergency", urgent: true },
  { id: "help-now", en: "I need help right now.", short: "I need help now", category: "emergency", urgent: true },
  { id: "cannot-breathe", en: "I cannot breathe properly.", short: "Cannot breathe", category: "emergency", urgent: true },
  { id: "chest-pain", en: "I have pain in my chest.", short: "Chest pain", category: "emergency", urgent: true },
  { id: "bleeding", en: "I am bleeding.", short: "I am bleeding", category: "emergency", urgent: true },
  { id: "call-family", en: "Please call my family.", short: "Call my family", category: "emergency", urgent: true },
  { id: "call-ambulance", en: "Please call an ambulance.", short: "Call an ambulance", category: "emergency", urgent: true },
  { id: "unconscious", en: "This person is unconscious.", short: "Someone is unconscious", category: "emergency", urgent: true },

  // Pain — location, then character, then severity. This is the sequence a
  // clinician asks in, so the board should answer in that order.
  { id: "pain-have", en: "I am in pain.", short: "I am in pain", category: "pain", urgent: true },
  { id: "pain-here", en: "The pain is here.", short: "It hurts here", category: "pain" },
  { id: "pain-head", en: "My head hurts.", short: "Headache", category: "pain" },
  { id: "pain-stomach", en: "My stomach hurts.", short: "Stomach pain", category: "pain" },
  { id: "pain-back", en: "My back hurts.", short: "Back pain", category: "pain" },
  { id: "pain-throat", en: "My throat hurts.", short: "Sore throat", category: "pain" },
  { id: "pain-ear", en: "My ear hurts.", short: "Earache", category: "pain" },
  { id: "pain-tooth", en: "My tooth hurts.", short: "Toothache", category: "pain" },
  { id: "pain-sharp", en: "The pain is sharp.", short: "Sharp pain", category: "pain" },
  { id: "pain-dull", en: "The pain is dull and constant.", short: "Dull, constant", category: "pain" },
  { id: "pain-burning", en: "The pain feels like burning.", short: "Burning", category: "pain" },
  { id: "pain-mild", en: "The pain is mild.", short: "Mild", category: "pain" },
  { id: "pain-severe", en: "The pain is severe.", short: "Severe", category: "pain", urgent: true },
  { id: "pain-worse", en: "The pain is getting worse.", short: "Getting worse", category: "pain", urgent: true },
  { id: "pain-days", en: "I have had this pain for several days.", short: "For several days", category: "pain" },
  { id: "pain-today", en: "The pain started today.", short: "Started today", category: "pain" },

  // Symptoms
  { id: "sick", en: "I am sick.", short: "I am sick", category: "symptoms", glosses: ["I", "sick"] },
  { id: "fever", en: "I have a fever.", short: "Fever", category: "symptoms", glosses: ["I", "hot"] },
  { id: "cold-feel", en: "I feel cold and shivery.", short: "Chills", category: "symptoms", glosses: ["I", "cold"] },
  { id: "weak", en: "I feel very weak.", short: "Weak", category: "symptoms", glosses: ["I", "weak"] },
  { id: "dizzy", en: "I feel dizzy.", short: "Dizzy", category: "symptoms" },
  { id: "vomiting", en: "I have been vomiting.", short: "Vomiting", category: "symptoms" },
  { id: "nausea", en: "I feel like vomiting.", short: "Nauseous", category: "symptoms" },
  { id: "diarrhoea", en: "I have loose motions.", short: "Loose motions", category: "symptoms" },
  { id: "cough", en: "I have a cough.", short: "Cough", category: "symptoms" },
  { id: "cannot-sleep", en: "I cannot sleep.", short: "Cannot sleep", category: "symptoms" },
  { id: "no-appetite", en: "I have no appetite.", short: "No appetite", category: "symptoms" },
  { id: "swelling", en: "There is swelling here.", short: "Swelling", category: "symptoms" },
  { id: "rash", en: "I have a rash on my skin.", short: "Rash", category: "symptoms" },
  { id: "injury", en: "I was injured.", short: "I was injured", category: "symptoms", urgent: true },
  { id: "fell", en: "I fell down.", short: "I fell", category: "symptoms" },
  { id: "pregnant", en: "I am pregnant.", short: "I am pregnant", category: "symptoms", urgent: true },
  { id: "child-sick", en: "My child is sick.", short: "My child is sick", category: "symptoms", urgent: true, glosses: ["Child", "sick"] },
  { id: "baby-sick", en: "My baby is sick.", short: "My baby is sick", category: "symptoms", urgent: true, glosses: ["Baby", "sick"] },

  // History. Allergy and current medication are the two answers that change
  // treatment immediately, so they lead.
  { id: "allergy", en: "I have an allergy.", short: "I have an allergy", category: "history", urgent: true },
  { id: "allergy-medicine", en: "I am allergic to some medicines.", short: "Allergic to medicine", category: "history", urgent: true },
  { id: "taking-medicine", en: "I am already taking medicine.", short: "Taking medicine", category: "history" },
  { id: "show-medicine", en: "I can show you my medicine.", short: "I can show my medicine", category: "history" },
  { id: "diabetes", en: "I have diabetes.", short: "Diabetes", category: "history" },
  { id: "bp", en: "I have high blood pressure.", short: "High BP", category: "history" },
  { id: "asthma", en: "I have asthma.", short: "Asthma", category: "history" },
  { id: "heart", en: "I have a heart condition.", short: "Heart condition", category: "history" },
  { id: "surgery-before", en: "I have had surgery before.", short: "Past surgery", category: "history" },
  { id: "no-conditions", en: "I have no other medical conditions.", short: "No other conditions", category: "history" },
  { id: "have-report", en: "I have my medical reports with me.", short: "I have my reports", category: "history" },

  // Needs — the everyday requests that make a hospital stay bearable.
  { id: "water", en: "I need water.", short: "Water", category: "needs" },
  { id: "toilet", en: "I need to use the toilet.", short: "Toilet", category: "needs" },
  { id: "food", en: "I need something to eat.", short: "Food", category: "needs" },
  { id: "blanket", en: "I am cold. Please give me something to cover myself.", short: "Blanket", category: "needs" },
  { id: "sit", en: "I need to sit down.", short: "Need to sit", category: "needs" },
  { id: "lie-down", en: "I need to lie down.", short: "Need to lie down", category: "needs" },
  { id: "medicine-need", en: "I need my medicine.", short: "My medicine", category: "needs", glosses: ["I", "Medicine"] },
  { id: "doctor-need", en: "I need to see a doctor.", short: "See a doctor", category: "needs", urgent: true, glosses: ["I", "Doctor"] },
  { id: "nurse", en: "Please call the nurse.", short: "Call the nurse", category: "needs" },
  { id: "phone", en: "I need to use a phone.", short: "Use a phone", category: "needs" },

  // Understanding — the feedback channel. Without these the patient cannot say
  // the conversation has gone wrong, which is how consent quietly breaks down.
  { id: "yes", en: "Yes.", short: "Yes", category: "understanding" },
  { id: "no", en: "No.", short: "No", category: "understanding" },
  { id: "understand", en: "I understand.", short: "I understand", category: "understanding" },
  { id: "not-understand", en: "I do not understand.", short: "I don't understand", category: "understanding", urgent: true },
  { id: "repeat", en: "Please repeat that.", short: "Please repeat", category: "understanding" },
  { id: "slower", en: "Please speak more slowly.", short: "Slower please", category: "understanding" },
  { id: "wait", en: "Please wait a moment.", short: "Please wait", category: "understanding" },
  { id: "explain-again", en: "Please explain that again in simple words.", short: "Explain simply", category: "understanding" },
  { id: "not-sure", en: "I am not sure.", short: "Not sure", category: "understanding" },
  { id: "agree", en: "I agree to the treatment.", short: "I agree", category: "understanding" },
  { id: "not-agree", en: "I do not agree. Please explain more.", short: "I don't agree", category: "understanding", urgent: true },

  // Practical
  { id: "how-long", en: "How long will it take?", short: "How long?", category: "logistics" },
  { id: "cost", en: "How much will this cost?", short: "How much?", category: "logistics" },
  { id: "when-doctor", en: "When will the doctor come?", short: "When is the doctor?", category: "logistics", glosses: ["Doctor", "Time"] },
  { id: "when-medicine", en: "When should I take this medicine?", short: "When to take it?", category: "logistics", glosses: ["I", "Medicine", "Time"] },
  { id: "how-many-times", en: "How many times a day should I take it?", short: "How many times?", category: "logistics" },
  { id: "results-when", en: "When will my test results be ready?", short: "When are results?", category: "logistics" },
  { id: "go-home", en: "When can I go home?", short: "When can I go home?", category: "logistics" },
  { id: "come-back", en: "Do I need to come back?", short: "Come back again?", category: "logistics" },
  { id: "where-ward", en: "Where do I go now?", short: "Where do I go?", category: "logistics" },
  { id: "where-toilet", en: "Where is the toilet?", short: "Where is the toilet?", category: "logistics", glosses: ["Bathroom", "Location"] },
  { id: "where-pharmacy", en: "Where is the pharmacy?", short: "Where is the pharmacy?", category: "logistics" },

  { id: "thank-you", en: "Thank you.", short: "Thank you", category: "courtesy", glosses: ["Thank you"] },
  { id: "hello", en: "Hello.", short: "Hello", category: "courtesy", glosses: ["Hello"] },
  { id: "please", en: "Please.", short: "Please", category: "courtesy" },
  { id: "sorry", en: "Sorry.", short: "Sorry", category: "courtesy" },
];

/* ------------------------------------------------------------------ travel */

export const TRAVEL_PHRASES: Phrase[] = [
  { id: "t-deaf", en: "I am Deaf.", short: "I am Deaf", category: "identity", urgent: true, glosses: ["I", "Deaf"] },
  { id: "t-deaf-sign", en: "I am Deaf and I use Indian Sign Language.", short: "I am Deaf — I sign ISL", category: "identity", glosses: ["I", "Deaf", "Sign"] },
  { id: "t-write", en: "I cannot hear you. Please write it down.", short: "Please write it down", category: "identity" },
  { id: "t-point", en: "Please point or show me on the map.", short: "Please point / show me", category: "identity" },
  { id: "t-slower", en: "Please speak more slowly.", short: "Slower please", category: "identity" },

  { id: "t-help", en: "I need help.", short: "I need help", category: "emergency", urgent: true },
  { id: "t-police", en: "I need the police.", short: "Police", category: "emergency", urgent: true },
  { id: "t-lost", en: "I am lost.", short: "I am lost", category: "emergency", urgent: true },
  { id: "t-stolen", en: "My bag has been stolen.", short: "My bag was stolen", category: "emergency", urgent: true },
  { id: "t-hospital-need", en: "I need a hospital.", short: "Hospital", category: "emergency", urgent: true },
  { id: "t-lost-ticket", en: "I have lost my ticket.", short: "Lost my ticket", category: "emergency" },

  { id: "t-station", en: "Where is the railway station?", short: "Railway station", category: "directions", glosses: ["Train Station", "Location"] },
  { id: "t-bus-stop", en: "Where is the bus stop?", short: "Bus stop", category: "directions", glosses: ["Bus", "Location"] },
  { id: "t-airport", en: "Where is the airport?", short: "Airport", category: "directions" },
  { id: "t-toilet", en: "Where is the toilet?", short: "Toilet", category: "directions", glosses: ["Bathroom", "Location"] },
  { id: "t-hotel", en: "Where can I find a hotel?", short: "Hotel", category: "directions" },
  { id: "t-restaurant", en: "Where can I eat?", short: "Food", category: "directions", glosses: ["Restaurant", "Location"] },
  { id: "t-atm", en: "Where is an ATM?", short: "ATM", category: "directions", glosses: ["Bank", "Location"] },
  { id: "t-market", en: "Where is the market?", short: "Market", category: "directions", glosses: ["Market", "Location"] },
  { id: "t-temple", en: "Where is the temple?", short: "Temple", category: "directions", glosses: ["Temple", "Location"] },
  { id: "t-here-map", en: "Where am I on this map?", short: "Where am I?", category: "directions" },
  { id: "t-far", en: "Is it far from here?", short: "Is it far?", category: "directions" },
  { id: "t-walk", en: "Can I walk there?", short: "Can I walk?", category: "directions" },

  { id: "t-ticket", en: "I need a train ticket.", short: "Train ticket", category: "transport", glosses: ["I", "train ticket"] },
  { id: "t-ticket-price", en: "How much does a ticket cost?", short: "Ticket price", category: "transport", glosses: ["train ticket", "Price"] },
  { id: "t-train-time", en: "When does the train leave?", short: "Train time", category: "transport", glosses: ["Train", "Time"] },
  { id: "t-bus-time", en: "When does the bus leave?", short: "Bus time", category: "transport", glosses: ["Bus", "Time"] },
  { id: "t-platform", en: "Which platform is my train on?", short: "Which platform?", category: "transport" },
  { id: "t-this-train", en: "Is this the right train?", short: "Right train?", category: "transport" },
  { id: "t-taxi", en: "I need a taxi.", short: "Taxi", category: "transport", glosses: ["I", "Car"] },
  { id: "t-stop-here", en: "Please stop here.", short: "Stop here", category: "transport" },
  { id: "t-tell-me", en: "Please tell me when to get off.", short: "Tell me when to get off", category: "transport" },
  { id: "t-luggage", en: "Where do I leave my luggage?", short: "Luggage", category: "transport" },

  { id: "t-price", en: "How much does this cost?", short: "How much?", category: "money", glosses: ["Price"] },
  { id: "t-expensive", en: "That is too expensive.", short: "Too expensive", category: "money", glosses: ["expensive"] },
  { id: "t-card", en: "Can I pay by card?", short: "Pay by card?", category: "money" },
  { id: "t-change", en: "Please give me the change.", short: "My change", category: "money" },
  { id: "t-write-price", en: "Please write the price down.", short: "Write the price", category: "money" },

  { id: "t-yes", en: "Yes.", short: "Yes", category: "understanding" },
  { id: "t-no", en: "No.", short: "No", category: "understanding" },
  { id: "t-understand", en: "I understand.", short: "I understand", category: "understanding" },
  { id: "t-not-understand", en: "I do not understand.", short: "I don't understand", category: "understanding", urgent: true },
  { id: "t-repeat", en: "Please repeat that.", short: "Please repeat", category: "understanding" },
  { id: "t-wait", en: "Please wait a moment.", short: "Please wait", category: "understanding" },

  { id: "t-thanks", en: "Thank you.", short: "Thank you", category: "courtesy", glosses: ["Thank you"] },
  { id: "t-hello", en: "Hello.", short: "Hello", category: "courtesy", glosses: ["Hello"] },
  { id: "t-please", en: "Please.", short: "Please", category: "courtesy" },
  { id: "t-sorry", en: "Sorry.", short: "Sorry", category: "courtesy" },
];

/** Category order for the UI — triage order, not alphabetical. */
export const HEALTH_ORDER: PhraseCategory[] = [
  "emergency", "identity", "pain", "symptoms", "needs",
  "history", "understanding", "logistics", "courtesy",
];

export const TRAVEL_ORDER: PhraseCategory[] = [
  "emergency", "identity", "directions", "transport",
  "money", "understanding", "courtesy",
];
