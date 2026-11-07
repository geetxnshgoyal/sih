/**
 * The phrase board: the part that always works.
 *
 * Why this is not the gloss corpus
 * -------------------------------
 * Tap-to-say never runs the classifier. That frees it completely from the 38
 * trained labels, and the freedom matters more than it sounds: the things a
 * patient most needs to say, Water, Help, Yes, No, Pain, Name, Please,
 * Hotel, have one clip each at best, so they can never be a trained class.
 * The dictionary bank reaches them as a shortlist of guesses. Here they are
 * just phrases, and they work.
 *
 * The design position
 * -------------------
 * Recognition is about 74% correct on a signer it has not seen. A board is
 * 100%. So the board is the product and recognition is a shortcut on top of
 * it, not the other way round. A Deaf patient tonight is better served by 97
 * phrases that always work than by 38 signs that are right three times in
 * four and 203 more that are right about half the time.
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
 * verified: automated checks caught empties, untranslated passthrough, Latin
 * script leakage and self-duplication, but no automated check catches GRAMMAR.
 * One that slipped through: "I do not understand" rendered in Hindi as
 * "मैं समझ में नहीं आता", which is wrong (roughly "I am not understood").
 *
 * A wrong phrase spoken confidently to a clinician is the same failure the
 * calibration work removed from the recognition path. Until a native speaker
 * has read every line, treat this table as a draft that happens to be wired up.
 */

import type { DomainId } from "./domains";

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
   * Optional on purpose: most useful phrases have no INCLUDE gloss, and
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
  { id: "deaf-sign", en: "I am Deaf and I use Indian Sign Language.", short: "I am Deaf: I sign ISL", category: "identity" },
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

  // Pain: location, then character, then severity. This is the sequence a
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
  { id: "baby-sick", en: "My baby is sick.", short: "My baby is sick", category: "symptoms", urgent: true, glosses: ["baby", "sick"] },

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

  // Needs: the everyday requests that make a hospital stay bearable.
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

  // Understanding: the feedback channel. Without these the patient cannot say
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
  { id: "t-deaf-sign", en: "I am Deaf and I use Indian Sign Language.", short: "I am Deaf: I sign ISL", category: "identity" },
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

/** Category order for the UI, triage order, not alphabetical. */
export const HEALTH_ORDER: PhraseCategory[] = [
  "emergency", "identity", "pain", "symptoms", "needs",
  "history", "understanding", "logistics", "courtesy",
];

export const TRAVEL_ORDER: PhraseCategory[] = [
  "emergency", "identity", "directions", "transport",
  "money", "understanding", "courtesy",
];

/* ---------------------------------------------------------------- service */

/**
 * Hospitality, financial services, entertainment and retail.
 *
 * Written for SIH26199. The four counters share more than they differ: every
 * one of them needs the customer to establish that they are Deaf, to ask a
 * price, to say they do not understand, and to refuse politely. Those live in
 * SERVICE_COMMON and are spread into each board rather than written four
 * times, so a fix to the wording of "I do not understand" fixes it everywhere.
 *
 * Register: these are spoken aloud BY the customer TO a staff member who is
 * usually busy and has a queue behind. Short complete sentences. No hedging,
 * because a hedged sentence read aloud by a synthesiser at a till does not
 * land. "I am Deaf" is capitalised as cultural identity, not a deficit.
 *
 * THE FINANCIAL ONES ARE THE DANGEROUS ONES. "I do not understand this form"
 * and "Do not process this yet" exist because the failure mode at a bank is
 * not confusion, it is a customer signing something. Those are marked urgent
 * so they sort to the top and are reachable without scrolling or searching.
 *
 * Same caveat as the clinical board and it is not smaller here: the English is
 * a hearing engineer's guess, and the translations are unreviewed NLLB-200
 * output. A wrong sentence spoken confidently at a bank counter is worse than
 * silence. Until a Deaf ISL user and a native speaker of each language have
 * read every line, this is a draft that happens to be wired up.
 */
const SERVICE_COMMON: Phrase[] = [
  { id: "s-deaf", en: "I am Deaf.", short: "I am Deaf", category: "identity", urgent: true, glosses: ["I", "Deaf"] },
  { id: "s-deaf-isl", en: "I am Deaf and I use Indian Sign Language.", short: "I am Deaf: I use ISL", category: "identity" },
  { id: "s-write", en: "I cannot hear you. Please write it down.", short: "Please write it down", category: "identity" },
  { id: "s-slower", en: "Please speak more slowly.", short: "Slower please", category: "identity" },
  { id: "s-face", en: "Please look at me when you speak.", short: "Please face me", category: "identity" },
  { id: "s-phone-no", en: "I cannot use the telephone. Please contact me another way.", short: "I cannot use the phone", category: "identity" },

  { id: "s-understand-no", en: "I do not understand. Please explain again.", short: "I do not understand", category: "understanding", urgent: true },
  { id: "s-repeat", en: "Please say that again.", short: "Say again", category: "understanding" },
  { id: "s-show", en: "Please show me.", short: "Show me", category: "understanding" },
  { id: "s-number", en: "Please write the number down.", short: "Write the number", category: "understanding" },
  { id: "s-understand-yes", en: "Yes, I understand.", short: "I understand", category: "understanding" },
  { id: "s-wait", en: "Please wait a moment.", short: "Please wait", category: "understanding" },
  { id: "s-time-more", en: "Please give me more time.", short: "More time please", category: "understanding" },

  { id: "s-price", en: "How much does this cost?", short: "How much", category: "money" },
  { id: "s-expensive", en: "That is too expensive.", short: "Too expensive", category: "money" },
  { id: "s-card", en: "I will pay by card.", short: "Pay by card", category: "money" },
  { id: "s-cash", en: "I will pay in cash.", short: "Pay cash", category: "money" },
  { id: "s-upi", en: "Can I pay by UPI?", short: "UPI?", category: "money" },
  { id: "s-bill", en: "May I have the bill, please?", short: "The bill please", category: "money" },
  { id: "s-receipt", en: "May I have a receipt?", short: "Receipt please", category: "money" },
  { id: "s-change", en: "I think the change is wrong.", short: "Change is wrong", category: "money" },

  { id: "s-help", en: "I need help.", short: "I need help", category: "emergency", urgent: true },
  { id: "s-manager", en: "May I speak to the manager?", short: "The manager please", category: "emergency" },
  { id: "s-toilet", en: "Where is the toilet?", short: "Toilet", category: "logistics" },
  { id: "s-thanks", en: "Thank you for your help.", short: "Thank you", category: "courtesy" },
  { id: "s-sorry", en: "Sorry, one moment.", short: "One moment", category: "courtesy" },
  { id: "s-patience", en: "Thank you for being patient with me.", short: "Thank you for waiting", category: "courtesy" },
];

export const HOTEL_PHRASES: Phrase[] = [
  ...SERVICE_COMMON,

  { id: "h-checkin", en: "I have a booking. I would like to check in.", short: "Check in", category: "needs" },
  { id: "h-nobooking", en: "I do not have a booking. Do you have a room?", short: "Do you have a room", category: "needs" },
  { id: "h-room-cost", en: "How much is the room per night?", short: "Room rate", category: "money" },
  { id: "h-included", en: "What is included in the price?", short: "What is included", category: "money" },
  { id: "h-breakfast", en: "Is breakfast included?", short: "Breakfast included?", category: "money" },
  { id: "h-deposit", en: "Is there a deposit? How much?", short: "Deposit?", category: "money" },
  { id: "h-cancel", en: "What is the cancellation policy?", short: "Cancellation policy", category: "money" },

  { id: "h-nights", en: "I am staying for two nights.", short: "Two nights", category: "logistics" },
  { id: "h-checkout-time", en: "What time is checkout?", short: "Checkout time", category: "logistics" },
  { id: "h-late", en: "Can I check out later?", short: "Late checkout?", category: "logistics" },
  { id: "h-room-where", en: "Where is my room?", short: "Where is my room", category: "logistics" },
  { id: "h-lift", en: "Where is the lift?", short: "Where is the lift", category: "logistics" },
  { id: "h-luggage", en: "Can you keep my luggage?", short: "Keep my luggage", category: "logistics" },
  { id: "h-wifi", en: "What is the wifi password?", short: "Wifi password", category: "logistics" },
  { id: "h-food-time", en: "What time is food served?", short: "Meal times", category: "logistics" },

  { id: "h-key", en: "My key does not work.", short: "Key does not work", category: "needs" },
  { id: "h-ac", en: "The air conditioning is not working.", short: "AC not working", category: "needs" },
  { id: "h-hot-water", en: "There is no hot water.", short: "No hot water", category: "needs" },
  { id: "h-towels", en: "Could I have more towels?", short: "More towels", category: "needs" },
  { id: "h-clean", en: "Please clean the room.", short: "Clean the room", category: "needs" },
  { id: "h-quiet", en: "The room is too noisy. Could I change rooms?", short: "Change rooms", category: "needs" },

  /* The reason a Deaf guest may not survive a fire alarm. Worth a button. */
  { id: "h-alarm", en: "I am Deaf. I will not hear the fire alarm. Please tell me how I will be warned.", short: "I cannot hear the alarm", category: "emergency", urgent: true },
  { id: "h-knock", en: "I will not hear you knock. Please message me instead.", short: "I will not hear knocking", category: "emergency", urgent: true },
  { id: "h-doctor", en: "I need a doctor.", short: "I need a doctor", category: "emergency", urgent: true },
];

export const BANK_PHRASES: Phrase[] = [
  ...SERVICE_COMMON,

  /* Consent first. At a bank the failure is not confusion, it is signing. */
  { id: "b-form-no", en: "I do not understand this form. Please explain it before I sign.", short: "Explain before I sign", category: "understanding", urgent: true },
  { id: "b-stop", en: "Please do not process this yet. I have a question.", short: "Please wait, I have a question", category: "understanding", urgent: true },
  { id: "b-read", en: "I need time to read this properly.", short: "I need to read it", category: "understanding", urgent: true },
  { id: "b-charges", en: "What are the charges on this?", short: "What are the charges", category: "money", urgent: true },
  { id: "b-sign-where", en: "Where do I sign?", short: "Where do I sign", category: "understanding" },

  { id: "b-open", en: "I want to open an account.", short: "Open an account", category: "needs" },
  { id: "b-deposit", en: "I want to deposit money.", short: "Deposit", category: "needs" },
  { id: "b-withdraw", en: "I want to withdraw money.", short: "Withdraw", category: "needs" },
  { id: "b-balance", en: "What is my account balance?", short: "My balance", category: "needs" },
  { id: "b-statement", en: "I need a statement.", short: "Statement", category: "needs" },
  { id: "b-transfer", en: "I want to transfer money.", short: "Transfer money", category: "needs" },

  { id: "b-card-new", en: "I need a new debit card.", short: "New card", category: "needs" },
  { id: "b-card-lost", en: "My card is lost. Please block it.", short: "Block my card", category: "emergency", urgent: true },
  { id: "b-card-stuck", en: "The machine has kept my card.", short: "Machine took my card", category: "emergency", urgent: true },
  { id: "b-fraud", en: "There is a transaction I did not make.", short: "I did not make this transaction", category: "emergency", urgent: true },
  { id: "b-pin", en: "I have forgotten my PIN.", short: "Forgot my PIN", category: "needs" },

  { id: "b-docs", en: "Here are my documents.", short: "My documents", category: "logistics" },
  { id: "b-docs-what", en: "Which documents do you need?", short: "Which documents", category: "logistics" },
  { id: "b-kyc", en: "I am here to complete my KYC.", short: "KYC", category: "logistics" },
  { id: "b-how-long", en: "How long will this take?", short: "How long", category: "logistics" },
  { id: "b-when-ready", en: "When should I come back?", short: "When do I return", category: "logistics" },
  { id: "b-token", en: "Do I need a token?", short: "Token?", category: "logistics" },
  { id: "b-counter", en: "Which counter should I go to?", short: "Which counter", category: "logistics" },

  { id: "b-loan", en: "I want to ask about a loan.", short: "About a loan", category: "money" },
  { id: "b-interest", en: "What is the interest rate?", short: "Interest rate", category: "money" },
  { id: "b-emi", en: "How much is the monthly instalment?", short: "Monthly instalment", category: "money" },
];

export const CINEMA_PHRASES: Phrase[] = [
  ...SERVICE_COMMON,

  /* The question that decides whether a Deaf customer buys a ticket at all. */
  { id: "c-subtitles", en: "I am Deaf. Does this film have subtitles?", short: "Does it have subtitles", category: "needs", urgent: true },
  { id: "c-subtitles-which", en: "Which shows have subtitles?", short: "Which shows are subtitled", category: "needs", urgent: true },
  { id: "c-captions", en: "Do you have captioning devices?", short: "Captioning devices?", category: "needs", urgent: true },
  { id: "c-access-seat", en: "I need accessible seating.", short: "Accessible seating", category: "needs", urgent: true },

  { id: "c-ticket", en: "I would like a ticket.", short: "One ticket", category: "needs" },
  { id: "c-tickets-two", en: "I would like two tickets.", short: "Two tickets", category: "needs" },
  { id: "c-ticket-child", en: "One ticket for a child.", short: "Child ticket", category: "needs" },
  { id: "c-ticket-price", en: "How much is a ticket?", short: "Ticket price", category: "money" },
  { id: "c-concession", en: "Is there a concession for disabled customers?", short: "Disability concession?", category: "money" },

  { id: "c-time", en: "What time does it start?", short: "Start time", category: "logistics" },
  { id: "c-length", en: "How long is it?", short: "How long", category: "logistics" },
  { id: "c-tomorrow", en: "A ticket for tomorrow, please.", short: "Ticket for tomorrow", category: "logistics" },
  { id: "c-seat-where", en: "Where is my seat?", short: "Where is my seat", category: "logistics" },
  { id: "c-screen", en: "Which screen?", short: "Which screen", category: "logistics" },
  { id: "c-entrance", en: "Which entrance do I use?", short: "Which entrance", category: "logistics" },
  { id: "c-interval", en: "Is there an interval?", short: "Interval?", category: "logistics" },
  { id: "c-food", en: "Where can I buy food?", short: "Where is food", category: "logistics" },

  { id: "c-refund", en: "Can I get a refund?", short: "Refund?", category: "money" },
  { id: "c-exchange", en: "Can I change this to another show?", short: "Change the show", category: "money" },
  { id: "c-announce", en: "I will not hear announcements. Please tell me if anything changes.", short: "I cannot hear announcements", category: "emergency", urgent: true },
];

export const RETAIL_PHRASES: Phrase[] = [
  ...SERVICE_COMMON,

  { id: "r-looking", en: "I am just looking, thank you.", short: "Just looking", category: "courtesy" },
  { id: "r-find", en: "I am looking for something. Can you help?", short: "I am looking for", category: "needs" },
  { id: "r-where", en: "Where can I find this?", short: "Where is this", category: "needs" },
  { id: "r-stock", en: "Do you have this in stock?", short: "In stock?", category: "needs" },
  { id: "r-other-size", en: "Do you have a different size?", short: "Different size", category: "needs" },
  { id: "r-bigger", en: "Do you have a bigger one?", short: "Bigger", category: "needs" },
  { id: "r-smaller", en: "Do you have a smaller one?", short: "Smaller", category: "needs" },
  { id: "r-other-colour", en: "Do you have another colour?", short: "Another colour", category: "needs" },
  { id: "r-new-one", en: "Could I have a new one, not the display piece?", short: "A new one please", category: "needs" },
  { id: "r-try", en: "May I try this on?", short: "May I try it", category: "needs" },

  { id: "r-cheaper", en: "Do you have something cheaper?", short: "Something cheaper", category: "money" },
  { id: "r-discount", en: "Is there a discount on this?", short: "Any discount", category: "money" },
  { id: "r-price-check", en: "Please check the price for me.", short: "Check the price", category: "money" },
  { id: "r-price-wrong", en: "The price on the label is different.", short: "Label price is different", category: "money" },
  { id: "r-take", en: "I will take it.", short: "I will take it", category: "money" },
  { id: "r-not-take", en: "I will not take it, thank you.", short: "Not this one", category: "money" },

  { id: "r-return", en: "I would like to return this.", short: "Return this", category: "logistics" },
  { id: "r-exchange", en: "I would like to exchange this.", short: "Exchange this", category: "logistics" },
  { id: "r-return-policy", en: "What is the return policy?", short: "Return policy", category: "logistics" },
  { id: "r-warranty", en: "Is there a warranty?", short: "Warranty?", category: "logistics" },
  { id: "r-faulty", en: "This is faulty.", short: "This is faulty", category: "logistics", urgent: true },
  { id: "r-bag", en: "May I have a bag?", short: "A bag please", category: "logistics" },
  { id: "r-deliver", en: "Do you deliver?", short: "Do you deliver", category: "logistics" },
  { id: "r-close", en: "What time do you close?", short: "Closing time", category: "logistics" },
];

/** Category order per setting. Urgent-first, then the shape of the exchange. */
export const HOTEL_ORDER: PhraseCategory[] = [
  "emergency", "identity", "needs", "logistics", "money", "understanding", "courtesy",
];
export const BANK_ORDER: PhraseCategory[] = [
  "emergency", "understanding", "identity", "needs", "money", "logistics", "courtesy",
];
export const CINEMA_ORDER: PhraseCategory[] = [
  "emergency", "needs", "identity", "logistics", "money", "understanding", "courtesy",
];
export const RETAIL_ORDER: PhraseCategory[] = [
  "emergency", "identity", "needs", "money", "logistics", "understanding", "courtesy",
];

/** The board for a setting. One lookup, so PhraseBoard stops branching. */
export const PHRASES_FOR: Record<DomainId, Phrase[]> = {
  health: HEALTH_PHRASES,
  travel: TRAVEL_PHRASES,
  hotel: HOTEL_PHRASES,
  bank: BANK_PHRASES,
  cinema: CINEMA_PHRASES,
  retail: RETAIL_PHRASES,
};

export const ORDER_FOR: Record<DomainId, PhraseCategory[]> = {
  health: HEALTH_ORDER,
  travel: TRAVEL_ORDER,
  hotel: HOTEL_ORDER,
  bank: BANK_ORDER,
  cinema: CINEMA_ORDER,
  retail: RETAIL_ORDER,
};

/* ------------------------------------------------------------ staff side */

/**
 * What the person BEHIND the counter says.
 *
 * The client boards above are the Deaf person speaking outward. This is the
 * other direction: a doctor, teller, clerk or shop assistant picking a
 * sentence, which is then shown to the Deaf person as text and played back as
 * signs where a recording exists.
 *
 * It replaces a hardcoded seven-entry clinical list that shipped on the
 * Phrases nav item at every counter, so a shop assistant was offered "Take 1
 * tablet after meals". That list also carried invented gloss sequences: "Apply
 * ointment twice daily" was mapped to HELP THANK YOU, and none of its uppercase
 * glosses existed in _signs.json at all. It was demo scaffolding that never got
 * finished and never got removed.
 *
 * Glosses are OMITTED unless a real recording sequence genuinely says the
 * sentence. A sentence with no glosses shows as text and speaks; that is an
 * honest degrade. Inventing a gloss so the animation has something to play is
 * how the old list ended up signing "help thank you" at an ointment.
 */
const SHARED_STAFF: Phrase[] = [
  { id: "sf-wait", en: "Please wait one moment.", short: "Please wait", category: "courtesy", glosses: ["please", "wait"] },
  { id: "sf-understand-q", en: "Do you understand?", short: "Do you understand?", category: "understanding", glosses: ["you", "understand"] },
  { id: "sf-again", en: "I will explain again.", short: "I will explain again", category: "understanding" },
  { id: "sf-write", en: "I will write it down for you.", short: "I will write it down", category: "understanding" },
  { id: "sf-show", en: "Let me show you.", short: "Let me show you", category: "understanding" },
  { id: "sf-slow", en: "Take your time. There is no hurry.", short: "Take your time", category: "courtesy" },
  { id: "sf-id", en: "May I see your identity document?", short: "Your ID please", category: "logistics", glosses: ["Card", "please"] },
  { id: "sf-sorry-wait", en: "Sorry for the wait.", short: "Sorry for the wait", category: "courtesy", glosses: ["sorry"] },
  { id: "sf-help-q", en: "How can I help you?", short: "How can I help?", category: "courtesy", glosses: ["I", "help", "you"] },
  { id: "sf-interpreter", en: "Would you like us to arrange an interpreter?", short: "Arrange an interpreter?", category: "understanding" },
  { id: "sf-anything-else", en: "Is there anything else?", short: "Anything else?", category: "courtesy", glosses: ["more"] },
  { id: "sf-thanks", en: "Thank you. Have a good day.", short: "Thank you", category: "courtesy", glosses: ["Thank you"] },
];

export const HEALTH_STAFF: Phrase[] = [
  ...SHARED_STAFF,
  { id: "sf-h-where-hurt", en: "Where does it hurt? Please point to the place.", short: "Where does it hurt?", category: "pain", urgent: true, glosses: ["pain", "Location"] },
  { id: "sf-h-how-long", en: "How long have you had this?", short: "How long?", category: "symptoms", glosses: ["Time", "more"] },
  { id: "sf-h-allergy", en: "Are you allergic to any medicine?", short: "Any allergies?", category: "history", urgent: true, glosses: ["Medicine", "bad"] },
  { id: "sf-h-current-med", en: "Are you taking any medicine at the moment?", short: "Taking any medicine?", category: "history", glosses: ["you", "Medicine"] },
  { id: "sf-h-tablet", en: "Take one tablet after food, twice a day.", short: "One tablet after food", category: "needs", glosses: ["tablet", "eat"] },
  { id: "sf-h-ointment", en: "Apply the ointment twice a day.", short: "Apply twice a day", category: "needs" },
  { id: "sf-h-blood", en: "You need a blood test tomorrow morning, before eating.", short: "Blood test tomorrow", category: "needs", glosses: ["blood", "Tomorrow", "Morning"] },
  { id: "sf-h-breathe", en: "Take a deep breath and hold it.", short: "Deep breath", category: "symptoms", glosses: ["breathe"] },
  { id: "sf-h-lie", en: "Please lie down.", short: "Please lie down", category: "symptoms", glosses: ["please", "sleep"] },
  { id: "sf-h-wait-doctor", en: "The doctor will see you soon.", short: "Doctor coming soon", category: "logistics", glosses: ["Doctor", "come"] },
  { id: "sf-h-serious", en: "This is serious. You need to stay in hospital.", short: "You must stay", category: "emergency", urgent: true, glosses: ["Hospital", "you"] },
  { id: "sf-h-return", en: "Come back in one week.", short: "Return in a week", category: "logistics", glosses: ["come", "Week"] },
];

export const HOTEL_STAFF: Phrase[] = [
  ...SHARED_STAFF,
  { id: "sf-ho-booking", en: "Do you have a booking?", short: "Do you have a booking?", category: "logistics" },
  { id: "sf-ho-room-rate", en: "The room rate is shown here.", short: "The rate is here", category: "money", glosses: ["Bedroom", "Price"] },
  { id: "sf-ho-included", en: "Breakfast is included.", short: "Breakfast included", category: "money", glosses: ["food", "Morning"] },
  { id: "sf-ho-key", en: "Here is your key.", short: "Your key", category: "logistics", glosses: ["Key", "give"] },
  { id: "sf-ho-floor", en: "Your room is on this floor.", short: "Your room", category: "logistics", glosses: ["Bedroom", "Location"] },
  { id: "sf-ho-checkout", en: "Checkout is at 11 in the morning.", short: "Checkout time", category: "logistics", glosses: ["Morning", "Time"] },
  { id: "sf-ho-deposit", en: "We need a deposit, which is refunded at checkout.", short: "Deposit needed", category: "money", glosses: ["Money"] },
  { id: "sf-ho-luggage", en: "We can keep your luggage.", short: "We can keep luggage", category: "logistics", glosses: ["Bag"] },
  { id: "sf-ho-alarm", en: "You said you cannot hear the alarm. We will come to your door.", short: "We will come to your door", category: "emergency", urgent: true, glosses: ["Door", "come"] },
  { id: "sf-ho-message", en: "We will message you instead of calling.", short: "We will message you", category: "understanding", glosses: ["phone"] },
  { id: "sf-ho-restaurant", en: "The restaurant is this way.", short: "Restaurant this way", category: "directions", glosses: ["Restaurant", "Location"] },
  { id: "sf-ho-full", en: "I am sorry, we have no rooms free.", short: "No rooms free", category: "logistics", glosses: ["sorry", "Bedroom", "no"] },
];

export const BANK_STAFF: Phrase[] = [
  ...SHARED_STAFF,
  { id: "sf-b-explain-first", en: "I will explain this fully before you sign anything.", short: "I will explain before you sign", category: "understanding", urgent: true },
  { id: "sf-b-no-rush", en: "Do not sign until you are sure.", short: "Do not sign until sure", category: "understanding", urgent: true },
  { id: "sf-b-charges", en: "These are the charges for this service.", short: "The charges", category: "money", urgent: true, glosses: ["Price"] },
  { id: "sf-b-docs-need", en: "I need your identity and address proof.", short: "ID and address proof", category: "logistics", glosses: ["Card", "please"] },
  { id: "sf-b-balance", en: "Your balance is shown on this screen.", short: "Your balance", category: "money", glosses: ["Money"] },
  { id: "sf-b-sign-here", en: "Please sign here.", short: "Sign here", category: "logistics", glosses: ["please", "name"] },
  { id: "sf-b-processing", en: "This will take about ten minutes.", short: "About ten minutes", category: "logistics", glosses: ["Minute"] },
  { id: "sf-b-card-days", en: "Your new card will arrive in about a week.", short: "Card in about a week", category: "logistics", glosses: ["Card", "Week"] },
  { id: "sf-b-blocked", en: "Your card is now blocked. You are not liable for further use.", short: "Card is blocked", category: "emergency", urgent: true, glosses: ["Card", "stop"] },
  { id: "sf-b-counter", en: "Please go to that counter.", short: "That counter", category: "directions", glosses: ["Location"] },
  { id: "sf-b-token", en: "Please take a token and wait for your number.", short: "Take a token", category: "logistics", glosses: ["Card", "wait"] },
  { id: "sf-b-declined", en: "I am sorry, this cannot be approved today.", short: "Cannot approve today", category: "logistics", glosses: ["sorry", "no"] },
];

export const CINEMA_STAFF: Phrase[] = [
  ...SHARED_STAFF,
  { id: "sf-c-subs-yes", en: "Yes, this show has subtitles.", short: "This show has subtitles", category: "needs", urgent: true, glosses: ["yes"] },
  { id: "sf-c-subs-no", en: "This show has no subtitles. These other shows do.", short: "No subtitles on this show", category: "needs", urgent: true, glosses: ["no"] },
  { id: "sf-c-access-seat", en: "We have accessible seating here.", short: "Accessible seating", category: "needs", glosses: ["sit", "Location"] },
  { id: "sf-c-ticket-price", en: "The ticket price is shown here.", short: "Ticket price", category: "money", glosses: ["ticket", "Price"] },
  { id: "sf-c-starts", en: "The show starts at this time.", short: "Start time", category: "logistics", glosses: ["Time"] },
  { id: "sf-c-length", en: "It runs for about two hours.", short: "About two hours", category: "logistics", glosses: ["Hour"] },
  { id: "sf-c-screen", en: "Your screen is this way.", short: "Your screen this way", category: "directions", glosses: ["Location"] },
  { id: "sf-c-seat", en: "Your seat number is on the ticket.", short: "Seat is on the ticket", category: "logistics", glosses: ["ticket", "sit"] },
  { id: "sf-c-announce", en: "You said you cannot hear announcements. We will come and tell you.", short: "We will come and tell you", category: "emergency", urgent: true, glosses: ["come"] },
  { id: "sf-c-sold-out", en: "I am sorry, this show is sold out.", short: "Sold out", category: "logistics", glosses: ["sorry", "no"] },
  { id: "sf-c-refund", en: "We can refund this.", short: "We can refund", category: "money", glosses: ["Money", "give"] },
  { id: "sf-c-food", en: "Food and drink are sold over there.", short: "Food over there", category: "directions", glosses: ["food", "Location"] },
];

export const RETAIL_STAFF: Phrase[] = [
  ...SHARED_STAFF,
  { id: "sf-r-price", en: "The price is shown here.", short: "The price", category: "money", glosses: ["Price"] },
  { id: "sf-r-discount", en: "There is a discount on this.", short: "There is a discount", category: "money", glosses: ["cheap"] },
  { id: "sf-r-stock-yes", en: "Yes, we have it. I will bring it.", short: "We have it", category: "needs", glosses: ["yes", "give"] },
  { id: "sf-r-stock-no", en: "I am sorry, we do not have that.", short: "We do not have it", category: "needs", glosses: ["sorry", "no"] },
  { id: "sf-r-size", en: "Which size do you need?", short: "Which size?", category: "needs", glosses: ["big", "small"] },
  { id: "sf-r-try", en: "You can try it on over there.", short: "Try it on there", category: "directions", glosses: ["Bedroom", "Location"] },
  { id: "sf-r-total", en: "The total is shown on the screen.", short: "The total", category: "money", glosses: ["Bill"] },
  { id: "sf-r-pay-how", en: "Card or cash?", short: "Card or cash?", category: "money", glosses: ["Card", "Money"] },
  { id: "sf-r-return-ok", en: "You can return this within seven days with the bill.", short: "Return within seven days", category: "logistics", glosses: ["back", "give", "Bill"] },
  { id: "sf-r-warranty", en: "This has a one year warranty.", short: "One year warranty", category: "logistics", glosses: ["year"] },
  { id: "sf-r-bag", en: "Would you like a bag?", short: "A bag?", category: "logistics", glosses: ["Bag"] },
  { id: "sf-r-closing", en: "We are closing soon.", short: "Closing soon", category: "logistics", glosses: ["Store or Shop", "close"] },
];

export const TRAVEL_STAFF: Phrase[] = [
  ...SHARED_STAFF,
  { id: "sf-t-where-go", en: "Where do you want to go?", short: "Where to?", category: "transport", glosses: ["you", "Location"] },
  { id: "sf-t-platform", en: "Your platform is this way.", short: "Platform this way", category: "directions", glosses: ["Train Station", "Location"] },
  { id: "sf-t-time", en: "It leaves at this time.", short: "Departure time", category: "transport", glosses: ["Time"] },
  { id: "sf-t-late", en: "It is running late.", short: "Running late", category: "transport", urgent: true, glosses: ["slow"] },
  { id: "sf-t-cancelled", en: "It has been cancelled. I will find you another.", short: "Cancelled", category: "emergency", urgent: true, glosses: ["no", "stop"] },
  { id: "sf-t-fare", en: "The fare is shown here.", short: "The fare", category: "money", glosses: ["Price"] },
  { id: "sf-t-ticket", en: "Here is your ticket.", short: "Your ticket", category: "transport", glosses: ["ticket", "give"] },
  { id: "sf-t-announce", en: "You said you cannot hear announcements. We will come and tell you.", short: "We will come and tell you", category: "understanding", urgent: true, glosses: ["come"] },
  { id: "sf-t-bus", en: "Take the bus from outside.", short: "Bus outside", category: "transport", glosses: ["Bus", "Location"] },
  { id: "sf-t-hotel", en: "There is a hotel nearby.", short: "Hotel nearby", category: "directions", glosses: ["hotel", "Location"] },
  { id: "sf-t-toilet", en: "The toilet is that way.", short: "Toilet that way", category: "directions", glosses: ["Bathroom", "Location"] },
  { id: "sf-t-police", en: "I will call the police for you.", short: "I will call the police", category: "emergency", urgent: true, glosses: ["Police"] },
];

export const STAFF_PHRASES_FOR: Record<DomainId, Phrase[]> = {
  health: HEALTH_STAFF,
  travel: TRAVEL_STAFF,
  hotel: HOTEL_STAFF,
  bank: BANK_STAFF,
  cinema: CINEMA_STAFF,
  retail: RETAIL_STAFF,
};
