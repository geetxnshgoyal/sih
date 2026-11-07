/**
 * The English rendering of every gloss sequence in the corpus.
 *
 * Why these are hand-written and not generated
 * --------------------------------------------
 * Turning an ISL gloss sequence into a sentence is two separate jobs, and they
 * have very different risk profiles:
 *
 *   1. reorder and inflect     "I Doctor"  ->  "I need to see a doctor"
 *   2. translate to 5 languages                 -> hi, ta, te, bn, mr
 *
 * Job 1 needs judgement about what the signer MEANS, "I Doctor" is a request,
 * not a statement: and it is short, English, and reviewable by anyone on the
 * team. So it is written by hand, here, where it can be argued with.
 *
 * Job 2 is mechanical and is done by NLLB-200 running locally
 * (train/translate_glosses.py). Free, open source, no account, no API key, and
 * nothing leaves the machine.
 *
 * That split matters: it puts the part requiring human judgement in a file a
 * human can read, and leaves the machine only the part it is actually good at.
 *
 * Register notes
 * --------------
 * These are said BY a patient or traveller, TO a stranger, often under stress.
 * So: complete sentences, no telegraphic shorthand, polite but not florid, and
 * "I need" rather than "I want" where something is genuinely needed.
 *
 * "I am Deaf" is capitalised deliberately, Deaf as cultural identity. It is
 * usually the first thing a Deaf person must establish before anything else can
 * happen, and it should not read as a medical deficit.
 */

/** gloss sequence joined by "|" -> the English sentence it should become. */
export const GLOSS_ENGLISH: Record<string, string> = {
  // ------------------------------------------------------------------ medical
  "I|sick": "I am sick.",
  "I|Doctor": "I need to see a doctor.",
  "I|Medicine": "I need medicine.",
  "I|Medicine|Time": "When should I take this medicine?",
  "Doctor|Location": "Where is the doctor?",
  "Hospital|Location": "Where is the hospital?",
  "Doctor|Time": "When will the doctor come?",
  "Medicine|Price": "How much does the medicine cost?",
  "I|Doctor|Today": "I need to see a doctor today.",
  "I|Doctor|Tomorrow": "I would like to see a doctor tomorrow.",
  "Patient|I": "I am the patient.",
  "I|Deaf": "I am Deaf.",
  "I|Deaf|Sign": "I am Deaf and I use sign language.",
  "I|sick|Today": "I have been sick today.",
  "I|hot": "I have a fever.",
  "I|cold": "I feel cold.",
  "I|weak": "I feel weak.",
  "I|healthy": "I am feeling well.",
  "Baby|sick": "My baby is sick.",
  "Child|sick": "My child is sick.",
  "Mother|sick": "My mother is sick.",
  "Father|sick": "My father is sick.",
  "Medicine|Time": "What time is my medicine due?",
  "I|Bathroom": "I need to use the bathroom.",
  "Bathroom|Location": "Where is the bathroom?",
  "Doctor|good": "The doctor is good.",
  "I|Hospital|Today": "I went to the hospital today.",

  // ------------------------------------------------------------------- travel
  "Train Station|Location": "Where is the railway station?",
  "I|train ticket": "I need a train ticket.",
  "train ticket|Price": "How much does a train ticket cost?",
  "Train|Time": "When does the train leave?",
  "Bus|Location": "Where is the bus stop?",
  "Bus|Time": "When does the bus leave?",
  "I|Bus": "I need to take the bus.",
  "Restaurant|Location": "Where can I find a restaurant?",
  "Market|Location": "Where is the market?",
  "Bank|Location": "Where is the bank?",
  "Street or Road|Location": "Which road is this?",
  "I|Money": "I need to withdraw money.",
  "Price": "How much does this cost?",
  "cheap": "This is cheap.",
  "expensive": "This is too expensive.",
  "Train|Time|Tomorrow": "When does the train leave tomorrow?",
  "I|Car": "I need a taxi.",
  "Plane|Time": "When does the flight leave?",
  "Temple|Location": "Where is the temple?",
  "Park|Location": "Where is the park?",
  "Library|Location": "Where is the library?",

  // -------------------------------------------------------------------- civic
  "Police|Location": "Where is the police station?",
  "I|Police": "I need the police.",
  "School|Location": "Where is the school?",
  "Office|Location": "Where is the office?",
  "I|Student": "I am a student.",
  "I|Teacher": "I am a teacher.",
  "I|Job": "I am looking for work.",
  "Bill|Price": "How much is the bill?",

  // ------------------------------------------------------------------- social
  "Hello": "Hello.",
  "Thank you": "Thank you.",
  "Good Morning": "Good morning.",
  "Good afternoon": "Good afternoon.",
  "Good evening": "Good evening.",
  "Good night": "Good night.",
  "How are you": "How are you?",
  "I|happy": "I am happy.",
  "I|sad": "I am sad.",
  "Pleased": "I am pleased to meet you.",
  "I|Friend": "This is my friend.",
  "I|Family": "This is my family.",
  "Mother|Father": "These are my parents.",
  "you|good": "You are kind.",
  "Alright": "That is alright.",
  "I|Sign": "I use sign language.",
  "I|Deaf|you|Sign": "I am Deaf. Can you sign?",
};
