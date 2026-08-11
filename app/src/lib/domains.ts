/**
 * Deployment domains — the same bridge, two settings.
 *
 * Setu is going to SIH under two themes: MedTech and Travel & Tourism. That is
 * not two products. The recognition stack is identical and entirely
 * domain-neutral: one 264-sign classifier, one feature contract, one segmenter.
 * INCLUDE is a general lexicon — "Doctor" and "Train Station" are the same kind
 * of sign to the model, and nothing below the UI knows which setting it is in.
 *
 * What genuinely differs between a hospital reception and a tourist help desk is
 * only ever three things:
 *
 *   1. which phrases a person reaches for first
 *   2. what the spoken words map onto  ("clinic" -> Hospital, "platform" -> Train Station)
 *   3. what the kiosk calls itself
 *
 * So that is all a domain is. Keeping it this small is the point: it means the
 * Travel submission is honestly the same system, not a fork maintained twice,
 * and a third setting (civic services, railway enquiry) is a data entry rather
 * than a rewrite.
 *
 * Quick phrases are gloss SEQUENCES, not sentences. They are rendered through
 * the same pipeline as a live recognition, so what a judge sees when they tap
 * one is exactly what they would see if someone signed it.
 */

export type DomainId = "health" | "travel";

export interface QuickPhrase {
  /** Signs in ISL order. Every gloss must exist in models/labels.json. */
  glosses: string[];
  /** Short English caption for the button face. */
  caption: string;
  /** Marks the phrases a person needs when something is going wrong. */
  urgent?: boolean;
}

export interface Domain {
  id: DomainId;
  label: string;
  /** Shown under the title — one line, plain. */
  tagline: string;
  /** What the kiosk calls its location. */
  station: string;
  /** Tapped by the Deaf user to say something common. */
  quick: QuickPhrase[];
  /** Spoken word -> gloss, layered on top of the shared map in reverse.ts. */
  synonyms: Record<string, string>;
}

/**
 * Identity and courtesy, needed in every setting.
 *
 * "I Deaf" leads deliberately. It is the first thing a Deaf person usually has
 * to establish, in any setting, before anything else can happen — and it is the
 * one phrase whose absence from a demo would be conspicuous.
 */
const SHARED_QUICK: QuickPhrase[] = [
  { glosses: ["I", "Deaf"], caption: "I am Deaf" },
  { glosses: ["I", "Deaf", "Sign"], caption: "I am Deaf — I sign" },
  { glosses: ["Hello"], caption: "Hello" },
  { glosses: ["Thank you"], caption: "Thank you" },
];

export const DOMAINS: Record<DomainId, Domain> = {
  health: {
    id: "health",
    label: "Healthcare",
    tagline: "Hospital reception, OPD and emergency triage",
    station: "RECEPTION · OPD",
    quick: [
      ...SHARED_QUICK,
      { glosses: ["I", "sick"], caption: "I am sick", urgent: true },
      { glosses: ["I", "Doctor"], caption: "I need a doctor", urgent: true },
      { glosses: ["Doctor", "Location"], caption: "Where is the doctor" },
      { glosses: ["Hospital", "Location"], caption: "Where is the hospital" },
      { glosses: ["I", "Medicine"], caption: "I need medicine" },
      { glosses: ["Medicine", "Time"], caption: "When is my medicine" },
      { glosses: ["Medicine", "Price"], caption: "What does it cost" },
      { glosses: ["Doctor", "Time"], caption: "When will the doctor come" },
      { glosses: ["Patient", "I"], caption: "I am the patient" },
      { glosses: ["I", "hot"], caption: "I have a fever", urgent: true },
      { glosses: ["I", "weak"], caption: "I feel weak", urgent: true },
      { glosses: ["Child", "sick"], caption: "My child is sick", urgent: true },
      { glosses: ["Bathroom", "Location"], caption: "Where is the bathroom" },
    ],
    synonyms: {
      physician: "Doctor", डॉक्टर: "Doctor", doctor: "Doctor",
      clinic: "Hospital", अस्पताल: "Hospital", ward: "Hospital", opd: "Hospital",
      medicine: "Medicine", dawai: "Medicine", दवा: "Medicine",
      tablet: "Medicine", pill: "Medicine", dose: "Medicine",
      ill: "sick", unwell: "sick", bimar: "sick", बीमार: "sick",
      fever: "hot", बुखार: "hot", temperature: "hot",
      nurse: "Doctor", chemist: "Medicine", pharmacy: "Medicine",
      appointment: "Time", report: "Paper", prescription: "Paper",
    },
  },

  travel: {
    id: "travel",
    label: "Travel & Tourism",
    tagline: "Station, airport, hotel desk and tourist help point",
    station: "ENQUIRY · HELP DESK",
    quick: [
      ...SHARED_QUICK,
      { glosses: ["Train Station", "Location"], caption: "Where is the station" },
      { glosses: ["I", "train ticket"], caption: "I need a ticket" },
      { glosses: ["train ticket", "Price"], caption: "How much is the ticket" },
      { glosses: ["Train", "Time"], caption: "When is the train" },
      { glosses: ["Bus", "Location"], caption: "Where is the bus" },
      { glosses: ["Bus", "Time"], caption: "When is the bus" },
      { glosses: ["Plane", "Time"], caption: "When is the flight" },
      { glosses: ["Restaurant", "Location"], caption: "Where can I eat" },
      { glosses: ["Bathroom", "Location"], caption: "Where is the toilet" },
      { glosses: ["Bank", "Location"], caption: "Where is a bank" },
      { glosses: ["Market", "Location"], caption: "Where is the market" },
      { glosses: ["Temple", "Location"], caption: "Where is the temple" },
      { glosses: ["Price"], caption: "How much" },
      { glosses: ["expensive"], caption: "Too expensive" },
      { glosses: ["I", "Police"], caption: "I need the police", urgent: true },
      { glosses: ["Police", "Location"], caption: "Where is the police", urgent: true },
    ],
    synonyms: {
      platform: "Train Station", station: "Train Station", स्टेशन: "Train Station",
      rail: "Train", railway: "Train", ट्रेन: "Train",
      ticket: "train ticket", टिकट: "train ticket", fare: "Price", किराया: "Price",
      airport: "Plane", flight: "Plane", हवाई: "Plane",
      taxi: "Car", auto: "Car", rickshaw: "Car", cab: "Car",
      hotel: "House", lodge: "House", stay: "House", room: "Bedroom",
      food: "Restaurant", eat: "Restaurant", khana: "Restaurant", खाना: "Restaurant",
      dhaba: "Restaurant", canteen: "Restaurant",
      atm: "Bank", बैंक: "Bank", exchange: "Money", currency: "Money",
      bazaar: "Market", बाज़ार: "Market", shopping: "Market",
      mandir: "Temple", मंदिर: "Temple", museum: "Library",
      luggage: "Bag", bag: "Bag", सामान: "Bag",
      guide: "Teacher", tourist: "you", map: "Paper",
    },
  },
};

export const DOMAIN_LIST: Domain[] = [DOMAINS.health, DOMAINS.travel];

const STORAGE_KEY = "setu.domain";

/**
 * The chosen setting is remembered per device, so a kiosk keeps its domain
 * across reloads. Exposed as a subscribable store rather than something a
 * component reads in an effect: localStorage is unavailable during prerender
 * and throws in some privacy modes, so the value has to start at a safe default
 * and be adopted on the client without a cascading re-render.
 */
const listeners = new Set<() => void>();
let current: DomainId | null = null;

function read(): DomainId {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "health" || v === "travel") return v;
  } catch {
    // storage unavailable — fall through to the default
  }
  return "health";
}

export function subscribeDomain(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getDomain(): DomainId {
  if (current === null) current = read();
  return current;
}

/** No localStorage before hydration; start on the default and adopt after. */
export function getServerDomain(): DomainId {
  return "health";
}

export function saveDomain(id: DomainId): void {
  current = id;
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // non-fatal: the choice simply will not persist
  }
  listeners.forEach((l) => l());
}
