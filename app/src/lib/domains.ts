/**
 * Deployment domains: the same bridge, one setting at a time.
 *
 * Setu goes to SIH under several themes: MedTech, Travel & Tourism, and the
 * tertiary-sector statement covering hospitality, financial services,
 * entertainment and retail. That is not several products. The recognition
 * stack is identical and entirely domain-neutral: one 38-sign classifier, one
 * 203-word dictionary bank, one feature contract, one segmenter. "Doctor" and
 * "Train Station" are the same kind of sign to the model, and nothing below
 * the UI knows which setting it is in.
 *
 * What genuinely differs between a hospital reception and a bank counter is
 * only ever four things:
 *
 *   1. which phrases a person reaches for first
 *   2. what the spoken words map onto  ("clinic" -> Hospital, "teller" -> Bank)
 *   3. what the place calls itself
 *   4. what the two people at the counter are called
 *
 * The fourth one used to be hardcoded as doctor and patient, which made a
 * retail till read as a hospital. It is data now, like the other three. So a
 * domain is still a data entry rather than a rewrite, and the claim that the
 * Travel and tertiary-sector submissions are honestly the same system is
 * something the code actually backs up rather than something we assert.
 *
 * GLOSS CONTRACT: every gloss in `quick` and every value in `synonyms` must be
 * a key of app/public/model/_signs.json (the 179 playback recordings), NOT of
 * labels.json. Those are different vocabularies and only the first can be
 * played back. `npm run check:glosses` enforces it.
 */

export type DomainId = "health" | "travel" | "hotel" | "bank" | "cinema" | "retail";

/** Which side of the counter a person is on. The labels are per domain. */
export type RoleId = "staff" | "client";

export interface QuickPhrase {
  /** Signs in ISL order. Every gloss must exist in _signs.json. */
  glosses: string[];
  /** Short English caption for the button face. */
  caption: string;
  /** Marks the phrases a person needs when something is going wrong. */
  urgent?: boolean;
}

export interface Domain {
  id: DomainId;
  label: string;
  /** Shown under the title: one line, plain. */
  tagline: string;
  /** What the kiosk calls its location. */
  station: string;
  /** Broad sector, used to group the picker. */
  sector: "Healthcare" | "Travel" | "Hospitality" | "Financial services" | "Entertainment" | "Retail";
  /**
   * What the two people are called here.
   *
   * `staff` serves, `client` is served. Both need a noun ("Doctor"), a
   * possessive-friendly lowercase form for mid-sentence use ("the doctor"),
   * and the word for what is happening between them.
   */
  roles: {
    staff: { label: string; lower: string };
    client: { label: string; lower: string };
  };
  /** What an exchange is called here: a consultation, a visit, a transaction. */
  session: string;
  /** Tapped by the Deaf user to say something common. */
  quick: QuickPhrase[];
  /** Spoken word -> gloss, layered on top of the shared map in reverse.ts. */
  synonyms: Record<string, string>;
}

/**
 * Identity and courtesy, needed in every setting.
 *
 * "I Deaf" leads deliberately. It is the first thing a Deaf person usually has
 * to establish, in any setting, before anything else can happen, and it is the
 * one phrase whose absence from a demo would be conspicuous.
 *
 * It used to be followed by ["I","Deaf","Sign"], which silently degraded to
 * text because there is no "Sign" recording in _signs.json. Replaced with
 * glosses that actually play.
 */
const SHARED_QUICK: QuickPhrase[] = [
  { glosses: ["I", "Deaf"], caption: "I am Deaf" },
  { glosses: ["I", "Deaf", "please", "understand"], caption: "I am Deaf: please understand" },
  { glosses: ["Hello"], caption: "Hello" },
  { glosses: ["Thank you"], caption: "Thank you" },
  { glosses: ["please", "wait"], caption: "Please wait" },
  { glosses: ["I", "no", "understand"], caption: "I do not understand" },
];

/** Courtesy and transaction words every counter in the service sectors needs. */
const COUNTER_QUICK: QuickPhrase[] = [
  { glosses: ["Price"], caption: "How much" },
  { glosses: ["expensive"], caption: "Too expensive" },
  { glosses: ["I", "Card"], caption: "I will pay by card" },
  { glosses: ["I", "Money"], caption: "I will pay cash" },
  { glosses: ["Bill", "please"], caption: "The bill please" },
  { glosses: ["please", "more", "slow"], caption: "Please go slower" },
  { glosses: ["Bathroom", "Location"], caption: "Where is the toilet" },
];

export const DOMAINS: Record<DomainId, Domain> = {
  health: {
    id: "health",
    label: "Healthcare",
    sector: "Healthcare",
    tagline: "Hospital reception, OPD and emergency triage",
    station: "RECEPTION · OPD",
    roles: {
      staff: { label: "Doctor", lower: "doctor" },
      client: { label: "Patient", lower: "patient" },
    },
    session: "consultation",
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
      { glosses: ["I", "fever"], caption: "I have a fever", urgent: true },
      { glosses: ["I", "weak"], caption: "I feel weak", urgent: true },
      { glosses: ["Child", "sick"], caption: "My child is sick", urgent: true },
      { glosses: ["I", "pain"], caption: "I am in pain", urgent: true },
      { glosses: ["Bathroom", "Location"], caption: "Where is the bathroom" },
    ],
    synonyms: {
      physician: "Doctor", डॉक्टर: "Doctor", doctor: "Doctor",
      clinic: "Hospital", अस्पताल: "Hospital", ward: "Hospital", opd: "Hospital",
      medicine: "Medicine", dawai: "Medicine", दवा: "Medicine",
      tablet: "tablet", pill: "tablet", dose: "Medicine",
      ill: "sick", unwell: "sick", bimar: "sick", बीमार: "sick",
      fever: "fever", बुखार: "fever", temperature: "fever",
      nurse: "nurse", chemist: "Medicine", pharmacy: "Medicine",
      appointment: "Time", report: "Card", prescription: "Card",
    },
  },

  travel: {
    id: "travel",
    label: "Travel & Tourism",
    sector: "Travel",
    tagline: "Station, airport and tourist help point",
    station: "ENQUIRY · HELP DESK",
    roles: {
      staff: { label: "Help desk", lower: "help desk" },
      client: { label: "Traveller", lower: "traveller" },
    },
    session: "enquiry",
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
      platform: "station", station: "station", स्टेशन: "Train Station",
      rail: "Train", railway: "Train", ट्रेन: "Train",
      ticket: "ticket", टिकट: "ticket", fare: "Price", किराया: "Price",
      airport: "airport", flight: "Plane", हवाई: "Plane",
      taxi: "Car", auto: "Car", rickshaw: "Car", cab: "Car",
      hotel: "hotel", lodge: "hotel", stay: "hotel", room: "Bedroom",
      food: "food", eat: "eat", khana: "food", खाना: "food",
      dhaba: "Restaurant", canteen: "Restaurant",
      atm: "Bank", बैंक: "Bank", exchange: "Money", currency: "Money",
      bazaar: "Market", बाज़ार: "Market", shopping: "Market",
      mandir: "Temple", मंदिर: "Temple", museum: "Library",
      luggage: "Bag", bag: "Bag", सामान: "Bag",
      guide: "Teacher", tourist: "you", map: "Card",
    },
  },

  hotel: {
    id: "hotel",
    label: "Hotel front desk",
    sector: "Hospitality",
    tagline: "Check-in, room, tariff and checkout",
    station: "FRONT DESK · RECEPTION",
    roles: {
      staff: { label: "Front desk", lower: "front desk" },
      client: { label: "Guest", lower: "guest" },
    },
    session: "stay",
    quick: [
      ...SHARED_QUICK,
      { glosses: ["I", "Bedroom"], caption: "I need a room" },
      { glosses: ["Bedroom", "Price"], caption: "What does the room cost" },
      { glosses: ["Bedroom", "Location"], caption: "Where is my room" },
      { glosses: ["I", "Key"], caption: "I need the key" },
      { glosses: ["Key", "no"], caption: "My key does not work" },
      { glosses: ["I", "hotel", "Today"], caption: "I am staying tonight" },
      { glosses: ["I", "hotel", "Tomorrow"], caption: "I am leaving tomorrow" },
      { glosses: ["food", "Time"], caption: "When is food served" },
      { glosses: ["Restaurant", "Location"], caption: "Where is the restaurant" },
      { glosses: ["water", "please"], caption: "Water please" },
      { glosses: ["Bedroom", "hot"], caption: "The room is too hot" },
      { glosses: ["Bedroom", "cold"], caption: "The room is too cold" },
      { glosses: ["I", "Bag"], caption: "My luggage" },
      { glosses: ["I", "help"], caption: "I need help", urgent: true },
      ...COUNTER_QUICK,
    ],
    synonyms: {
      room: "Bedroom", suite: "Bedroom", कमरा: "Bedroom",
      hotel: "hotel", lodge: "hotel", guesthouse: "hotel", होटल: "hotel",
      reception: "hotel", lobby: "hotel", desk: "hotel",
      key: "Key", keycard: "Key", चाबी: "Key",
      tariff: "Price", rate: "Price", rent: "Price", किराया: "Price",
      checkout: "Time", checkin: "Time", booking: "Card", reservation: "Card",
      breakfast: "Morning", dinner: "Night", lunch: "Afternoon",
      restaurant: "Restaurant", kitchen: "Kitchen", food: "food",
      luggage: "Bag", baggage: "Bag", सामान: "Bag",
      towel: "Bag", bathroom: "Bathroom", washroom: "Bathroom",
      wifi: "phone", ac: "cool", heater: "warm",
      porter: "help", housekeeping: "help", manager: "Office",
      id: "Card", passport: "Card", aadhaar: "Card", document: "Card",
    },
  },

  bank: {
    id: "bank",
    label: "Bank counter",
    sector: "Financial services",
    tagline: "Account, balance, documents and consent",
    station: "COUNTER · BRANCH",
    roles: {
      staff: { label: "Teller", lower: "teller" },
      client: { label: "Customer", lower: "customer" },
    },
    session: "visit",
    quick: [
      ...SHARED_QUICK,
      { glosses: ["I", "Bank", "new"], caption: "I want to open an account" },
      { glosses: ["I", "Money", "give"], caption: "I want to deposit" },
      { glosses: ["I", "Money", "take"], caption: "I want to withdraw" },
      { glosses: ["Money", "Price"], caption: "What is my balance" },
      { glosses: ["I", "Card", "no"], caption: "My card does not work" },
      { glosses: ["I", "Card", "new"], caption: "I need a new card" },
      { glosses: ["Bill", "Location"], caption: "Where do I pay" },
      { glosses: ["I", "Card", "no", "understand"], caption: "I do not understand this form", urgent: true },
      { glosses: ["please", "more", "Time"], caption: "Please give me more time" },
      { glosses: ["I", "Card", "give"], caption: "Here are my documents" },
      { glosses: ["Time", "more"], caption: "How long will it take" },
      { glosses: ["Office", "Location"], caption: "Where is the manager" },
      { glosses: ["stop"], caption: "Stop, I have a question", urgent: true },
      ...COUNTER_QUICK,
    ],
    synonyms: {
      bank: "Bank", branch: "Bank", बैंक: "Bank", teller: "Bank",
      atm: "Bank", cashier: "Money", counter: "Bank",
      account: "Bank", khata: "Bank", खाता: "Bank",
      balance: "Money", deposit: "Money", withdraw: "Money",
      cash: "Money", rupees: "Money", पैसा: "Money", paisa: "Money",
      card: "Card", debit: "Card", credit: "Card", atmcard: "Card",
      cheque: "Card", check: "Card", passbook: "Card", statement: "Card",
      form: "Card", document: "Card", kyc: "Card", aadhaar: "Card",
      pan: "Card", passport: "Card", signature: "name", sign: "name",
      loan: "Money", interest: "Price", emi: "Bill", instalment: "Bill",
      bill: "Bill", payment: "Bill", transfer: "give",
      manager: "Office", officer: "Office", queue: "wait", token: "Card",
    },
  },

  cinema: {
    id: "cinema",
    label: "Box office",
    sector: "Entertainment",
    tagline: "Cinema, venue and event counter",
    station: "BOX OFFICE · VENUE",
    roles: {
      staff: { label: "Box office", lower: "box office" },
      client: { label: "Guest", lower: "guest" },
    },
    session: "visit",
    quick: [
      ...SHARED_QUICK,
      { glosses: ["I", "ticket"], caption: "I need a ticket" },
      { glosses: ["ticket", "Price"], caption: "How much is a ticket" },
      { glosses: ["ticket", "Time"], caption: "What time does it start" },
      { glosses: ["I", "Deaf", "understand", "no"], caption: "I am Deaf: does it have subtitles", urgent: true },
      { glosses: ["I", "sit", "Location"], caption: "Where is my seat" },
      { glosses: ["Door", "Location"], caption: "Which entrance" },
      { glosses: ["Time", "more"], caption: "How long is it" },
      { glosses: ["Child", "ticket"], caption: "A ticket for a child" },
      { glosses: ["food", "Location"], caption: "Where can I buy food" },
      { glosses: ["ticket", "Tomorrow"], caption: "A ticket for tomorrow" },
      { glosses: ["I", "help"], caption: "I need help", urgent: true },
      ...COUNTER_QUICK,
    ],
    synonyms: {
      cinema: "Court", theatre: "Court", theater: "Court", hall: "Court",
      movie: "Court", film: "Court", show: "Time", screening: "Time",
      ticket: "ticket", टिकट: "ticket", booking: "ticket", seat: "sit",
      row: "sit", balcony: "up", screen: "Court", interval: "Time",
      subtitle: "understand", subtitles: "understand", caption: "understand",
      popcorn: "food", snack: "food", canteen: "Restaurant",
      entrance: "Door", exit: "Door", gate: "Door",
      refund: "Money", cancel: "stop", event: "Time", concert: "Time",
      stadium: "Ground", match: "Ground", museum: "Library",
    },
  },

  retail: {
    id: "retail",
    label: "Shop counter",
    sector: "Retail",
    tagline: "Till, price, size and returns",
    station: "TILL · STORE",
    roles: {
      staff: { label: "Staff", lower: "staff member" },
      client: { label: "Customer", lower: "customer" },
    },
    session: "visit",
    quick: [
      ...SHARED_QUICK,
      { glosses: ["Price"], caption: "How much is this" },
      { glosses: ["expensive"], caption: "Too expensive" },
      { glosses: ["cheap", "more"], caption: "Something cheaper" },
      { glosses: ["big", "more"], caption: "A bigger size" },
      { glosses: ["small", "more"], caption: "A smaller size" },
      { glosses: ["new", "give"], caption: "A new one please" },
      { glosses: ["I", "take"], caption: "I will take it" },
      { glosses: ["I", "no", "take"], caption: "I will not take it" },
      { glosses: ["Bill", "please"], caption: "The bill please" },
      { glosses: ["I", "Card"], caption: "I will pay by card" },
      { glosses: ["I", "Money"], caption: "I will pay cash" },
      { glosses: ["back", "give"], caption: "Can I return this" },
      { glosses: ["Store or Shop", "Time"], caption: "When do you close" },
      { glosses: ["I", "help"], caption: "I need help" },
      { glosses: ["Bathroom", "Location"], caption: "Where is the toilet" },
    ],
    synonyms: {
      shop: "Store or Shop", store: "Store or Shop", दुकान: "Store or Shop",
      counter: "Store or Shop", till: "Bill", checkout: "Bill",
      price: "Price", cost: "Price", rate: "Price", दाम: "Price", कीमत: "Price",
      discount: "cheap", sale: "cheap", offer: "cheap", cheaper: "cheap",
      size: "big", large: "big", medium: "big", xl: "big",
      colour: "new", color: "new", brand: "name",
      bill: "Bill", receipt: "Bill", invoice: "Bill",
      cash: "Money", card: "Card", upi: "phone", qr: "phone",
      change: "Money", refund: "Money", return: "back", exchange: "back",
      warranty: "Card", guarantee: "Card", bag: "Bag",
      trial: "Bedroom", fitting: "Bedroom", stock: "more",
      grocery: "Market", vegetable: "food", clothes: "Bag",
    },
  },
};

export const DOMAIN_LIST: Domain[] = [
  DOMAINS.health, DOMAINS.hotel, DOMAINS.bank,
  DOMAINS.cinema, DOMAINS.retail, DOMAINS.travel,
];

const VALID: ReadonlySet<string> = new Set<DomainId>([
  "health", "travel", "hotel", "bank", "cinema", "retail",
]);

const DEFAULT_DOMAIN: DomainId = "health";

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
    if (v !== null && VALID.has(v)) return v as DomainId;
  } catch {
    // storage unavailable: fall through to the default
  }
  return DEFAULT_DOMAIN;
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
  return DEFAULT_DOMAIN;
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
