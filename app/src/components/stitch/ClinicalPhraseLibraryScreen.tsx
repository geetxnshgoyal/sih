import { useState } from "react";
import { Search, Send, Check, MessageSquare } from "lucide-react";
import { useSession } from "../../context/SessionContext";


interface ClinicalPhrase {
  id: string;
  category: "symptoms" | "examination" | "medication" | "emergency";
  tag: string;
  code: string;
  english: string;
  hindi: string;
  tamil: string;
  glossSequence: string;
  glosses: string[];
  duration: string;
}

const PHRASE_DATA: ClinicalPhrase[] = [
  {
    id: "p1",
    category: "medication",
    tag: "POST-MEAL",
    code: "RX-01",
    english: "Take 1 tablet after meals",
    hindi: "भोजन के बाद 1 गोली लें",
    tamil: "சாப்பாட்டுக்குப் பிறகு 1 மாத்திரை",
    glossSequence: "EAT (FOOD) ➔ 1 FINGER ➔ PILL-SWALLOW GESTURE",
    glosses: ["MEDICINE TABLET", "EAT AFTER"],
    duration: "2.8s",
  },
  {
    id: "p2",
    category: "medication",
    tag: "RESTRICTION",
    code: "RX-02",
    english: "Do not drink cold water",
    hindi: "ठंडा पानी मत पिएं",
    tamil: "குளிர்ந்த நீர் குடிக்க வேண்டாம்",
    glossSequence: "COLD-SHIVER ➔ WATER DRINK ➔ NEGATIVE HEADSHAKE",
    glosses: ["WATER", "HELP"],
    duration: "2.1s",
  },
  {
    id: "p3",
    category: "examination",
    tag: "LABORATORY / FASTING",
    code: "LAB-09",
    english: "Take blood test on empty stomach tomorrow morning",
    hindi: "कल सुबह खाली पेट खून की जांच कराएं",
    tamil: "நாளை காலை வெறும் வயிற்றில் ரத்த பரிசோதனை",
    glossSequence: "SUNRISE (TOMORROW) ➔ STOMACH EMPTY ➔ ARM NEEDLE PINCH",
    glosses: ["GOOD MORNING", "HELP"],
    duration: "3.6s",
  },
  {
    id: "p4",
    category: "medication",
    tag: "TOPICAL DOSAGE",
    code: "RX-03",
    english: "Apply ointment twice daily",
    hindi: "दिन में दो बार मरहम लगाएं",
    tamil: "தினமும் இரண்டு முறை களிம்பு தடவவும்",
    glossSequence: "FOREARM RUBBING (OINTMENT) ➔ TWO TIMES (DAY + NIGHT)",
    glosses: ["HELP", "THANK YOU"],
    duration: "2.5s",
  },
  {
    id: "p5",
    category: "symptoms",
    tag: "PAIN ASSESSMENT",
    code: "SYM-01",
    english: "Where does it hurt the most? Point to the spot",
    hindi: "सबसे ज्यादा दर्द कहां हो रहा है? जगह दिखाएं",
    tamil: "வலி எங்கு அதிகமாக உள்ளது?",
    glossSequence: "PAIN GESTURE ➔ POINT TO BODY ➔ QUESTION FINGER",
    glosses: ["HOW ARE YOU", "HELP"],
    duration: "2.4s",
  },
  {
    id: "p6",
    category: "emergency",
    tag: "RESPIRATORY / URGENT",
    code: "EMG-04",
    english: "Take a deep breath and hold it for 3 seconds",
    hindi: "गहरी सांस लें और 3 सेकंड के लिए रोकें",
    tamil: "ஆழ்ந்த மூச்சை எடுத்து 3 வினாடிகள் பிடித்து வைக்கவும்",
    glossSequence: "EXPAND CHEST GESTURE ➔ THREE COUNT FINGERS ➔ HOLD",
    glosses: ["HELP", "WATER"],
    duration: "4.0s",
  },
];

export function ClinicalPhraseLibraryScreen() {
  const { projectToPatient, selectedLang } = useSession();
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [sent, setSent] = useState("");
  const [status, setStatus] = useState("");
  const phrases = PHRASE_DATA.filter(p => (category === "all" || p.category === category) && `${p.english} ${p.hindi} ${p.tamil}`.toLowerCase().includes(query.toLowerCase().trim()));
  function send(phrase: ClinicalPhrase) {
    const text = selectedLang === "hi-IN" ? phrase.hindi : selectedLang === "ta-IN" ? phrase.tamil : phrase.english;
    projectToPatient({text, textEn: phrase.english});
    setSent(phrase.id); setStatus(`Sent to the patient view: ${text}`);
  }
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div className="flex items-center justify-between"><div><span className="eyebrow-label">Quick communication</span><h1 className="text-display-md font-bold mt-2">Clinical phrase library</h1><p className="text-secondary mt-3">Choose a phrase to add it to the conversation.</p></div><a href="#bridge" className="secondary-action">Return to consultation</a></div>
    <section className="content-card flex flex-col gap-4"><div className="flex items-center gap-3 border border-outline-variant rounded-xl px-4 py-3"><Search size={20} className="text-secondary shrink-0"/><label htmlFor="phrase-search" className="sr-only">Search phrases</label><input id="phrase-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search phrases…" className="w-full bg-transparent outline-none"/></div><div className="flex flex-wrap gap-2" role="group" aria-label="Phrase categories">{["all", "symptoms", "examination", "medication", "emergency"].map(value => <button key={value} className={`${category === value ? "primary-action" : "secondary-action"} capitalize`} aria-pressed={category === value} onClick={() => setCategory(value)}>{value === "all" ? "All phrases" : value}</button>)}</div></section>
    {selectedLang !== "hi-IN" && selectedLang !== "ta-IN" && selectedLang !== "en-IN" && <p className="text-label-md text-secondary">These phrases are available in English, Hindi, and Tamil. English is shown for your selected language.</p>}
    {status && <div role="status" className="content-card text-primary flex flex-wrap items-center justify-between gap-3"><span>{status}</span><a href="#bridge" className="secondary-action">View message</a></div>}
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">{phrases.map(phrase => <article key={phrase.id} className="content-card flex flex-col gap-4"><span className="eyebrow-label">{phrase.category}</span><h2 className="text-headline-md font-semibold">{phrase.english}</h2>{selectedLang === "hi-IN" && <p className="text-body-lg text-secondary">{phrase.hindi}</p>}{selectedLang === "ta-IN" && <p className="text-body-lg text-secondary">{phrase.tamil}</p>}<button className="secondary-action mt-auto" onClick={() => send(phrase)}>{sent === phrase.id ? <Check size={18}/> : <Send size={18}/>} {sent === phrase.id ? "Send again" : "Send to patient"}</button></article>)}</div>
    {!phrases.length && <section className="content-card empty-state"><MessageSquare size={32} className="mx-auto mb-4"/><h2 className="text-headline-md font-semibold">No matching phrases</h2><p className="mt-2">Try a different search or category.</p><button className="secondary-action mt-4" onClick={() => {setQuery(""); setCategory("all");}}>Clear filters</button></section>}
  </div></main>;
}
