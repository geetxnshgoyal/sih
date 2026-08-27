import { useEffect, useMemo, useState } from "react";
import { Search, Send, Check, MessageSquare } from "lucide-react";
import { useSession } from "../../context/SessionContext";
import { useDomain } from "../../lib/useDomain";
import {
  CATEGORY_LABEL, STAFF_PHRASES_FOR, type Phrase, type PhraseCategory,
} from "../../lib/phrasebook";
import { loadPhrasebook, phraseText } from "../../lib/phrasebookTable";

/**
 * The staff side of the phrase board.
 *
 * The client board (components/PhraseBoard) is the Deaf person speaking
 * outward. This is the other direction: the person behind the counter picks a
 * sentence and it is projected to the client view as text, spoken, and played
 * back as signs where a recording exists.
 *
 * WHAT THIS REPLACES, because it is worth remembering. This screen used to
 * hold seven hardcoded clinical sentences in three languages, shown at all six
 * counters, so a shop assistant was offered "Take 1 tablet after meals" and a
 * Telugu, Bengali or Marathi user got English. Its gloss sequences were
 * invented: "Apply ointment twice daily" was mapped to HELP THANK YOU, and its
 * uppercase glosses matched no recording in any vocabulary. It was scaffolding
 * that never got finished and sat on a top-level nav item.
 *
 * Now it reads STAFF_PHRASES_FOR[domain] and the same committed NLLB table the
 * client board uses, so all six languages work and the phrases match the
 * counter. Categories are derived from the data rather than hardcoded, which
 * is what stopped "medication" appearing in a shop.
 */
export function PhraseLibraryScreen() {
  const { projectToClient, selectedLang } = useSession();
  const domain = useDomain();
  const [category, setCategory] = useState<PhraseCategory | "all">("all");
  const [query, setQuery] = useState("");
  const [sent, setSent] = useState("");
  const [status, setStatus] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => { void loadPhrasebook().then(setReady); }, []);
  // Switching counters must not leave a filter selected that this board has
  // no phrases for, which would show an empty state that looks like a fault.
  useEffect(() => { setCategory("all"); }, [domain.id]);

  const all = STAFF_PHRASES_FOR[domain.id];

  const categories = useMemo(() => {
    const seen: PhraseCategory[] = [];
    for (const p of all) if (!seen.includes(p.category)) seen.push(p.category);
    return seen;
  }, [all]);

  const phrases = useMemo(() => {
    const q = query.trim().toLowerCase();
    return all.filter(
      (p) =>
        (category === "all" || p.category === category) &&
        (!q || p.en.toLowerCase().includes(q) || (p.short ?? "").toLowerCase().includes(q))
    );
  }, [all, category, query]);

  function send(phrase: Phrase) {
    const text = phraseText(phrase, selectedLang);
    projectToClient({ text, textEn: phrase.en, glosses: phrase.glosses });
    setSent(phrase.id);
    setStatus(`Sent to the ${domain.roles.client.lower} view: ${text}`);
  }

  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div className="flex items-center justify-between gap-3 flex-wrap">
      <div>
        <span className="eyebrow-label">{domain.station}</span>
        <h1 className="text-display-md font-bold mt-2">{domain.label} phrases</h1>
        <p className="text-secondary mt-3">
          Pick a sentence to send to the {domain.roles.client.lower} view.
          It appears as text, is read aloud, and plays as signs where a
          recording exists.
        </p>
      </div>
      <a href="#bridge" className="secondary-action">Return to the counter</a>
    </div>

    <section className="content-card flex flex-col gap-4">
      <div className="flex items-center gap-3 border border-outline-variant rounded-xl px-4 py-3">
        <Search size={20} className="text-secondary shrink-0"/>
        <label htmlFor="phrase-search" className="sr-only">Search phrases</label>
        <input id="phrase-search" value={query} onChange={e => setQuery(e.target.value)}
               placeholder="Search phrases…" className="w-full bg-transparent outline-none"/>
      </div>
      <div className="flex flex-wrap gap-2" role="group" aria-label="Phrase categories">
        <button className={category === "all" ? "primary-action" : "secondary-action"}
                aria-pressed={category === "all"} onClick={() => setCategory("all")}>
          All phrases
        </button>
        {categories.map(value => (
          <button key={value}
                  className={category === value ? "primary-action" : "secondary-action"}
                  aria-pressed={category === value} onClick={() => setCategory(value)}>
            {CATEGORY_LABEL[value]}
          </button>
        ))}
      </div>
      <p className="text-label-md text-secondary" role="status">
        {phrases.length} {phrases.length === 1 ? "phrase" : "phrases"}
        {ready ? "" : " · loading translations"}
      </p>
    </section>

    {status && <div role="status" className="content-card text-primary flex flex-wrap items-center justify-between gap-3"><span>{status}</span><a href="#bridge" className="secondary-action">View message</a></div>}

    {phrases.length ? (
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
        {phrases.map(phrase => {
          const translated = phraseText(phrase, selectedLang);
          return (
            <article key={phrase.id} className={`content-card flex flex-col gap-3 ${phrase.urgent ? "border-primary" : ""}`}>
              <span className="eyebrow-label">{CATEGORY_LABEL[phrase.category]}</span>
              <h2 className="text-headline-md font-semibold">{phrase.en}</h2>
              {translated !== phrase.en && <p className="text-body-lg text-secondary">{translated}</p>}
              <button className="secondary-action mt-auto" onClick={() => send(phrase)}>
                {sent === phrase.id ? <Check size={18}/> : <Send size={18}/>}
                {sent === phrase.id ? "Send again" : `Send to ${domain.roles.client.lower}`}
              </button>
            </article>
          );
        })}
      </div>
    ) : (
      <section className="content-card empty-state">
        <MessageSquare size={32} className="mx-auto mb-4"/>
        <h2 className="text-headline-md font-semibold">No matching phrases</h2>
        <p className="mt-2">Try a different search or category.</p>
        <button className="secondary-action mt-4" onClick={() => {setQuery(""); setCategory("all");}}>Clear filters</button>
      </section>
    )}
  </div></main>;
}
