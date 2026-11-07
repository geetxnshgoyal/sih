import { useEffect, useMemo, useState } from "react";
import { Search, Send, Check, MessageSquare, X } from "lucide-react";
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
 * It reads STAFF_PHRASES_FOR[domain] and the same committed NLLB table the
 * client board uses, so all six languages work and the phrases match the
 * counter. Categories are derived from the data rather than hardcoded, which
 * is what stops "medication" appearing in a shop.
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

  return <main className="page"><div className="page-inner">
    <div className="page-head">
      <div>
        <span className="eyebrow indigo">{domain.station}</span>
        <h1>{domain.label} phrases</h1>
        <p className="lede">
          One tap sends a sentence to the {domain.roles.client.lower}'s screen: as text in their language, read aloud, and as signs where a recording exists.
        </p>
      </div>
      <div className="page-head-actions"><a href="#bridge" className="btn btn-secondary">Return to the counter</a></div>
    </div>

    <section className="card card-pad" style={{ display: 'flex', flexDirection: 'column', gap: 14 }} aria-label="Search and filter phrases">
      <div className="search">
        <Search size={20} aria-hidden="true" />
        <label htmlFor="phrase-search" className="sr-only">Search phrases</label>
        <input id="phrase-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search phrases…" />
        {query && <button className="btn btn-ghost btn-sm" onClick={() => setQuery('')} aria-label="Clear search"><X size={16} aria-hidden="true" /></button>}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <div className="chip-row" role="group" aria-label="Phrase categories">
          <button className="chip" aria-pressed={category === "all"} onClick={() => setCategory("all")}>All phrases</button>
          {categories.map(value => (
            <button key={value} className="chip" aria-pressed={category === value} onClick={() => setCategory(value)}>{CATEGORY_LABEL[value]}</button>
          ))}
        </div>
        <p className="card-note" role="status" style={{ fontWeight: 700 }}>
          {phrases.length} {phrases.length === 1 ? "phrase" : "phrases"}{ready ? "" : " · loading translations"}
        </p>
      </div>
    </section>

    {status && <div role="status" className="notice notice-ok pop" style={{ justifyContent: 'space-between', alignItems: 'center' }}><span style={{ display: 'flex', gap: 10, alignItems: 'center' }}><Check size={18} aria-hidden="true" />{status}</span><a href="#bridge" className="btn btn-secondary btn-sm">View on the counter</a></div>}

    {phrases.length ? (
      <div className="phrase-grid stagger">
        {phrases.map(phrase => {
          const translated = phraseText(phrase, selectedLang);
          const isSent = sent === phrase.id;
          return (
            <article key={phrase.id} className={`phrase-card${phrase.urgent ? " urgent" : ""}`}>
              <span className="phrase-cat">{CATEGORY_LABEL[phrase.category]}{phrase.urgent ? ' · urgent' : ''}</span>
              <h2>{phrase.en}</h2>
              {translated !== phrase.en && <p className="phrase-tr" lang={selectedLang}>{translated}</p>}
              <button className={`btn ${isSent ? 'btn-secondary is-sent' : 'btn-primary'}`} onClick={() => send(phrase)}>
                {isSent ? <Check size={18} aria-hidden="true" /> : <Send size={18} aria-hidden="true" />}
                {isSent ? "Sent · send again" : `Send to ${domain.roles.client.lower}`}
              </button>
            </article>
          );
        })}
      </div>
    ) : (
      <section className="card"><div className="empty">
        <span className="empty-glyph"><MessageSquare size={30} aria-hidden="true" /></span>
        <h2>No matching phrases</h2>
        <p>Try a different word or category.</p>
        <button className="btn btn-secondary" onClick={() => { setQuery(""); setCategory("all"); }}>Clear filters</button>
      </div></section>
    )}
  </div></main>;
}
