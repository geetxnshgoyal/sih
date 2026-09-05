import { useEffect, useMemo, useState } from "react";
import {
  CATEGORY_LABEL, HEALTH_ORDER, HEALTH_PHRASES, TRAVEL_ORDER, TRAVEL_PHRASES,
  type Phrase, type PhraseCategory,
} from "../lib/phrasebook";
import { loadPhrasebook, phraseText } from "../lib/phrasebookTable";
import type { LangCode } from "../lib/speech";
import type { DomainId } from "../lib/domains";

/**
 * The phrase board.
 *
 * This is the path that always works. Recognition is 40.4% correct on a signer
 * it has not seen; a tapped phrase is 100%, every time, offline. So this is the
 * primary surface and recognition is the shortcut, not the reverse.
 *
 * Layout follows a triage conversation rather than the alphabet, Emergency and
 * About me first, because "I am Deaf" and "I cannot breathe" are the two things
 * that must never be more than one tap away. Search exists because 97 phrases
 * is more than anyone will scroll in pain.
 */
export default function PhraseBoard({
  domain,
  lang,
  onSay,
}: {
  domain: DomainId;
  lang: LangCode;
  onSay: (text: string, phrase: Phrase) => void;
}) {
  const [ready, setReady] = useState(false);
  const [query, setQuery] = useState("");
  const [openCat, setOpenCat] = useState<PhraseCategory | null>(null);

  useEffect(() => { void loadPhrasebook().then(setReady); }, []);

  const phrases = domain === "health" ? HEALTH_PHRASES : TRAVEL_PHRASES;
  const order = domain === "health" ? HEALTH_ORDER : TRAVEL_ORDER;

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    // Search the English and the button label. Not the translation: someone
    // typing here is reading the English side of the screen.
    return phrases.filter(
      (p) => p.en.toLowerCase().includes(q) || (p.short ?? "").toLowerCase().includes(q)
    );
  }, [query, phrases]);

  const byCat = useMemo(() => {
    const m = new Map<PhraseCategory, Phrase[]>();
    for (const p of phrases) {
      if (!m.has(p.category)) m.set(p.category, []);
      m.get(p.category)!.push(p);
    }
    return m;
  }, [phrases]);

  const say = (p: Phrase) => onSay(phraseText(p, lang), p);

  const Button = ({ p }: { p: Phrase }) => (
    <button
      className={p.urgent ? "pb-item urgent" : "pb-item"}
      onClick={() => say(p)}
      title={p.en}
    >
      {p.short ?? p.en}
    </button>
  );

  return (
    <section className="card phraseboard">
      <div className="card-h">
        <span>Tap to speak</span>
        <span className="mono">
          {phrases.length} phrases{ready ? "" : " · loading"}
        </span>
      </div>

      <div className="pb-search">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search: pain, water, doctor…"
          aria-label="Search phrases"
        />
        {query && (
          <button className="pb-clear" onClick={() => setQuery("")} aria-label="Clear search">
            clear
          </button>
        )}
      </div>

      {matches ? (
        <div className="pb-grid">
          {matches.length === 0 ? (
            <p className="note">No phrase matches “{query}”.</p>
          ) : (
            matches.map((p) => <Button key={p.id} p={p} />)
          )}
        </div>
      ) : (
        order.map((cat) => {
          const items = byCat.get(cat) ?? [];
          if (items.length === 0) return null;
          // Emergency is never collapsed. Everything else opens on demand so
          // the board is scannable rather than a wall of 97 buttons.
          const open = cat === "emergency" || openCat === cat;
          return (
            <div key={cat} className={`pb-cat ${cat === "emergency" ? "always" : ""}`}>
              <button
                className="pb-cat-h"
                onClick={() => setOpenCat(open && cat !== "emergency" ? null : cat)}
                aria-expanded={open}
              >
                <span>{CATEGORY_LABEL[cat]}</span>
                <span className="mono">{items.length}</span>
              </button>
              {open && (
                <div className="pb-grid">
                  {items.map((p) => <Button key={p.id} p={p} />)}
                </div>
              )}
            </div>
          );
        })
      )}
    </section>
  );
}
