import { useState } from "react";
import { SERVICE_PHRASES, type Phrase } from "../lib/phrasebook";
import { phraseText } from "../lib/phrasebookTable";
import type { LangCode } from "../lib/speech";
import type { DomainId } from "../lib/domains";
export default function PhraseBoard({
  lang,
  onSay,
}: {
  domain: DomainId;
  lang: LangCode;
  onSay: (text: string, phrase: Phrase) => void;
}) {
  const [query, setQuery] = useState("");
  const filtered = SERVICE_PHRASES.filter((phrase) =>
    phrase.en.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <section className="panel library">
      <h2>Service phrase board</h2>
      <label>
        Search phrases
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="quick-phrases">
        {filtered.map((phrase) => (
          <button
            className="phrase"
            key={phrase.id}
            onClick={() => onSay(phraseText(phrase, lang), phrase)}
          >
            {phraseText(phrase, lang)}
          </button>
        ))}
      </div>
      {!filtered.length && (
        <p>No matching phrases. Type your own message instead.</p>
      )}
    </section>
  );
}
