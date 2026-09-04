import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import SignBridge from "./components/SignBridge";
import HearingSide from "./components/HearingSide";
import Recorder from "./components/Recorder";
import PhraseBoard from "./components/PhraseBoard";
import { LANGUAGES, speak, type LangCode } from "./lib/speech";
import { loadGlossTable, sourceLabel, type TranslationSource } from "./lib/glossTranslate";
import {
  DOMAIN_LIST,
  DOMAINS,
  getDomain,
  getServerDomain,
  saveDomain,
  subscribeDomain,
  type DomainId,
} from "./lib/domains";
import "./App.css";

type Mode = "sign" | "conversation" | "record";

export default function App() {
  const [mode, setMode] = useState<Mode>("sign");
  const [lang, setLang] = useState<LangCode>("hi-IN");
  // Subscribed rather than read in an effect: localStorage is unavailable
  // during prerender, so the value starts at the default and is adopted on the
  // client without a cascading re-render. See lib/domains.ts.
  const domainId = useSyncExternalStore(subscribeDomain, getDomain, getServerDomain);
  const [spoken, setSpoken] = useState<{ text: string; source: TranslationSource } | null>(null);

  useEffect(() => {
    void loadGlossTable();
  }, []);

  const domain = DOMAINS[domainId];

  const pickDomain = useCallback((id: DomainId) => saveDomain(id), []);

  /** Speak a tapped phrase. No recognition, no threshold, no guessing. */
  const say = useCallback(
    (text: string) => {
      speak(text, lang);
      setSpoken({ text, source: "phrasebook" });
    },
    [lang]
  );

  return (
    <div className="app">
      <nav className="modes">
        <button className={mode === "sign" ? "on" : ""} onClick={() => setMode("sign")}>
          Signs to speech
        </button>
        <button
          className={mode === "conversation" ? "on" : ""}
          onClick={() => setMode("conversation")}
        >
          Conversation
        </button>
        <button className={mode === "record" ? "on" : ""} onClick={() => setMode("record")}>
          Record signs
        </button>

        <div className="spacer" />

        {/* The recognition stack is identical across settings; this only changes
            which phrases lead and how spoken words map. See lib/domains.ts. */}
        <div className="domainpick" role="group" aria-label="Deployment setting">
          {DOMAIN_LIST.map((d) => (
            <button
              key={d.id}
              className={d.id === domainId ? "on" : ""}
              onClick={() => pickDomain(d.id)}
              title={d.tagline}
            >
              {d.label}
            </button>
          ))}
        </div>

        <select
          className="langpick"
          value={lang}
          onChange={(e) => setLang(e.target.value as LangCode)}
        >
          {LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>{l.label}</option>
          ))}
        </select>
      </nav>

      {mode === "record" ? (
        <Recorder />
      ) : mode === "sign" ? (
        <>
          <SignBridge lang={lang} onLang={setLang} />
          <PhraseBoard domain={domainId} lang={lang} onSay={say} />
          {spoken && (
            <p className="spoken-note">
              spoke: “{spoken.text}”
              <span className={spoken.source === "gloss-order" ? "prov warn" : "prov"}>
                {sourceLabel(spoken.source)}
              </span>
            </p>
          )}
        </>
      ) : (
        <div className="conversation">
          <div className="half deaf">
            <SignBridge lang={lang} onLang={setLang} compact />
            <PhraseBoard domain={domainId} lang={lang} onSay={say} />
            {spoken && (
            <p className="spoken-note">
              spoke: “{spoken.text}”
              <span className={spoken.source === "gloss-order" ? "prov warn" : "prov"}>
                {sourceLabel(spoken.source)}
              </span>
            </p>
          )}
          </div>
          <div className="divider"><span>turn</span></div>
          <div className="half hearing-half">
            <HearingSide lang={lang} synonyms={domain.synonyms} />
          </div>
        </div>
      )}
    </div>
  );
}
