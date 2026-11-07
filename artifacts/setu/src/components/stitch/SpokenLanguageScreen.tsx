import { useEffect, useState } from 'react';
import { ArrowRight, Check, Languages } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { LANGUAGES, hasVoiceFor, refreshVoices, type LangCode } from '../../lib/speech';

/** One native glyph per language, so the card is recognisable without reading the label. */
const GLYPH: Record<LangCode, string> = { 'hi-IN': 'अ', 'ta-IN': 'அ', 'te-IN': 'అ', 'bn-IN': 'অ', 'mr-IN': 'म', 'en-IN': 'A' };

export function SpokenLanguageScreen() {
  const { selectedLang, setSelectedLang } = useSession();
  /**
   * Which languages this device can actually speak.
   *
   * Recomputed on "voiceschanged" because the list is empty on first paint in
   * Chrome and Safari; without the re-render every language would be labelled
   * as having no voice for the first second, which is worse than saying
   * nothing. Marathi is the one that usually really has none.
   */
  const [, bump] = useState(0);
  useEffect(() => {
    const onChange = () => { refreshVoices(); bump(n => n + 1); };
    window.speechSynthesis?.addEventListener('voiceschanged', onChange);
    return () => window.speechSynthesis?.removeEventListener('voiceschanged', onChange);
  }, []);
  return <main className="page"><div className="page-inner">
    <div className="page-head">
      <div><span className="eyebrow indigo">Setup · Step 1 of 2</span><h1>Which language is spoken here?</h1><p className="lede">Used for speech input and for reading messages aloud. Written text and the phrase board are exact in every one of these.</p></div>
    </div>
    <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
      <legend className="sr-only">Spoken language</legend>
      <div className="lang-grid stagger">
        {LANGUAGES.map(language => {
          const on = selectedLang === language.code;
          const [native, english] = language.label.split(' · ');
          return <label key={language.code} className={`lang-card${on ? ' on' : ''}`}>
            <input type="radio" name="spoken-language" value={language.code} checked={on} onChange={() => setSelectedLang(language.code)} />
            <span className="lang-glyph" aria-hidden="true">{GLYPH[language.code]}</span>
            <span style={{ minWidth: 0 }}>
              <span className="lang-name">{native}</span>
              <span className="lang-meta">{english ? english : 'Indian English'}{!hasVoiceFor(language.code) && ' · no voice on this device, read in the nearest one'}</span>
            </span>
            {on && <span className="lang-check" aria-hidden="true"><Check size={18} /></span>}
          </label>;
        })}
      </div>
    </fieldset>
    <div className="notice notice-info"><Languages size={20} aria-hidden="true" /><p>Which voices exist is decided by your device, not by Setu. A language with no installed voice is read in the nearest one that shares its script. Text is never affected.</p></div>
    <div className="chip-row"><a href="#devices" className="btn btn-primary btn-lg">Continue to device check<ArrowRight size={20} aria-hidden="true" /></a><a href="#bridge" className="btn btn-secondary btn-lg">Go to the counter</a></div>
  </div></main>;
}
