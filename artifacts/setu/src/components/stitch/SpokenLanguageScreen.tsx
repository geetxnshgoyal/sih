import { useEffect, useState } from 'react';
import { ArrowRight, Check, Languages } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { LANGUAGES, hasVoiceFor, refreshVoices } from '../../lib/speech';

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
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div><span className="eyebrow-label">Setup · Step 1</span><h1 className="text-display-md font-bold mt-2">Choose a spoken language</h1><p className="text-secondary mt-3">Use the language that feels most comfortable for your visit.</p></div>
    <fieldset><legend className="sr-only">Spoken language</legend><div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">{LANGUAGES.map(language => <label key={language.code} className={`content-card cursor-pointer flex items-center gap-4 ${selectedLang === language.code ? 'border-primary bg-surface-container-low' : ''}`}><input className="accent-primary w-5 h-5 shrink-0" type="radio" name="spoken-language" value={language.code} checked={selectedLang === language.code} onChange={() => setSelectedLang(language.code)}/><span className="flex-1"><span className="text-headline-md font-medium">{language.label}</span>{!hasVoiceFor(language.code) && <span className="block text-label-md text-secondary mt-1">No {language.label.split(' · ').pop()} voice on this device: spoken in the nearest voice. Text is exact.</span>}</span>{selectedLang === language.code && <Check size={20} className="text-primary shrink-0"/>}</label>)}</div></fieldset>
    <div className="content-card flex gap-4 items-start"><Languages size={24} className="text-primary shrink-0"/><p className="text-secondary">This language is used for speech input and playback. Which voices exist is decided by your device, not by Setu, and a language with no voice is read in the nearest one that shares its script. The written text and the phrase board are exact either way.</p></div>
    <div className="flex flex-wrap gap-3"><a href="#devices" className="primary-action">Continue to device check<ArrowRight size={18}/></a><a href="#bridge" className="secondary-action">Return to the counter</a></div>
  </div></main>;
}
