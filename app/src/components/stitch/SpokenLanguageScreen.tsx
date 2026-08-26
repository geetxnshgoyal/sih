import { ArrowRight, Check, Languages } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { LANGUAGES } from '../../lib/speech';
export function SpokenLanguageScreen() {
  const { selectedLang, setSelectedLang } = useSession();
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div><span className="eyebrow-label">Consultation setup · Step 1</span><h1 className="text-display-md font-bold mt-2">Choose a spoken language</h1><p className="text-secondary mt-3">Use the language that feels most comfortable for your visit.</p></div>
    <fieldset><legend className="sr-only">Spoken language</legend><div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">{LANGUAGES.map(language => <label key={language.code} className={`content-card cursor-pointer flex items-center gap-4 ${selectedLang === language.code ? 'border-primary bg-surface-container-low' : ''}`}><input className="accent-primary w-5 h-5 shrink-0" type="radio" name="spoken-language" value={language.code} checked={selectedLang === language.code} onChange={() => setSelectedLang(language.code)}/><span className="flex-1 text-headline-md font-medium">{language.label}</span>{selectedLang === language.code && <Check size={20} className="text-primary"/>}</label>)}</div></fieldset>
    <div className="content-card flex gap-4 items-start"><Languages size={24} className="text-primary shrink-0"/><p className="text-secondary">This language is used for speech input and playback. Voice availability depends on your browser. Typed messages are always available.</p></div>
    <div className="flex flex-wrap gap-3"><a href="#devices" className="primary-action">Continue to device check<ArrowRight size={18}/></a><a href="#bridge" className="secondary-action">Return to consultation</a></div>
  </div></main>;
}
