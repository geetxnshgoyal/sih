import { lazy, Suspense, useState } from 'react';
import { Camera, Send, MessageSquare, Volume2 } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { speak } from '../../lib/speech';
import PhraseBoard from '../PhraseBoard';
const SignBridge = lazy(() => import('../SignBridge'));
export function PatientViewScreen() {
  const { selectedLang, setSelectedLang, activeProjection, addTranscriptItem } = useSession();
  const [showCamera, setShowCamera] = useState(false);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('');
  const send = (text: string) => { if (!text.trim()) return; addTranscriptItem({speaker: 'patient', speakerName: 'Patient', text: text.trim(), category: 'patient'}); setInput(''); setStatus(`Sent: ${text}`); };
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div className="flex items-center justify-between"><div><span className="eyebrow-label">Live consultation · Patient view</span><h1 className="text-display-md font-bold mt-2">Your side of the conversation</h1><p className="text-secondary mt-2">Read the doctor's message and respond in the way that works for you.</p></div><a href="#transcript" className="secondary-action"><MessageSquare size={18}/>View conversation</a></div>
    <section className="content-card"><span className="eyebrow-label">Doctor's latest message</span><p className="text-headline-lg mt-4 leading-relaxed">{activeProjection?.text || 'Your doctor’s next message will appear here.'}</p>{activeProjection?.textEn && activeProjection.textEn !== activeProjection.text && <p className="text-body-lg text-secondary mt-3">{activeProjection.textEn}</p>}{activeProjection && <button className="secondary-action mt-4" onClick={() => speak(activeProjection.text, selectedLang)}><Volume2 size={18}/>Read aloud</button>}</section>
    <div className="grid lg:grid-cols-2 gap-6 items-start"><section className="content-card"><h2 className="text-headline-md font-semibold mb-3">Sign with your camera</h2><p className="text-secondary mb-5">Keep your face and both hands in the frame. Video is processed on this device and never uploaded.</p>{showCamera ? <><button className="secondary-action mb-4" onClick={() => setShowCamera(false)}>Close camera panel</button><Suspense fallback={<p role="status">Loading sign recognition…</p>}><SignBridge compact lang={selectedLang} onLang={setSelectedLang} onRecognized={text => send(text)} /></Suspense></> : <button className="primary-action" onClick={() => setShowCamera(true)}><Camera size={18}/>Open sign recognition</button>}</section>
    <section className="content-card flex flex-col gap-4"><h2 className="text-headline-md font-semibold">Write a response</h2><form className="flex flex-col gap-3" onSubmit={e => {e.preventDefault(); send(input);}}><label htmlFor="patient-message" className="sr-only">Your response</label><textarea id="patient-message" rows={3} className="w-full resize-y border border-outline-variant rounded-xl p-4 bg-surface-container-low" placeholder="Type your response…" value={input} onChange={e => setInput(e.target.value)}/><button className="primary-action" disabled={!input.trim()}><Send size={18}/>Send response</button></form><p role="status" className="text-primary text-label-md">{status}</p></section></div>
    {/* The phrase board, not a handful of hardcoded strings.
        This is the path that always works. Recognition is 13.9% correct on a
        signer it has never seen; a tapped phrase is right every time, offline,
        in all 11 languages, because the translations are generated once with
        NLLB-200 and committed (lib/phrasebookTable.ts). Ordering follows a
        triage conversation rather than the alphabet, so "I am Deaf" and "I
        cannot breathe" are never more than one tap away.
        It speaks aloud AND lands in the transcript: the hearing clinician needs
        to hear it, and the record needs to show it was said. */}
    <PhraseBoard domain="health" lang={selectedLang} onSay={text => { speak(text, selectedLang); send(text); }} />
  </div></main>;
}
