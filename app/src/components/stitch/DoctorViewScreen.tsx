import { useState, useEffect, useRef } from 'react';
import { Mic, Square, Send, Volume2, RotateCcw, MessageSquare, Hand } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import SignPlayer from '../SignPlayer';
import { speak } from '../../lib/speech';
import { createRecogniser, textToGlosses, type SignLibrary } from '../../lib/reverse';
import { asset } from '../../lib/assetUrl';

export function DoctorViewScreen() {
  const { selectedLang, activeProjection, projectToPatient, isSlowMode, setIsSlowMode } = useSession();
  const [input, setInput] = useState('');
  const [library, setLibrary] = useState<SignLibrary>({});
  const [libraryError, setLibraryError] = useState('');
  const [listening, setListening] = useState(false);
  const [error, setError] = useState('');
  const [index, setIndex] = useState(0);
  const [replay, setReplay] = useState(0);
  const recognition = useRef<SpeechRecognition | null>(null);
  const text = activeProjection?.textEn || activeProjection?.text || '';
  const { matched, skipped } = textToGlosses(text, library);
  const sequenceKey = matched.join('|');
  const current = matched[index];
  const frames = current ? library[current] : null;

  useEffect(() => {
    let cancelled = false;
    fetch(asset('/model/_signs.json')).then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(data => { if (!cancelled) setLibrary(data); }).catch(() => { if (!cancelled) setLibraryError('Sign playback is unavailable. Written messages still work.'); });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => { setIndex(0); }, [sequenceKey, activeProjection?.timestamp, replay]);
  useEffect(() => {
    if (!frames || index >= matched.length - 1) return;
    const timer = window.setTimeout(() => setIndex(i => i + 1), Math.max(1400, frames.length / (isSlowMode ? 10 : 14) * 1000));
    return () => window.clearTimeout(timer);
  }, [frames, index, matched.length, isSlowMode, replay]);
  useEffect(() => () => { const rec = recognition.current; if (rec) { rec.onresult = null; rec.onerror = null; rec.onend = null; rec.abort(); } }, [selectedLang]);

  function sendText(message: string) {
    const value = message.trim(); if (!value) return;
    projectToPatient({text: value, textEn: value}); setInput(''); setError('');
  }
  function toggleListening() {
    if (listening) { recognition.current?.stop(); setListening(false); return; }
    const rec = createRecogniser(selectedLang);
    if (!rec) { setError('Speech input is not supported in this browser. Type a message below.'); return; }
    recognition.current = rec;
    rec.onresult = e => {
      const value = Array.from(e.results).map(r => r[0].transcript).join(' '); setInput(value);
      if (e.results[e.results.length - 1].isFinal) { sendText(value); setListening(false); }
    };
    rec.onerror = () => { setListening(false); setError('Speech input could not start. Check microphone permissions or type a message.'); };
    rec.onend = () => setListening(false);
    try { rec.start(); setListening(true); setError(''); } catch { setError('Microphone is busy. Please try again.'); setListening(false); }
  }
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div className="flex items-center justify-between"><div><span className="eyebrow-label">Live consultation · Doctor view</span><h1 className="text-display-md font-bold mt-2">Speak. Type. Connect.</h1><p className="text-secondary mt-2">Your latest message appears in the patient view.</p></div><a href="#transcript" className="secondary-action"><MessageSquare size={18}/>View transcript</a></div>
    <div className="grid lg:grid-cols-12 gap-6 items-start">
      <section className="content-card lg:col-span-7"><div className="flex items-center justify-between mb-4"><h2 className="text-headline-md font-semibold">Sign playback</h2><span className="text-label-sm text-secondary">{matched.length ? `${index + 1} of ${matched.length} signs` : 'Waiting for a message'}</span></div>
        <div className="rounded-xl bg-surface-container-low min-h-[300px] flex flex-col items-center justify-center p-5">{frames ? <><SignPlayer key={`${activeProjection?.timestamp}-${replay}-${index}`} frames={frames} fps={isSlowMode ? 10 : 14}/><span className="text-headline-md text-primary mt-3 font-semibold">{current}</span></> : <div className="empty-state"><Hand size={40} className="mx-auto mb-4 text-primary"/><p>{text ? 'No matching sign recording for this message.' : 'Send a message to play available signs.'}</p></div>}</div>
        {text && <p className="text-body-xl mt-5 leading-relaxed">{activeProjection?.text}</p>}
        {(libraryError || (skipped.length > 0 && text)) && <p role="status" className="text-label-md text-secondary mt-3">{libraryError || 'Some words have no sign recording. The complete message is shown as text.'}</p>}
        <div className="flex flex-wrap gap-3 mt-5"><button className="secondary-action" disabled={!frames} onClick={() => {setIndex(0); setReplay(r => r + 1);}}><RotateCcw size={18}/>Replay signs</button><button className="secondary-action" aria-pressed={isSlowMode} onClick={() => setIsSlowMode(!isSlowMode)}>{isSlowMode ? 'Speed: slow' : 'Speed: normal'}</button></div>
      </section>
      <section className="content-card lg:col-span-5 flex flex-col gap-4"><h2 className="text-headline-md font-semibold">Your message</h2><p className="text-label-md text-secondary">Speech input: {selectedLang}</p><button className="secondary-action" onClick={toggleListening}>{listening ? <Square size={18}/> : <Mic size={18}/>} {listening ? 'Stop listening' : 'Start speaking'}</button>
        <form className="flex flex-col gap-3" onSubmit={e => {e.preventDefault(); sendText(input);}}><label htmlFor="doctor-message" className="text-label-md font-semibold">Or type a message</label><textarea id="doctor-message" value={input} onChange={e => setInput(e.target.value)} placeholder="Type a question or instruction…" rows={4} className="w-full resize-y border border-outline-variant rounded-xl p-4 bg-surface-container-low text-body-md"/><button className="primary-action" disabled={!input.trim()} type="submit"><Send size={18}/>Send to patient</button></form>
        {error && <p role="alert" className="text-error text-label-md">{error}</p>}
        <div className="border-t border-outline-variant/30 pt-4 flex flex-wrap gap-3"><button className="secondary-action" disabled={!activeProjection?.text} onClick={() => speak(activeProjection!.text, selectedLang)}><Volume2 size={18}/>Read aloud</button><a href="#phrases" className="secondary-action">Quick phrases</a></div>
      </section>
    </div>
  </div></main>;
}
