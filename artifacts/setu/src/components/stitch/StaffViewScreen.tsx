import { useState, useEffect, useRef } from 'react';
import { Mic, Square, Send, Volume2, RotateCcw, MessageSquare, Hand, Turtle, Check, AlertCircle, Info } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import SignPlayer from '../SignPlayer';
import { speak, LANGUAGES } from '../../lib/speech';
import { createRecogniser, textToGlosses } from '../../lib/reverse';
import { useSignLibrary } from '../../hooks/useSignLibrary';
import { useDomain } from '../../lib/useDomain';

export function StaffViewScreen() {
  const { selectedLang, activeProjection, projectToClient, isSlowMode, setIsSlowMode } = useSession();
  const domain = useDomain();
  const [input, setInput] = useState('');
  const { library, getClip, error: libraryError } = useSignLibrary();
  const [listening, setListening] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState({ key: '', index: 0 });
  const [replay, setReplay] = useState(0);
  const [justSent, setJustSent] = useState(false);
  const recognition = useRef<SpeechRecognition | null>(null);
  const text = activeProjection?.textEn || activeProjection?.text || '';
  const { matched, skipped } = textToGlosses(text, library);
  const sequenceKey = matched.join('|');
  const playbackKey = `${activeProjection?.timestamp ?? 0}|${replay}|${sequenceKey}`;
  const index = progress.key === playbackKey ? Math.min(progress.index, Math.max(0, matched.length - 1)) : 0;
  const current = matched[index];
  const clip = current ? library[current] : null;
  const frames = clip?.body ?? null;
  const langLabel = LANGUAGES.find(l => l.code === selectedLang)?.label ?? selectedLang;

  useEffect(() => { if (current && !frames?.length) void getClip(current); }, [current, frames, getClip]);
  useEffect(() => () => { const rec = recognition.current; if (rec) { rec.onresult = null; rec.onerror = null; rec.onend = null; rec.abort(); } }, [selectedLang]);
  useEffect(() => { if (!justSent) return; const t = window.setTimeout(() => setJustSent(false), 1400); return () => clearTimeout(t); }, [justSent]);

  function sendText(message: string) {
    const value = message.trim(); if (!value) return;
    projectToClient({ text: value, textEn: value }); setInput(''); setError(''); setJustSent(true);
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

  const showFullText = text && (matched.length !== 1 || skipped.length > 0 || current?.localeCompare(text, undefined, { sensitivity: 'base' }) !== 0);

  return <main className="page"><div className="page-inner">
    <div className="page-head">
      <div>
        <span className="eyebrow indigo">{domain.station} · {domain.roles.staff.label} view</span>
        <h1>Speak. The {domain.roles.client.lower} sees it in signs.</h1>
        <p className="lede">Whatever you say or type appears on the {domain.roles.client.lower}'s screen as text and sign playback.</p>
      </div>
      <div className="page-head-actions"><a href="#phrases" className="btn btn-secondary"><MessageSquare size={18} aria-hidden="true" />Ready phrases</a><a href="#transcript" className="btn btn-ghost">Transcript</a></div>
    </div>

    <div className="counter-grid">
      <section className="card side-staff" aria-labelledby="playback-heading">
        <div className="card-h"><span><Hand size={15} aria-hidden="true" /> Sign playback</span><span>{matched.length ? `${index + 1} of ${matched.length} signs` : 'Waiting for a message'}</span></div>
        <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <h2 id="playback-heading" className="sr-only">Sign playback</h2>
          <div className="sign-stage">
            {matched.length > 1 && <div className="sign-progress" aria-hidden="true">{matched.map((g, i) => <span key={`${g}-${i}`} className={i < index ? 'done' : i === index ? 'now' : ''} />)}</div>}
            <div className="sign-stage-inner">
              {frames?.length ? <>
                <SignPlayer key={`${playbackKey}-${index}`} frames={frames} faceFrames={clip?.face} fps={isSlowMode ? Math.min(8, clip?.fps ?? 14) : clip?.fps ?? 14} onComplete={() => setProgress({ key: playbackKey, index: Math.min(index + 1, matched.length - 1) })} />
                <span className="sign-word pop" key={current}>{current}</span>
              </> : current ? <div className="empty" role="status"><span className="spinner" style={{ color: 'var(--indigo)' }} aria-hidden="true" /><p>Loading the sign for {current}…</p></div>
                : <div className="empty"><span className="empty-glyph"><Hand size={30} aria-hidden="true" /></span><h3>{text ? 'No recorded sign for this message' : 'Nothing playing yet'}</h3><p>{text ? `The ${domain.roles.client.lower} still sees the full text.` : 'Speak or type a message and the matching signs play here.'}</p></div>}
            </div>
          </div>
          {matched.length > 1 && <div className="sign-chips" aria-label="Signs in this message">{matched.map((g, i) => <span key={`${g}-${i}`} className={i === index ? 'on' : ''}>{g}</span>)}</div>}
          {showFullText && <p className="said">{activeProjection?.text}</p>}
          {(libraryError || (skipped.length > 0 && text)) && <p role="status" className="notice notice-info"><Info size={18} aria-hidden="true" /><span>{libraryError || `No sign recording for: ${skipped.join(', ')}. The complete message is shown as text.`}</span></p>}
          <div className="chip-row">
            <button className="btn btn-secondary" disabled={!frames} onClick={() => setReplay(r => r + 1)}><RotateCcw size={18} aria-hidden="true" />Replay signs</button>
            <button className="btn btn-secondary" aria-pressed={isSlowMode} onClick={() => setIsSlowMode(!isSlowMode)}><Turtle size={18} aria-hidden="true" />{isSlowMode ? 'Slow speed on' : 'Slow speed'}</button>
            <button className="btn btn-ghost" disabled={!activeProjection?.text} onClick={() => speak(activeProjection!.text, selectedLang)}><Volume2 size={18} aria-hidden="true" />Read aloud</button>
          </div>
        </div>
      </section>

      <section className="card side-staff" aria-labelledby="compose-heading">
        <div className="card-h"><span><Mic size={15} aria-hidden="true" /> Your message</span><span>{langLabel}</span></div>
        <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <h2 id="compose-heading" className="sr-only">Compose a message</h2>
          <button type="button" className={`mic-button${listening ? ' on' : ''}`} onClick={toggleListening} aria-pressed={listening}>
            <span className="mic-orb" aria-hidden="true">{listening ? <Square size={22} /> : <Mic size={26} />}</span>
            <span><strong>{listening ? 'Listening… tap to stop' : 'Tap and speak'}</strong><small>{listening ? 'Sent automatically when you pause.' : `Spoken ${langLabel.split(' · ').pop()} becomes text and signs.`}</small></span>
          </button>
          <div className="or-rule">or type</div>
          <form className="field" onSubmit={e => { e.preventDefault(); sendText(input); }}>
            <label htmlFor="staff-message" className="sr-only">Type a message</label>
            <textarea id="staff-message" value={input} onChange={e => setInput(e.target.value)} placeholder="Type a question or instruction…" rows={4} className="textarea" />
            <button className={`btn btn-primary btn-lg${justSent ? ' is-sent' : ''}`} disabled={!input.trim() && !justSent} type="submit">{justSent ? <Check size={20} aria-hidden="true" /> : <Send size={20} aria-hidden="true" />}{justSent ? 'Sent' : `Send to ${domain.roles.client.lower}`}</button>
          </form>
          {error && <p role="alert" className="notice notice-err"><AlertCircle size={18} aria-hidden="true" /><span>{error}</span></p>}
          <p role="status" className="sr-only">{justSent ? 'Message sent' : ''}</p>
        </div>
      </section>
    </div>
  </div></main>;
}
