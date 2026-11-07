import { lazy, Suspense, useEffect, useState } from 'react';
import { Camera, Send, MessageSquare, Volume2, X, Check, Keyboard, ShieldCheck } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { useDomain } from '../../lib/useDomain';
import { speak } from '../../lib/speech';
import PhraseBoard from '../PhraseBoard';
import QuickPhrases from '../QuickPhrases';
import { translateGlosses } from '../../lib/glossTranslate';
const SignBridge = lazy(() => import('../SignBridge'));

export function ClientViewScreen() {
  const { selectedLang, setSelectedLang, activeProjection, addTranscriptItem } = useSession();
  const domain = useDomain();
  const [showCamera, setShowCamera] = useState(false);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('');
  const [justSent, setJustSent] = useState(false);
  useEffect(() => { if (!justSent) return; const t = window.setTimeout(() => setJustSent(false), 1400); return () => clearTimeout(t); }, [justSent]);
  const send = (text: string) => { if (!text.trim()) return; addTranscriptItem({ speaker: 'client', speakerName: domain.roles.client.label, text: text.trim(), category: 'client' }); setInput(''); setStatus(`Sent: ${text}`); setJustSent(true); };

  return <main className="page"><div className="page-inner">
    <div className="page-head">
      <div>
        <span className="eyebrow">{domain.station} · {domain.roles.client.label} view</span>
        <h1>Your side of the counter.</h1>
        <p className="lede">Read what the {domain.roles.staff.lower} said. Answer by signing, tapping a phrase, or typing. All of it is spoken aloud for them.</p>
      </div>
      <div className="page-head-actions"><a href="#transcript" className="btn btn-secondary"><MessageSquare size={18} aria-hidden="true" />Whole conversation</a></div>
    </div>

    <section className="board rise" aria-labelledby="board-heading" aria-live="polite">
      <div className="board-k"><span id="board-heading">The {domain.roles.staff.lower} says</span><span className={`status${activeProjection ? ' live' : ''}`}><i aria-hidden="true" />{activeProjection ? 'Latest message' : 'Waiting'}</span></div>
      <p className={`board-text${activeProjection ? '' : ' idle'}`} key={activeProjection?.timestamp ?? 0}>{activeProjection?.text || `The next message from the ${domain.roles.staff.lower} will appear here, in large text.`}</p>
      {activeProjection?.textEn && activeProjection.textEn !== activeProjection.text && <p className="board-en">{activeProjection.textEn}</p>}
      <div className="board-actions">
        <button className="btn btn-primary" disabled={!activeProjection} onClick={() => activeProjection && speak(activeProjection.text, selectedLang)}><Volume2 size={18} aria-hidden="true" />Read aloud</button>
      </div>
    </section>

    <div className="reply-grid">
      <section className="card side-client" aria-labelledby="camera-heading">
        <div className="card-h"><span><Camera size={15} aria-hidden="true" /> Sign with your camera</span><span><ShieldCheck size={14} aria-hidden="true" /> On this device only</span></div>
        <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <h2 id="camera-heading" className="card-title">Sign, then choose the right word</h2>
          <p className="card-note">Keep your face and both hands in the frame. Nothing is uploaded. You confirm each sign before it is sent.</p>
          {showCamera ? <>
            <Suspense fallback={<div className="empty" role="status"><span className="spinner" style={{ color: 'var(--teal)' }} aria-hidden="true" /><p>Loading sign recognition…</p></div>}>
              <SignBridge compact confirmBeforeSend lang={selectedLang} onLang={setSelectedLang} onRecognized={text => send(text)} />
            </Suspense>
            <button className="btn btn-ghost" onClick={() => setShowCamera(false)}><X size={18} aria-hidden="true" />Close camera</button>
          </> : <button className="btn btn-teal btn-lg" onClick={() => setShowCamera(true)}><Camera size={20} aria-hidden="true" />Open sign recognition</button>}
        </div>
      </section>

      <section className="card side-client" aria-labelledby="type-heading">
        <div className="card-h"><span><Keyboard size={15} aria-hidden="true" /> Write a reply</span></div>
        <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <h2 id="type-heading" className="card-title">Type it instead</h2>
          <form className="field" onSubmit={e => { e.preventDefault(); send(input); }}>
            <label htmlFor="client-message" className="sr-only">Your reply</label>
            <textarea id="client-message" rows={4} className="textarea" placeholder="Type your reply…" value={input} onChange={e => setInput(e.target.value)} />
            <button className={`btn btn-teal btn-lg${justSent ? ' is-sent' : ''}`} disabled={!input.trim() && !justSent}>{justSent ? <Check size={20} aria-hidden="true" /> : <Send size={20} aria-hidden="true" />}{justSent ? 'Sent' : 'Send reply'}</button>
          </form>
          <p role="status" className="card-note" style={{ color: 'var(--teal-deep)', fontWeight: 700 }}>{status}</p>
        </div>
      </section>
    </div>

    {/* Tap-to-say, above the full board.
        These are gloss SEQUENCES, so they go through translateGlosses and the
        same _utterances.json the recognition path uses: what the staff member
        hears from a tap is exactly what they would hear from a sign. The board
        below is the exhaustive, searchable version; this is the six things a
        person reaches for without reading. */}
    <QuickPhrases domain={domain} onSay={q => {
      const t = translateGlosses(q.glosses, selectedLang);
      const text = t.text || q.caption;
      speak(text, selectedLang);
      send(text);
    }} />

    {/* The phrase board, not a handful of hardcoded strings.
        This is the path that always works. Recognition is about 74% correct on
        a signer it has never seen; a tapped phrase is right every time, offline,
        in all six languages, because the translations are generated once with
        NLLB-200 and committed (lib/phrasebookTable.ts). */}
    <PhraseBoard domain={domain.id} lang={selectedLang} onSay={text => { speak(text, selectedLang); send(text); }} />
  </div></main>;
}
