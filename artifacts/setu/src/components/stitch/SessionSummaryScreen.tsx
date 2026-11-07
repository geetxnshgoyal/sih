import { useState } from 'react';
import { Download, Printer, Plus, FileText, Check, AlertTriangle } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { useDomain } from '../../lib/useDomain';
import { LANGUAGES } from '../../lib/speech';

/** Sentence case for a domain noun that is stored lowercase. */
const cap = (w: string) => w.charAt(0).toUpperCase() + w.slice(1);
export function SessionSummaryScreen() {
  const { transcript, selectedLang, resetSession } = useSession();
  const domain = useDomain();
  const [confirmReset, setConfirmReset] = useState(false);
  const [status, setStatus] = useState('');
  const staffCount = transcript.filter(t => t.speaker === 'staff').length;
  const clientCount = transcript.filter(t => t.speaker === 'client').length;
  const langLabel = LANGUAGES.find(l => l.code === selectedLang)?.label ?? selectedLang;
  function exportNotes() {
    const data = { exportedAt: new Date().toISOString(), language: selectedLang, messages: transcript };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `setu-${domain.id}-${new Date().toISOString().slice(0, 10)}.json`; document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000); setStatus('Session notes exported.');
  }
  return <main className="page"><div className="page-inner">
    <div className="page-head"><div><span className="eyebrow indigo">Session review</span><h1>{cap(domain.session)} summary</h1><p className="lede">The record of this {domain.session}, in the order it happened. Save a copy before starting the next one.</p></div></div>
    <div className="stat-row stagger">
      <div className="stat"><b>{transcript.length}</b><span>{transcript.length === 1 ? 'message' : 'messages'}</span></div>
      <div className="stat"><b style={{ color: 'var(--indigo)' }}>{staffCount}</b><span>from the {domain.roles.staff.lower}</span></div>
      <div className="stat"><b style={{ color: 'var(--teal-deep)' }}>{clientCount}</b><span>from the {domain.roles.client.lower}</span></div>
    </div>
    <div className="summary-grid">
      <section className="card" aria-labelledby="notes-heading">
        <div className="card-h"><span><FileText size={15} aria-hidden="true" /> Conversation notes</span><span>{langLabel}</span></div>
        <div className="card-b">
          <h2 id="notes-heading" className="sr-only">Conversation notes</h2>
          {transcript.length === 0
            ? <div className="empty"><span className="empty-glyph"><FileText size={30} aria-hidden="true" /></span><h3>No messages yet</h3><p>Start a {domain.session} and every line from both sides lands here.</p><a href="#bridge" className="btn btn-primary no-print">Open the {domain.session}</a></div>
            : <ol className="summary-list" style={{ listStyle: 'none', margin: 0, padding: 0 }}>{transcript.map(item => <li key={item.id} className={`summary-item ${item.speaker}`}><div className="summary-who"><span>{item.speakerName}</span><time>{item.timestamp}</time></div><div><p className="summary-text">{item.text}</p>{item.textEn && item.textEn !== item.text && <p className="summary-en">{item.textEn}</p>}{item.medication && <p className="summary-en">{item.medication.instructions}</p>}</div></li>)}</ol>}
        </div>
      </section>
      <aside className="card no-print" aria-labelledby="save-heading">
        <div className="card-h"><span>Save this {domain.session}</span></div>
        <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <h2 id="save-heading" className="sr-only">Save this {domain.session}</h2>
          <button className="btn btn-primary btn-block" disabled={!transcript.length} onClick={exportNotes}><Download size={18} aria-hidden="true" />Export session notes</button>
          <button className="btn btn-secondary btn-block" disabled={!transcript.length} onClick={() => window.print()}><Printer size={18} aria-hidden="true" />Print / save as PDF</button>
          <div className="or-rule" style={{ margin: '6px 0' }}>then</div>
          <button className="btn btn-secondary btn-block" onClick={() => transcript.length ? setConfirmReset(true) : resetSession()}><Plus size={18} aria-hidden="true" />New {domain.session}</button>
          {confirmReset && <div className="notice notice-warn pop" role="alert" style={{ flexDirection: 'column' }}>
            <p style={{ display: 'flex', gap: 10 }}><AlertTriangle size={18} aria-hidden="true" /><span>Starting a new {domain.session} clears these messages. Export a copy first if you need one.</span></p>
            <div className="chip-row" style={{ marginTop: 10 }}><button className="btn btn-danger btn-sm" onClick={resetSession}>Clear & start new</button><button className="btn btn-secondary btn-sm" onClick={() => setConfirmReset(false)}>Cancel</button></div>
          </div>}
          {status && <p role="status" className="notice notice-ok"><Check size={18} aria-hidden="true" /><span>{status}</span></p>}
          {!status && <p role="status" className="sr-only" />}
        </div>
      </aside>
    </div>
  </div></main>;
}
