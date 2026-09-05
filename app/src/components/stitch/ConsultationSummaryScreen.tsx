import { useState } from 'react';
import { Download, Printer, Plus, FileText } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
export function ConsultationSummaryScreen() {
  const { transcript, selectedLang, resetSession } = useSession();
  const [confirmReset, setConfirmReset] = useState(false);
  const [status, setStatus] = useState('');
  function exportNotes() {
    const data = { exportedAt: new Date().toISOString(), language: selectedLang, messages: transcript };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = `setu-consultation-${new Date().toISOString().slice(0,10)}.json`; document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000); setStatus('Session notes exported.');
  }
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div><span className="eyebrow-label">Session review</span><h1 className="text-display-md font-bold mt-2">Consultation summary</h1><p className="text-secondary mt-3">Review the messages from this visit before saving a copy.</p></div>
    <div className="grid lg:grid-cols-3 gap-6 items-start"><section className="content-card lg:col-span-2"><div className="flex items-center justify-between mb-5"><h2 className="text-headline-md font-semibold">Conversation notes</h2><span className="text-label-md text-secondary">{transcript.length} {transcript.length === 1 ? "message" : "messages"} · {selectedLang}</span></div>
      {transcript.length === 0 ? <div className="empty-state"><FileText size={32} className="mx-auto mb-4"/><h3 className="text-headline-md font-semibold">No messages yet</h3><p className="mt-2">Start a consultation to add messages to your summary.</p><a href="#bridge" className="primary-action mt-5 no-print">Open consultation</a></div> : <ol className="flex flex-col gap-5">{transcript.map(item => <li key={item.id} className="border-b border-outline-variant/30 pb-4"><div className="flex items-center justify-between text-label-sm text-secondary mb-2"><span className="font-semibold capitalize">{item.speaker}</span><time>{item.timestamp}</time></div><p className="text-body-lg">{item.text}</p>{item.textEn && item.textEn !== item.text && <p className="text-secondary mt-2">{item.textEn}</p>}{item.medication && <p className="mt-2">{item.medication.instructions}</p>}</li>)}</ol>}
    </section><aside className="content-card no-print flex flex-col gap-3"><h2 className="text-headline-md font-semibold mb-2">Save this visit</h2><button className="primary-action" disabled={!transcript.length} onClick={exportNotes}><Download size={18}/>Export session notes</button><button className="secondary-action" disabled={!transcript.length} onClick={() => window.print()}><Printer size={18}/>Print / save as PDF</button><button className="secondary-action" onClick={() => transcript.length ? setConfirmReset(true) : resetSession()}><Plus size={18}/>New consultation</button>{confirmReset && <div className="rounded-xl bg-surface-container-low p-4" role="alert"><p className="text-label-md mb-3">Starting a new visit clears these messages. Export a copy first if needed.</p><div className="flex flex-wrap gap-2"><button className="primary-action" onClick={resetSession}>Clear & start new</button><button className="secondary-action" onClick={() => setConfirmReset(false)}>Cancel</button></div></div>}<p role="status" className="text-label-md text-primary">{status}</p></aside></div>
  </div></main>;
}
