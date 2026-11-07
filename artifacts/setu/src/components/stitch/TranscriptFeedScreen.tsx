import { useState } from 'react';
import { Search, FileText, Volume2, X } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { speak } from '../../lib/speech';
import { useDomain } from '../../lib/useDomain';

export function TranscriptFeedScreen() {
  const { transcript, selectedLang } = useSession();
  const domain = useDomain();
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const items = transcript.filter(item => (filter === 'all' || item.speaker === filter) && `${item.text} ${item.textEn || ''}`.toLowerCase().includes(query.toLowerCase().trim()));
  const initial = (name: string) => name.trim().charAt(0).toUpperCase();
  return <main className="page"><div className="page-inner">
    <div className="page-head">
      <div><span className="eyebrow indigo">Conversation history</span><h1>Transcript</h1><p className="lede">Both sides of the {domain.session}, in order. The {domain.roles.staff.lower} on the left, the {domain.roles.client.lower} on the right.</p></div>
      <div className="page-head-actions"><a href="#summary" className="btn btn-secondary">Review summary</a></div>
    </div>

    <section className="card card-pad" style={{ display: 'flex', flexDirection: 'column', gap: 14 }} aria-label="Search and filter">
      <div className="search">
        <Search size={20} aria-hidden="true" />
        <label className="sr-only" htmlFor="transcript-search">Search transcript</label>
        <input id="transcript-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search messages…" />
        {query && <button className="btn btn-ghost btn-sm" onClick={() => setQuery('')} aria-label="Clear search"><X size={16} aria-hidden="true" /></button>}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <div className="chip-row" role="group" aria-label="Filter by speaker">
          {['all', 'staff', 'client'].map(value => <button key={value} aria-pressed={filter === value} className="chip" onClick={() => setFilter(value)}>{value === 'all' ? 'Everyone' : value === 'staff' ? domain.roles.staff.label : domain.roles.client.label}</button>)}
        </div>
        <p className="card-note" role="status" style={{ fontWeight: 700 }}>{items.length} {items.length === 1 ? 'message' : 'messages'}</p>
      </div>
    </section>

    {items.length ? <>
      <div className="legend" aria-hidden="true"><span><i style={{ background: 'var(--indigo)' }} />{domain.roles.staff.label}</span><span><i style={{ background: 'var(--teal)' }} />{domain.roles.client.label}</span></div>
      <ol className="thread stagger" style={{ listStyle: 'none', margin: 0, padding: 0 }}>
        {items.map(item => <li className={`bubble ${item.speaker}`} key={item.id}>
          <div className="bubble-meta"><span className="bubble-who"><i aria-hidden="true">{initial(item.speakerName)}</i>{item.speakerName}</span><time>{item.timestamp}</time></div>
          <p className="bubble-text">{item.text}</p>
          {item.textEn && item.textEn !== item.text && <p className="bubble-en">{item.textEn}</p>}
          <div className="bubble-actions"><button className="btn btn-secondary btn-sm" onClick={() => speak(item.text, selectedLang)} aria-label={`Read ${item.speakerName} message aloud`}><Volume2 size={16} aria-hidden="true" />Read aloud</button></div>
        </li>)}
      </ol>
    </> : <section className="card"><div className="empty">
      <span className="empty-glyph"><FileText size={30} aria-hidden="true" /></span>
      <h2>{transcript.length ? 'No matching messages' : 'The conversation starts here'}</h2>
      <p>{transcript.length ? 'Try another word, or show everyone.' : `Every message from the ${domain.session} appears here as it is said or signed.`}</p>
      {transcript.length ? <button className="btn btn-secondary" onClick={() => { setQuery(''); setFilter('all'); }}>Clear filters</button> : <a href="#bridge" className="btn btn-primary">Open the {domain.session}</a>}
    </div></section>}
  </div></main>;
}
