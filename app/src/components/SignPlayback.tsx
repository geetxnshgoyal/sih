import { useMemo, useState } from 'react';
import { Hand, RotateCcw } from 'lucide-react';
import { useSignLibrary } from '../hooks/useSignLibrary';
import { textToGlosses, type SignLibrary } from '../lib/reverse';
import SignPlayer from './SignPlayer';

function Sequence({ matched, library, slow }: { matched: string[]; library: SignLibrary; slow: boolean }) {
  const [index, setIndex] = useState(0);
  const [replay, setReplay] = useState(0);
  const [finished, setFinished] = useState(false);
  const current = matched[index];
  return <>
    <div className="rounded-xl bg-surface-container-low min-h-[300px] flex flex-col items-center justify-center p-4">
      <SignPlayer key={`${replay}-${index}`} frames={library[current]} fps={slow ? 10 : 14} label={`Sign: ${current}`} onComplete={() => {
        if (index < matched.length - 1) setIndex(i => i + 1);
        else setFinished(true);
      }}/>
      <strong className="text-headline-md text-primary">{current}</strong>
      <p className="text-label-md text-secondary mt-2" role="status">{index + 1} of {matched.length} signs{finished ? ' · Playback complete' : ''}</p>
    </div>
    <div className="flex flex-wrap gap-2 mt-3" aria-label="Sign sequence">{matched.map((g, i) =>
      <button key={`${g}-${i}`} className="secondary-action" aria-current={index === i ? 'step' : undefined}
        onClick={() => { setIndex(i); setReplay(r => r + 1); setFinished(false); }}>{g}</button>)}</div>
    <button className="secondary-action mt-4" onClick={() => { setIndex(0); setReplay(r => r + 1); setFinished(false); }}><RotateCcw size={18}/>Replay signs</button>
  </>;
}
export default function SignPlayback({ text, messageId, slow = false }: { text: string; messageId?: string | number; slow?: boolean }) {
  const { library, loading, error, retry } = useSignLibrary();
  const { matched, skipped } = useMemo(() => textToGlosses(text, library), [text, library]);
  return <div>
    {error ? <div role="alert"><p>Sign recordings could not load. {error}</p><button className="secondary-action mt-3" onClick={retry}>Retry playback</button></div>
      : loading ? <p role="status">Loading sign recordings…</p>
      : matched.length ? <Sequence key={`${messageId}-${text}-${slow}`} matched={matched} library={library} slow={slow}/>
      : <div className="empty-state"><Hand size={36} className="mx-auto mb-3 text-primary"/><p>{text ? 'No sign recordings match this message.' : 'Send a message to play available signs.'}</p></div>}
    {skipped.length > 0 && !loading && !error && <p className="text-label-md text-secondary mt-4" role="status">Text only: {skipped.join(', ')}. Read the complete message alongside these signs.</p>}
    {matched.length > 0 && <p className="text-label-sm text-secondary mt-3">Recorded signs in word order. This preview does not translate full ISL grammar.</p>}
  </div>;
}
