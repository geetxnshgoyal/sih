import { CircleDot, ClipboardCheck, FileText, House, Languages, MessageSquare, Video } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { navigation } from './StitchHeader';
const icons = { home: House, bridge: Video, capture: CircleDot, language: Languages, transcript: FileText, phrases: MessageSquare, summary: ClipboardCheck };
const mobileLabels = { home: 'Home', bridge: 'Consult', capture: 'Capture', language: 'Lang', transcript: 'Log', phrases: 'Phrases', summary: 'Summary' };
export function StitchNav() {
  const { activeView } = useSession();
  return <nav className="mobile-nav" aria-label="Main navigation"><div>
    {navigation.map(({view}) => { const Icon = icons[view]; return <a key={view} className="nav-link" href={`#${view}`} aria-current={activeView === view ? 'page' : undefined}><Icon size={21} /><span>{mobileLabels[view]}</span></a>; })}
  </div></nav>;
}
