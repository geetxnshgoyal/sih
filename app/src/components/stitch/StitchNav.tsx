import { House, Video, Languages, FileText, MessageSquare, ClipboardCheck } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { navigation } from './StitchHeader';
const icons = [House, Video, Languages, FileText, MessageSquare, ClipboardCheck];
export function StitchNav() {
  const { activeView } = useSession();
  return <nav className="mobile-nav" aria-label="Main navigation"><div>
    {navigation.map(({view, label}, index) => { const Icon = icons[index]; return <a key={view} className="nav-link" href={`#${view}`} aria-current={activeView === view ? 'page' : undefined}><Icon size={21} /><span>{view === "home" ? "Home" : view === "bridge" ? "Consult" : view === "language" ? "Language" : label}</span></a>; })}
  </div></nav>;
}
