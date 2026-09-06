import { Activity, Settings2 } from 'lucide-react';
import { useSession, type ActiveView } from '../../context/SessionContext';

type NavView = Exclude<ActiveView, 'diagnostics' | 'devices'>;

export const navigation: { view: NavView; label: string }[] = [
  { view: 'home', label: 'Overview' }, { view: 'bridge', label: 'Consultation' },
  { view: 'capture', label: 'Capture' }, { view: 'language', label: 'Languages' }, { view: 'transcript', label: 'Transcript' },
  { view: 'phrases', label: 'Phrases' }, { view: 'summary', label: 'Summary' },
];

export function StitchHeader() {
  const { activeView, selectedRole, setSelectedRole } = useSession();
  return <header className="app-header">
    <div className="header-inner">
      <a href="#home" className="brand-link" aria-label="Setu overview"><span className="brand-icon"><Activity size={24} /></span><span className="brand-title">Setu</span></a>
      <nav className="desktop-nav" aria-label="Main navigation">
        {navigation.map(({view, label}) => <a key={view} href={`#${view}`} className="nav-link" aria-current={activeView === view ? 'page' : undefined}>{label}</a>)}
      </nav>
      <div className="flex items-center gap-2">
        <label className="sr-only" htmlFor="consultation-role">Consultation role</label>
        <select id="consultation-role" className="role-select" value={selectedRole} onChange={e => setSelectedRole(e.target.value as 'doctor' | 'patient')}>
          <option value="doctor">Doctor view</option><option value="patient">Patient view</option>
        </select>
        <a href="#diagnostics" className="nav-link" aria-label="System diagnostics" aria-current={activeView === 'diagnostics' ? 'page' : undefined}><Settings2 size={20} /></a>
      </div>
    </div>
  </header>;
}
