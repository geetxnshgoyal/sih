import { Activity, Settings2 } from 'lucide-react';
import { useSession, type ActiveView } from '../../context/SessionContext';
import { useDomain, useSetDomain } from '../../lib/useDomain';
import { DOMAIN_LIST, type DomainId, type RoleId } from '../../lib/domains';

type NavView = Exclude<ActiveView, 'diagnostics' | 'devices'>;

export const navigation: { view: NavView; label: string }[] = [
  { view: 'home', label: 'Overview' }, { view: 'bridge', label: 'Counter' },
  { view: 'capture', label: 'Capture' }, { view: 'language', label: 'Languages' }, { view: 'transcript', label: 'Transcript' },
  { view: 'phrases', label: 'Phrases' }, { view: 'summary', label: 'Summary' },
];

export function StitchHeader() {
  const { activeView, selectedRole, setSelectedRole } = useSession();
  const domain = useDomain();
  const setDomain = useSetDomain();

  return <header className="app-header">
    <div className="header-inner">
      <a href="#home" className="brand-link" aria-label="Setu overview"><span className="brand-icon"><Activity size={24} /></span><span className="brand-title">Setu</span></a>
      <nav className="desktop-nav" aria-label="Main navigation">
        {navigation.map(({view, label}) => <a key={view} href={`#${view}`} className="nav-link" aria-current={activeView === view ? 'page' : undefined}>{label}</a>)}
      </nav>
      <div className="flex items-center gap-2">
        {/* The setting, and then the side of the counter you are on.
            Setting first because it renames the second control: the same
            toggle reads Doctor/Patient in a hospital and Teller/Customer at
            a bank. One build, one model, six counters. */}
        <label className="sr-only" htmlFor="counter-setting">Counter setting</label>
        <select
          id="counter-setting"
          className="role-select"
          value={domain.id}
          onChange={e => setDomain(e.target.value as DomainId)}
        >
          {DOMAIN_LIST.map(d => <option key={d.id} value={d.id}>{d.label}</option>)}
        </select>

        <label className="sr-only" htmlFor="counter-role">Your side of the counter</label>
        <select
          id="counter-role"
          className="role-select"
          value={selectedRole}
          onChange={e => setSelectedRole(e.target.value as RoleId)}
        >
          <option value="staff">{domain.roles.staff.label} view</option>
          <option value="client">{domain.roles.client.label} view</option>
        </select>

        <a href="#diagnostics" className="nav-link" aria-label="System diagnostics" aria-current={activeView === 'diagnostics' ? 'page' : undefined}><Settings2 size={20} /></a>
      </div>
    </div>
  </header>;
}
