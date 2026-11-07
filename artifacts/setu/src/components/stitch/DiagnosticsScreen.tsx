import { useState } from 'react';
import { Settings2, CheckCircle2, AlertTriangle, XCircle, Camera } from 'lucide-react';
import { asset } from '../../lib/assetUrl';

type Check = { label: string; level: 'ok' | 'warn' | 'bad' };

export function DiagnosticsScreen() {
  const [checking, setChecking] = useState(false);
  const [results, setResults] = useState<Check[]>([]);
  async function runChecks() {
    setChecking(true); setResults([]);
    const checks = await Promise.allSettled(['/model/model.json', '/model/_signs.json'].map(async path => { const response = await fetch(asset(path)); if (!response.ok) throw new Error('Unavailable'); const data = await response.json(); return path.includes('_signs') ? `${Object.keys(data).length} sign recordings available` : 'Recognition model manifest available'; }));
    const speech = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
    setResults([
      { label: `Secure browser context: ${window.isSecureContext ? 'yes' : 'no'}`, level: window.isSecureContext ? 'ok' : 'bad' },
      { label: `Device access API: ${navigator.mediaDevices ? 'available' : 'unavailable'}`, level: navigator.mediaDevices ? 'ok' : 'bad' },
      { label: `Speech input: ${speech ? 'supported' : 'not supported; use text input'}`, level: speech ? 'ok' : 'warn' },
      ...checks.map((r, i): Check => r.status === 'fulfilled' ? { label: r.value, level: 'ok' } : { label: `${i === 0 ? 'Recognition model' : 'Sign library'} unavailable. Check the model files and try again.`, level: 'bad' }),
    ]);
    setChecking(false);
  }
  const Icon = { ok: CheckCircle2, warn: AlertTriangle, bad: XCircle };
  return <main className="page"><div className="page-inner">
    <div className="page-head"><div><span className="eyebrow indigo">Workspace settings</span><h1>System checks</h1><p className="lede">What this browser supports and which resources are loaded on this machine. Run it when speech or sign playback seems unavailable.</p></div></div>
    <section className="card" aria-labelledby="checks-heading">
      <div className="card-h"><span><Settings2 size={15} aria-hidden="true" /> Browser & resources</span><span>{results.length ? `${results.filter(r => r.level === 'ok').length} of ${results.length} passed` : 'Not run'}</span></div>
      <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
        <h2 id="checks-heading" className="sr-only">Browser and resources</h2>
        <button className="btn btn-primary btn-lg" style={{ alignSelf: 'flex-start' }} onClick={runChecks} disabled={checking}>{checking ? <><span className="spinner" aria-hidden="true" /> Checking…</> : results.length ? 'Run again' : 'Run system checks'}</button>
        <ul className="check-list" aria-live="polite" style={{ listStyle: 'none', margin: 0, padding: 0 }}>
          {checking && [0, 1, 2, 3, 4].map(i => <li key={i} className="skeleton" style={{ height: 56 }} aria-hidden="true" />)}
          {results.map((r, i) => { const I = Icon[r.level]; return <li key={r.label} className={`check-item ${r.level}`} style={{ animationDelay: `${i * 60}ms` }}><i><I size={18} aria-hidden="true" /></i><span>{r.label}</span></li>; })}
        </ul>
        {!checking && !results.length && <p className="card-note">Nothing has been checked yet. The checks take a second and change nothing on the device.</p>}
      </div>
    </section>
    <div className="chip-row"><a href="#devices" className="btn btn-secondary btn-lg"><Camera size={20} aria-hidden="true" />Test camera & microphone</a><a href="#bridge" className="btn btn-ghost btn-lg">Return to the counter</a></div>
  </div></main>;
}
