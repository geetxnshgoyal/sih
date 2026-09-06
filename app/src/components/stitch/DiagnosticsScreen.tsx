import { useState } from 'react';
import { Settings2 } from 'lucide-react';
import { asset } from '../../lib/assetUrl';
export function DiagnosticsScreen() {
  const [checking, setChecking] = useState(false);
  const [results, setResults] = useState<string[]>([]);
  async function runChecks() {
    setChecking(true); setResults([]);
    const checks = await Promise.allSettled(['/model/model.json', '/model/_signs.json'].map(async path => { const response = await fetch(asset(path)); if (!response.ok) throw new Error('Unavailable'); const data = await response.json(); return path.includes('_signs') ? `${Object.keys(data).length} sign recordings available` : 'Recognition model manifest available'; }));
    setResults([`Secure browser context: ${window.isSecureContext ? 'yes' : 'no'}`, `Device access API: ${navigator.mediaDevices ? 'available' : 'unavailable'}`, `Speech input: ${('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) ? 'supported' : 'not supported; use text input'}`, ...checks.map((r, i) => r.status === 'fulfilled' ? r.value : `${i === 0 ? 'Recognition model' : 'Sign library'} unavailable. Check the model files and try again.`)]);
    setChecking(false);
  }
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6"><div><span className="eyebrow-label">Workspace settings</span><h1 className="text-display-md font-bold mt-2">System checks</h1><p className="text-secondary mt-3">Check browser support and the resources available to this app.</p></div><section className="content-card"><Settings2 size={28} className="text-primary mb-4"/><h2 className="text-headline-md font-semibold">Browser & resources</h2><p className="text-secondary my-3">Run a check if speech or sign playback is unavailable.</p><button className="primary-action" onClick={runChecks} disabled={checking}>{checking ? 'Checking…' : 'Run system checks'}</button><ul className="flex flex-col gap-3 mt-5" aria-live="polite">{results.map(result => <li key={result} className="bg-surface-container-low rounded-xl p-3 text-label-md">{result}</li>)}</ul></section><div className="flex flex-wrap gap-3"><a href="#devices" className="secondary-action">Test camera & microphone</a><a href="#bridge" className="primary-action">Return to consultation</a></div></div></main>;
}
