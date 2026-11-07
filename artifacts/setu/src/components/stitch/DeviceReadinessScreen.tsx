import { useEffect, useRef, useState } from 'react';
import { Camera, Mic, ArrowRight, CheckCircle2, CircleDashed, Info } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { useDomain } from '../../lib/useDomain';

export function DeviceReadinessScreen() {
  const { deviceReady, setDeviceReady } = useSession();
  const domain = useDomain();
  const [checking, setChecking] = useState(false);
  const [tested, setTested] = useState(false);
  const [message, setMessage] = useState('Test device access before starting. You can also continue using text.');
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  async function checkDevices() {
    setChecking(true);
    const results = await Promise.allSettled([
      navigator.mediaDevices?.getUserMedia({ video: true }) ?? Promise.reject(new Error('Camera unavailable')),
      navigator.mediaDevices?.getUserMedia({ audio: true }) ?? Promise.reject(new Error('Microphone unavailable')),
    ]);
    results.forEach(result => { if (result.status === 'fulfilled') result.value.getTracks().forEach(track => track.stop()); });
    if (!mounted.current) return;
    const ready = { camera: results[0].status === 'fulfilled', mic: results[1].status === 'fulfilled' };
    setDeviceReady(ready); setChecking(false); setTested(true);
    setMessage(ready.camera && ready.mic ? `Both devices are available. You can start your ${domain.session}.` : 'Some devices are unavailable. Check browser permissions and try again, or continue with text.');
  }
  const allGood = deviceReady.camera && deviceReady.mic;
  const devices = [
    { key: 'camera' as const, title: 'Camera', Icon: Camera, detail: `Sees the ${domain.roles.client.lower}'s hands and face. Video stays on this device.` },
    { key: 'mic' as const, title: 'Microphone', Icon: Mic, detail: `Hears the ${domain.roles.staff.lower}. Speech is turned into text and signs.` },
  ];
  return <main className="page"><div className="page-inner">
    <div className="page-head">
      <div><span className="eyebrow indigo">Setup · Step 2 of 2</span><h1>Check the camera and microphone.</h1><p className="lede">Both are optional. If either is blocked, typing and the phrase board still work completely.</p></div>
    </div>
    <div className="device-grid stagger">
      {devices.map(({ key, title, Icon, detail }) => {
        const ok = deviceReady[key];
        return <section className={`device${ok ? ' ok' : checking ? ' checking' : ''}`} key={key} aria-labelledby={`device-${key}`}>
          <span className="device-icon"><Icon size={28} aria-hidden="true" /></span>
          <h2 id={`device-${key}`}>{title}</h2>
          <p>{detail}</p>
          <p className="device-state" role="status">
            {checking ? <><span className="spinner" aria-hidden="true" /> Checking…</>
              : ok ? <><CheckCircle2 size={20} aria-hidden="true" /> Access confirmed</>
              : <><CircleDashed size={20} aria-hidden="true" /> {tested ? 'Not available' : 'Not tested yet'}</>}
          </p>
        </section>;
      })}
    </div>
    <p role="status" className={`notice ${allGood ? 'notice-ok' : tested ? 'notice-warn' : 'notice-info'}`}><Info size={18} aria-hidden="true" /><span>{message}</span></p>
    <div className="chip-row">
      <button className="btn btn-primary btn-lg" onClick={checkDevices} disabled={checking}>{checking ? 'Checking devices…' : tested ? 'Test again' : 'Test camera & microphone'}</button>
      <a href="#bridge" className={`btn btn-lg ${allGood ? 'btn-teal' : 'btn-secondary'}`}>{allGood ? `Start the ${domain.session}` : 'Continue with text'}<ArrowRight size={20} aria-hidden="true" /></a>
      <a href="#language" className="btn btn-ghost btn-lg">Back to languages</a>
    </div>
  </div></main>;
}
