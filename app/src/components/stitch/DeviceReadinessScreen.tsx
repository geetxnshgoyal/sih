import { useEffect, useRef, useState } from 'react';
import { Camera, Mic, ArrowRight, CheckCircle2, AlertCircle } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
export function DeviceReadinessScreen() {
  const { deviceReady, setDeviceReady } = useSession();
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState('Test device access before starting. You can also continue using text.');
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  async function checkDevices() {
    setChecking(true);
    const results = await Promise.allSettled([
      navigator.mediaDevices?.getUserMedia({video: true}) ?? Promise.reject(new Error('Camera unavailable')),
      navigator.mediaDevices?.getUserMedia({audio: true}) ?? Promise.reject(new Error('Microphone unavailable')),
    ]);
    results.forEach(result => { if (result.status === 'fulfilled') result.value.getTracks().forEach(track => track.stop()); });
    if (!mounted.current) return;
    const ready = {camera: results[0].status === 'fulfilled', mic: results[1].status === 'fulfilled'};
    setDeviceReady(ready); setChecking(false);
    setMessage(ready.camera && ready.mic ? 'Both devices are available. You can start your consultation.' : 'Some devices are unavailable. Check browser permissions and try again, or continue with text.');
  }
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-6">
    <div><span className="eyebrow-label">Consultation setup · Step 2</span><h1 className="text-display-md font-bold mt-2">Check your devices</h1><p className="text-secondary mt-3">Make sure your camera and microphone are available for the visit.</p></div>
    <div className="grid md:grid-cols-2 gap-5">{[{key: 'camera' as const, title: 'Camera', Icon: Camera, detail: 'For capturing sign language gestures.'}, {key: 'mic' as const, title: 'Microphone', Icon: Mic, detail: 'For spoken messages from the doctor.'}].map(({key, title, Icon, detail}) => <section className="content-card" key={key}><Icon className="text-primary mb-4" size={28}/><h2 className="text-headline-md font-semibold">{title}</h2><p className="text-secondary mt-2">{detail}</p><p className="flex gap-2 items-center text-label-md mt-5">{deviceReady[key] ? <CheckCircle2 size={18}/> : <AlertCircle size={18}/>} {deviceReady[key] ? 'Access verified' : 'Access not verified'}</p></section>)}</div>
    <p role="status" className="text-secondary">{message}</p>
    <div className="flex flex-wrap gap-3"><button className="primary-action" onClick={checkDevices} disabled={checking}>{checking ? 'Checking devices…' : 'Test camera & microphone'}</button><a href="#bridge" className="secondary-action">{deviceReady.camera && deviceReady.mic ? 'Start consultation' : 'Continue with text'}<ArrowRight size={18}/></a><a href="#language" className="secondary-action">Back to languages</a></div>
  </div></main>;
}
