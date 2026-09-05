import { ArrowRight, Video, Languages, MessageSquare, FileText, ClipboardCheck, Settings2 } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
const modules = [
  { view: 'bridge', title: 'Live consultation', description: 'Speak, type, or sign to keep the conversation moving.', cta: 'Open consultation', icon: Video },
  { view: 'language', title: 'Spoken language', description: 'Choose the language used for speech and playback.', cta: 'Choose language', icon: Languages },
  { view: 'phrases', title: 'Clinical phrases', description: 'Find common questions and instructions for your visit.', cta: 'Browse phrases', icon: MessageSquare },
  { view: 'transcript', title: 'Conversation transcript', description: 'Review and search messages from this consultation.', cta: 'View transcript', icon: FileText },
  { view: 'summary', title: 'Consultation summary', description: 'Review the conversation and export your session notes.', cta: 'Review summary', icon: ClipboardCheck },
  { view: 'diagnostics', title: 'Device & system checks', description: 'Check device access and the available sign library.', cta: 'Check system', icon: Settings2 },
];
export function HomeScreen() {
  const { selectedRole, transcript } = useSession();
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-8">
    <section className="home-hero">
      <div><span className="eyebrow-label">Healthcare communication, together</span>
        <h1 className="text-display-lg font-bold tracking-tight">A clearer conversation.<br /><span className="text-primary">A more connected visit.</span></h1>
        <p className="text-body-lg">A shared space for doctors and Deaf or hard of hearing patients. Bring speech, sign playback, and written messages into one consultation.</p>
        <div className="home-actions"><a href="#language" className="primary-action">Set up a consultation <ArrowRight size={18} /></a><a href="#bridge" className="secondary-action">Open {selectedRole} view</a></div>
      </div>
      <aside className="setup-card"><h2>Ready when you are</h2>
        {[['Choose your language', 'Select a spoken language for this visit.'], ['Check your devices', 'Test camera and microphone access.'], ['Start the conversation', 'Use speech, text, and quick phrases.']].map(([title, description], i) => <div className="setup-step" key={title}><span>{i + 1}</span><div><strong>{title}</strong><p>{description}</p></div></div>)}
      </aside>
    </section>
    <section aria-labelledby="workspace-heading"><div className="flex items-center justify-between mb-5 gap-3"><div><span className="eyebrow-label">Your workspace</span><h2 id="workspace-heading" className="text-headline-lg font-semibold mt-1">Everything for the conversation</h2></div><span className="text-label-md text-secondary">{transcript.length} {transcript.length === 1 ? "message" : "messages"} this session</span></div>
      <div className="module-grid">{modules.map(({view, title, description, cta, icon: Icon}) => <a href={`#${view}`} key={view} className="module-card"><span className="module-icon"><Icon size={22}/></span><h3>{title}</h3><p>{description}</p><span className="module-cta flex items-center gap-2">{cta}<ArrowRight size={16}/></span></a>)}</div>
    </section>
  </div></main>;
}
