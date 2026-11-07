import { InstallButton } from '../InstallPrompt';
import { ArrowRight, Video, Languages, MessageSquare, FileText, ClipboardCheck, Settings2 } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { useDomain } from '../../lib/useDomain';
import type { Domain } from '../../lib/domains';
const modules = (d: Domain) => [
  { view: 'bridge', title: `Live ${d.session}`, description: 'Speak, type, or sign to keep the conversation moving.', cta: `Open ${d.session}`, icon: Video },
  { view: 'language', title: 'Spoken language', description: 'Choose the language used for speech and playback.', cta: 'Choose language', icon: Languages },
  { view: 'phrases', title: `${d.label} phrases`, description: 'Find the common questions and answers for this counter.', cta: 'Browse phrases', icon: MessageSquare },
  { view: 'transcript', title: 'Conversation transcript', description: `Review and search messages from this ${d.session}.`, cta: 'View transcript', icon: FileText },
  { view: 'summary', title: `${d.session.charAt(0).toUpperCase() + d.session.slice(1)} summary`, description: 'Review the conversation and export your session notes.', cta: 'Review summary', icon: ClipboardCheck },
  { view: 'diagnostics', title: 'Device & system checks', description: 'Check device access and the available sign library.', cta: 'Check system', icon: Settings2 },
];
export function HomeScreen() {
  const { selectedRole, transcript } = useSession();
  const domain = useDomain();
  return <main><div className="mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-8">
    <section className="home-hero">
      <div><span className="eyebrow-label">{domain.tagline}</span>
        <h1 className="text-display-lg font-bold tracking-tight">A clearer conversation.<br /><span className="text-primary">A more connected visit.</span></h1>
        <p className="text-body-lg">A shared space for Deaf or hard of hearing people and the staff they need to talk to. Bring speech, sign playback, and written messages to one counter.</p>
        <div className="home-actions"><a href="#language" className="primary-action">Set up the counter <ArrowRight size={18} /></a><a href="#bridge" className="secondary-action">Open {domain.roles[selectedRole].lower} view</a></div>
      </div>
      <aside className="setup-card"><h2>Ready when you are</h2>
        {[['Choose your language', 'Select a spoken language for this visit.'], ['Check your devices', 'Test camera and microphone access.'], ['Start the conversation', 'Use speech, text, and quick phrases.']].map(([title, description], i) => <div className="setup-step" key={title}><span>{i + 1}</span><div><strong>{title}</strong><p>{description}</p></div></div>)}
      </aside>
    </section>
    <section aria-labelledby="workspace-heading"><div className="flex items-center justify-between mb-5 gap-3"><div><span className="eyebrow-label">Your workspace</span><h2 id="workspace-heading" className="text-headline-lg font-semibold mt-1">Everything for the conversation</h2></div><span className="text-label-md text-secondary">{transcript.length} {transcript.length === 1 ? "message" : "messages"} this session</span></div>
      <div className="module-grid">{modules(domain).map(({view, title, description, cta, icon: Icon}) => <a href={`#${view}`} key={view} className="module-card"><span className="module-icon"><Icon size={22}/></span><h3>{title}</h3><p>{description}</p><span className="module-cta flex items-center gap-2">{cta}<ArrowRight size={16}/></span></a>)}</div>
    </section>

    {/* What this is, who built it, and what it cannot do.
        The last part is not modesty. Sign recognition is right about 74% of the
        time for someone the model has never seen, and the dictionary
        shortlist behind it about half the time, so anyone using this in a
        clinic needs to know the phrase board is the reliable path and the
        camera is the shortcut. Saying so here is cheaper than a doctor
        discovering it mid-consultation. */}
    <section aria-labelledby="about-heading" className="about-setu">
      <div>
        <span className="eyebrow-label">About</span>
        <h2 id="about-heading" className="text-headline-lg font-semibold mt-1">Setu means bridge</h2>
      </div>
      <div className="about-grid">
        <div>
          <h3>The problem</h3>
          <p>
            India's National Programme for Prevention and Control of Deafness
            puts significant hearing loss at 63 million people, and severe to
            profound loss at 291 per lakh, about four million. The government's
            own directory of certified Indian Sign Language interpreters lists a
            few hundred. In a hospital that gap is dangerous: symptoms get
            described by a relative, consent is given without being understood,
            and a patient who signs fluently is treated as though they cannot
            communicate at all.
          </p>
        </div>
        <div>
          <h3>What Setu does</h3>
          <p>
            Signs are recognised by the camera and spoken aloud in six Indian
            languages. Speech comes back as text and sign playback. A tap-to-speak
            phrase board covers the things that matter most when nothing else is
            working.
          </p>
        </div>
        <div>
          <h3>It runs on the device</h3>
          <p>
            The classifier is under 2&nbsp;MB and the hand and face tracker it
            sits on is another 14&nbsp;MB. Both are served from this site and run
            in the browser. No video leaves the machine, nothing is sent to a
            server, and once the page has loaded it keeps working with the
            network down. There is no account, no API key and no running cost.
          </p>
        </div>
        <div>
          <h3>What it cannot do yet</h3>
          <p>
            Setu recognises 38 signs on its own, and gets the right one
            first about 74% of the time for a signer it has never seen, with
            the right answer among its top five about 95% of the time. Another
            203 words, <em>pain</em>, <em>water</em> and <em>help</em> among
            them, have a single reference clip each and are offered only as a
            shortlist of closest guesses; that path is right about half the
            time, so read it as a suggestion. Some words are in no corpus at
            all, including <em>toilet</em>, <em>chest</em>, <em>leg</em> and
            <em>vomit</em>. Accuracy also drops sharply for signers recorded in
            very different conditions. The camera is a shortcut, not a
            substitute for an interpreter. Translations are machine generated
            and await review by Deaf signers.
          </p>
        </div>
      </div>
      <p className="about-install">
        <InstallButton />
      </p>
      <p className="about-credit">
        Built by <strong>Team Awaaz</strong> for Smart India Hackathon 2026.
        Free to use, and always will be.{" "}
        <a href="https://github.com/geetxnshgoyal/sih/blob/main/PRIVACY.md"
           target="_blank" rel="noopener noreferrer">Privacy</a>
        {" · "}
        <a href="https://github.com/geetxnshgoyal/sih" target="_blank"
           rel="noopener noreferrer">Source</a>
      </p>
    </section>
  </div></main>;
}
