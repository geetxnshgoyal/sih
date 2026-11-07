import { InstallButton } from '../InstallPrompt';
import { ArrowRight, Hand, Languages, MessageSquare, FileText, ClipboardCheck, Settings2, ShieldCheck, WifiOff, Cpu } from 'lucide-react';
import { useSession } from '../../context/SessionContext';
import { useDomain } from '../../lib/useDomain';
import type { Domain } from '../../lib/domains';
import heroArt from '../../assets/hero.png';

const modules = (d: Domain) => [
  { view: 'bridge', title: `Live ${d.session}`, description: 'Speak, type, or sign. The other side of the counter sees it at once, in their language.', cta: `Open the ${d.session}`, icon: Hand },
  { view: 'phrases', title: `${d.label} phrases`, description: 'The questions and answers this counter actually needs, ready to send in one tap.', cta: 'Browse phrases', icon: MessageSquare },
  { view: 'language', title: 'Spoken language', description: 'Hindi, Tamil, Telugu, Bengali, Marathi or English for speech and playback.', cta: 'Choose language', icon: Languages },
  { view: 'transcript', title: 'Transcript', description: `Every line from both sides of this ${d.session}, searchable.`, cta: 'View transcript', icon: FileText },
  { view: 'summary', title: `${d.session.charAt(0).toUpperCase() + d.session.slice(1)} summary`, description: 'Review, export or print the record before the next person steps up.', cta: 'Review summary', icon: ClipboardCheck },
  { view: 'diagnostics', title: 'Device & system checks', description: 'Camera, microphone, speech support and the sign library on this machine.', cta: 'Check system', icon: Settings2 },
];

export function HomeScreen() {
  const { selectedRole, transcript } = useSession();
  const domain = useDomain();
  return <main className="page"><div className="page-inner">
    <section className="hero rise">
      <div>
        <span className="eyebrow">{domain.tagline}</span>
        <h1>Two people.<br />One counter.<br /><em>No interpreter needed.</em></h1>
        <p className="lede">Setu turns Indian Sign Language into speech and speech back into signs, right here at the {domain.station.split(' · ')[0].toLowerCase()}. Nothing leaves this device.</p>
        <div className="hero-actions">
          <a href="#bridge" className="btn btn-primary btn-lg">Open the {domain.roles[selectedRole].lower} view <ArrowRight size={20} /></a>
          <a href="#language" className="btn btn-secondary btn-lg">Set up this counter</a>
        </div>
        <div className="hero-facts" aria-label="Key facts">
          <span><Languages size={16} aria-hidden="true" /> 6 Indian languages</span>
          <span><Cpu size={16} aria-hidden="true" /> Runs on the device</span>
          <span><WifiOff size={16} aria-hidden="true" /> Works offline</span>
          <span><ShieldCheck size={16} aria-hidden="true" /> No video uploaded</span>
        </div>
      </div>
      <div className="hero-art"><img src={heroArt} alt="Two people at a service counter under a marigold bridge arch, one of them signing" width={880} height={880} /></div>
    </section>

    <section aria-labelledby="steps-heading">
      <h2 id="steps-heading" className="sr-only">Set up in three steps</h2>
      <div className="steps stagger">
        {[
          ['Choose the language', 'Pick what the hearing person speaks. Text and phrases are exact in all six.', '#language'],
          ['Check the devices', 'Camera for signs, microphone for speech. Text works even if both fail.', '#devices'],
          ['Turn the screen around', 'Both sides share one screen. Switch the view from the top bar.', '#bridge'],
        ].map(([title, description, href], i) => <a className="step" href={href} key={title}><span className="step-n" aria-hidden="true">{i + 1}</span><div><strong>{title}</strong><p>{description}</p></div></a>)}
      </div>
    </section>

    <section aria-labelledby="workspace-heading">
      <div className="section-head" style={{ marginBottom: 18 }}>
        <div><span className="eyebrow">Your workspace</span><h2 id="workspace-heading">Everything for the conversation</h2></div>
        <span className="count">{transcript.length} {transcript.length === 1 ? 'message' : 'messages'} this session</span>
      </div>
      <div className="module-grid stagger">{modules(domain).map(({ view, title, description, cta, icon: Icon }) => <a href={`#${view}`} key={view} className="module"><span className="module-icon"><Icon size={24} aria-hidden="true" /></span><h3>{title}</h3><p>{description}</p><span className="module-cta">{cta}<ArrowRight size={16} aria-hidden="true" /></span></a>)}</div>
    </section>

    {/* What this is, who built it, and what it cannot do.
        The last part is not modesty. Sign recognition is right about 74% of the
        time for someone the model has never seen, and the dictionary
        shortlist behind it about half the time, so anyone using this in a
        clinic needs to know the phrase board is the reliable path and the
        camera is the shortcut. Saying so here is cheaper than a doctor
        discovering it mid-consultation. */}
    <section aria-labelledby="about-heading" className="about">
      <div>
        <span className="eyebrow">About</span>
        <h2 id="about-heading" style={{ fontSize: 'clamp(1.75rem, 1.3rem + 1.6vw, 2.5rem)' }}>Setu means bridge.</h2>
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
      <div className="about-foot">
        <InstallButton />
        <p>
          Built by <strong>Team Awaaz</strong> for Smart India Hackathon 2026.
          Free to use, and always will be.{' '}
          <a href="https://github.com/geetxnshgoyal/sih/blob/main/PRIVACY.md" target="_blank" rel="noopener noreferrer">Privacy</a>
          {' · '}
          <a href="https://github.com/geetxnshgoyal/sih" target="_blank" rel="noopener noreferrer">Source</a>
        </p>
      </div>
    </section>
  </div></main>;
}
