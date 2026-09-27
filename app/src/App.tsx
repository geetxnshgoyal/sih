import { LANGUAGES } from "./lib/speech";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowLeftRight,
  Headphones,
  MessageSquare,
  BookOpen,
  Accessibility,
  Hand,
  Volume2,
  Mic,
  Send,
  Check,
  RotateCcw,
  Download,
  X,
  Info,
  ChevronRight,
  WifiOff,
  Type,
} from "lucide-react";
import {
  SessionProvider,
  useSession,
  type Message,
} from "./context/SessionContext";
import {
  phrases,
  phraseText,
  services,
  type ServiceId,
  type Role,
  type Language,
  type Phrase,
} from "./lib/phrases";
import {
  recognitionConstructor,
  speak,
  type Recognition,
} from "./lib/browserSpeech";
import "./App.css";
import "./ServiceWorkspace.css";
const SignBridge = lazy(() => import("./components/SignBridge"));
const Recorder = lazy(() => import("./components/Recorder"));
const SignPlayback = lazy(() => import("./components/SignPlayback"));
import { InstallButton } from "./components/InstallPrompt";
import { ScreenErrorBoundary } from "./components/ScreenErrorBoundary";
type View = "counter" | "phrases" | "access";
function Workspace() {
  const { messages, addMessage, resetSession } = useSession();
  const [showSigns, setShowSigns] = useState(false);
  const [showPlayback, setShowPlayback] = useState(false);
  const [showRecorder, setShowRecorder] = useState(false);
  const [view, setView] = useState<View>(() =>
    location.hash === "#phrases" ? "phrases" : "counter",
  );
  const [service, setService] = useState<ServiceId>("general");
  const currentService = services.find((item) => item.id === service)!;
  const demoSteps = currentService.steps;
  const servicePhrases = phrases.filter(
    (phrase) => phrase.service === service || !phrase.service,
  );
  const [role, setRole] = useState<Role>("customer");
  const [language, setLanguage] = useState<Language>("en-IN");
  const [draft, setDraft] = useState("");
  const [selected, setSelected] = useState<Phrase | null>(null);
  const [source, setSource] = useState<Message["source"]>("Typed text");
  const [notice, setNotice] = useState("");
  const [listening, setListening] = useState(false);
  const [online, setOnline] = useState(navigator.onLine);
  const [large, setLarge] = useState(false);
  const [showReset, setShowReset] = useState(false);
  const [query, setQuery] = useState("");
  const [step, setStep] = useState(0);
  const recognition = useRef<Recognition | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    window.speechSynthesis?.getVoices();
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
      recognition.current?.abort();
      window.speechSynthesis?.cancel();
    };
  }, []);
  useEffect(() => {
    const navigate = () =>
      setView(location.hash === "#phrases" ? "phrases" : "counter");
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  useEffect(() => {
    if (showReset) dialog.current?.showModal();
    else dialog.current?.close();
  }, [showReset]);
  useEffect(() => {
    if (messages.length && end.current?.parentElement) {
      const history = end.current.parentElement;
      history.scrollTop = history.scrollHeight;
    }
  }, [messages]);
  function stopListening() {
    if (recognition.current) {
      recognition.current.onresult = null;
      recognition.current.abort();
      recognition.current = null;
    }
    setListening(false);
  }
  function changeRole(next: Role) {
    stopListening();
    setShowSigns(false);
    setRole(next);
    setDraft("");
    setSelected(null);
    setSource("Typed text");
    setNotice("");
  }
  function changeService(next: ServiceId) {
    stopListening();
    setShowSigns(false);
    setService(next);
    setStep(0);
    setQuery("");
    setNotice(
      "Service changed. Your current draft and earlier messages are kept; select a new phrase when ready.",
    );
  }
  function changeLanguage(next: Language) {
    stopListening();
    setLanguage(next);
    if (selected) setDraft(phraseText(selected, next));
  }
  function choose(phrase: Phrase) {
    stopListening();
    setRole(phrase.role);
    setDraft(phraseText(phrase, language));
    setSelected(phrase);
    setSource("Phrase board");
    setView("counter");
    setNotice("Phrase selected. Review it, then send.");
  }
  function send() {
    if (!draft.trim()) return;
    stopListening();
    addMessage({
      role,
      text: draft.trim(),
      language: selected && language !== "hi-IN" ? "en-IN" : language,
      source,
      phraseId: selected?.id,
      service: services.find(
        (item) => item.id === (selected?.service ?? service),
      )!.label,
    });
    if (selected?.id === demoSteps[step]) setStep((previous) => previous + 1);
    setDraft("");
    setSelected(null);
    setSource("Typed text");
    setNotice(
      `Message sent to the ${role === "customer" ? "staff member" : "service user"}.`,
    );
  }
  function dictate() {
    if (listening) {
      recognition.current?.stop();
      return;
    }
    const Constructor = recognitionConstructor();
    if (!Constructor) {
      setNotice(
        "Speech input is unavailable in this browser. Choose a phrase or type your message.",
      );
      return;
    }
    const rec = new Constructor();
    recognition.current = rec;
    rec.lang = language;
    rec.continuous = false;
    rec.interimResults = false;
    rec.onresult = (event) => {
      setDraft(
        Array.from(event.results)
          .map((result) => result[0].transcript)
          .join(" "),
      );
      setSelected(null);
      setSource("Speech draft");
      setNotice("Speech captured. Review and correct the text before sending.");
    };
    rec.onerror = (event) => {
      setNotice(
        `Speech input could not finish (${event.error}). Type or choose a phrase instead.`,
      );
      setListening(false);
    };
    rec.onend = () => setListening(false);
    try {
      rec.start();
      setListening(true);
      setNotice("Listening. Your browser may use an online speech service.");
    } catch {
      setListening(false);
      setNotice("Microphone could not start. Type or choose a phrase instead.");
    }
  }
  function download() {
    const text = [
      "Setu | Service conversation",
      "Communication only. No bookings, payments or applications are processed.",
      ...messages.map(
        (message) =>
          `${message.role === "customer" ? "SERVICE USER" : "STAFF"} | ${message.timestamp} | ${message.source} | ${message.service}\n${message.text}`,
      ),
    ].join("\n\n");
    const url = URL.createObjectURL(
      new Blob([text], { type: "text/plain;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "setu-conversation.txt";
    anchor.click();
    URL.revokeObjectURL(url);
    setNotice("Conversation text download requested.");
  }
  const latest = messages.at(-1);
  const visiblePhrases = servicePhrases.filter(
    (phrase) =>
      phrase.role === role &&
      `${phrase.en} ${phrase.hi} ${phrase.category}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <div className={`app ${large ? "large-text" : ""}`}>
      <a className="skip" href="#main">
        Skip to conversation
      </a>
      <aside className="sidebar">
        <a className="brand" href="#main" onClick={() => setView("counter")}>
          <span className="brand-mark">से</span>
          <span>
            Setu<span className="brand-caption">A bridge to understanding</span>
          </span>
        </a>
        <div className="workspace-label">YOUR WORKSPACE</div>
        <nav aria-label="Main navigation">
          {(
            [
              { id: "counter", label: "Service counter", icon: ArrowLeftRight },
              { id: "phrases", label: "Phrase library", icon: BookOpen },
              {
                id: "access",
                label: "Access & capabilities",
                icon: Accessibility,
              },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "nav-item active" : "nav-item"}
              aria-current={view === item.id ? "page" : undefined}
              onClick={() => {
                stopListening();
                setShowSigns(false);
                setShowRecorder(false);
                setView(item.id);
              }}
            >
              <item.icon size={20} />
              {item.label}
              {view === item.id && <ChevronRight size={16} />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="project-tag">SIH26199</span>
          <p>Accessible tertiary services</p>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <Headphones size={19} />
            <span>Tertiary services workspace</span>
          </div>
          <button
            className="text-size"
            aria-pressed={large}
            onClick={() => setLarge(!large)}
          >
            <Type size={19} /> Larger text
          </button>
        </header>
        <main id="main" tabIndex={-1}>
          <div className="page-heading">
            <div>
              <h1>
                {view === "counter"
                  ? "Let’s talk."
                  : view === "phrases"
                    ? "The right words, within reach."
                    : "Access, without assumptions."}
              </h1>
              <p className="subtitle">
                {view === "counter"
                  ? "Choose a phrase or write a message."
                  : view === "phrases"
                    ? "Select a service phrase, review it, and share it across the counter."
                    : "Know what works, what depends on your device, and what comes next."}
              </p>
            </div>
            <label className="service-choice">
              <span id="service-type-label">Service type</span>
              <select
                aria-labelledby="service-type-label"
                value={service}
                onChange={(event) =>
                  changeService(event.target.value as ServiceId)
                }
              >
                {services.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {!online && (
            <div className="notice">
              <WifiOff size={20} />
              <span>
                You’re offline. Phrases and text work while this page remains
                open. Offline reload needs a previously cached production build.
              </span>
            </div>
          )}
          {view === "counter" && (
            <>
              <section className="demo-strip" aria-label="Guided demo">
                <div>
                  <div className="demo-label">Optional demo</div>
                  <h2>{currentService.title}</h2>
                  {step > 0 && (
                    <p>
                      {step < demoSteps.length
                        ? `Step ${step + 1} of 5 · ${phrases.find((item) => item.id === demoSteps[step])!.role === "customer" ? "Service user" : "Staff"} selects and sends the next phrase.`
                        : "Conversation complete. Both sides shared real messages using the phrase board."}
                    </p>
                  )}
                </div>
                <button
                  className="dark-button"
                  onClick={() =>
                    step < demoSteps.length
                      ? choose(
                          phrases.find((item) => item.id === demoSteps[step])!,
                        )
                      : setShowReset(true)
                  }
                >
                  {step === 0
                    ? "Start guided demo"
                    : step < demoSteps.length
                      ? "Select next phrase"
                      : "Start again"}
                  <ArrowRight size={17} />
                </button>
              </section>
              <div className="counter-grid">
                <section
                  className="composer panel"
                  aria-label="Message composer"
                >
                  <div className="panel-heading">
                    <div>
                      <h2>Your message</h2>
                      <p>Choose who is communicating</p>
                    </div>
                  </div>
                  <div
                    className="role-switch"
                    role="group"
                    aria-label="Choose speaker"
                  >
                    <button
                      aria-pressed={role === "customer"}
                      className={role === "customer" ? "selected" : ""}
                      onClick={() => changeRole("customer")}
                    >
                      <Hand size={19} /> Service user
                    </button>
                    <button
                      aria-pressed={role === "staff"}
                      className={role === "staff" ? "selected" : ""}
                      onClick={() => changeRole("staff")}
                    >
                      <Headphones size={19} /> Service staff
                    </button>
                  </div>
                  <div className="compose-tools">
                    <h3>
                      <BookOpen size={18} /> Quick phrases
                    </h3>
                    <label className="language">
                      <span className="sr-only">Message language</span>
                      <select
                        value={language}
                        onChange={(event) => {
                          changeLanguage(event.target.value as Language);
                        }}
                      >
                        {LANGUAGES.map((item) => (
                          <option key={item.code} value={item.code}>
                            {item.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  {language !== "en-IN" && language !== "hi-IN" && (
                    <p className="small-note">
                      Written phrases are available in English and Hindi.
                      English phrases are shown and read in English; your
                      selected language applies to speech input and typed
                      messages.
                    </p>
                  )}
                  <div className="quick-phrases">
                    {servicePhrases
                      .filter((item) => item.role === role)
                      .slice(0, 2)
                      .map((item) => (
                        <button
                          key={item.id}
                          aria-pressed={selected?.id === item.id}
                          className={
                            selected?.id === item.id
                              ? "phrase selected-phrase"
                              : "phrase"
                          }
                          onClick={() => choose(item)}
                        >
                          <span lang={language === "hi-IN" ? "hi" : "en"}>
                            {phraseText(item, language)}
                          </span>
                          {selected?.id === item.id ? (
                            <Check size={18} />
                          ) : (
                            <ArrowRight size={18} />
                          )}
                        </button>
                      ))}
                  </div>
                  <button
                    className="inline-link"
                    onClick={() => {
                      setQuery("");
                      setView("phrases");
                    }}
                  >
                    Browse all phrases <ArrowRight size={16} />
                  </button>
                  <div className="draft-label">
                    <label htmlFor="message">Review or type a message</label>
                    <span>{source}</span>
                  </div>
                  <textarea
                    id="message"
                    lang={selected && language !== "hi-IN" ? "en-IN" : language}
                    value={draft}
                    maxLength={1500}
                    placeholder={
                      role === "customer"
                        ? "What would you like to ask the staff member?"
                        : "Write your response to the service user…"
                    }
                    onChange={(event) => {
                      setDraft(event.target.value);
                      setSelected(null);
                      setSource("Typed text");
                    }}
                  />
                  <div className="send-row">
                    <button
                      className="mic-button"
                      onClick={dictate}
                      aria-pressed={listening}
                    >
                      <Mic size={19} />
                      {listening ? "Stop listening" : "Use microphone"}
                    </button>
                    <button
                      className="primary"
                      onClick={send}
                      disabled={!draft.trim()}
                    >
                      <Send size={18} /> Send to{" "}
                      {role === "customer" ? "staff" : "user"}
                    </button>
                  </div>
                  <p className="small-note">
                    Speech needs browser support and may need internet.
                  </p>
                  {role === "customer" && (
                    <div className="sign-tools">
                      <button
                        className="inline-link"
                        onClick={() => setShowSigns(!showSigns)}
                      >
                        <Hand size={18} />
                        {showSigns
                          ? "Close sign recognition"
                          : "Try sign recognition"}{" "}
                        <span className="beta">Experimental</span>
                      </button>
                      {showSigns && (
                        <div className="sign-panel">
                          <p className="small-note">
                            General vocabulary only. Full service conversations
                            are not recognised as complete sentences. Review
                            every result. Camera permission is required.
                          </p>
                          <Suspense
                            fallback={
                              <p role="status">Loading recognition tools…</p>
                            }
                          >
                            <ScreenErrorBoundary>
                              <SignBridge
                                compact
                                confirmBeforeSend
                                lang={language}
                                onLang={changeLanguage}
                                onRecognized={(text) => {
                                  setDraft(text);
                                  setSelected(null);
                                  setSource("Sign recognition draft");
                                  setNotice(
                                    "Recognition result added to your draft. Check it before sending.",
                                  );
                                }}
                              />
                            </ScreenErrorBoundary>
                          </Suspense>
                        </div>
                      )}
                    </div>
                  )}
                </section>
                <section
                  className="receiver panel"
                  aria-label="Shared conversation"
                >
                  <div className="panel-heading">
                    <div>
                      <h2>Shared message</h2>
                    </div>
                  </div>
                  <div
                    className={`message-display ${latest ? "has-message" : ""}`}
                    aria-live="polite"
                    aria-atomic="true"
                  >
                    {latest ? (
                      <>
                        <span className="display-tag">
                          {latest.role === "customer"
                            ? "SERVICE USER → SERVICE STAFF"
                            : "SERVICE STAFF → SERVICE USER"}
                        </span>
                        <p lang={latest.language}>{latest.text}</p>
                        <div className="message-actions">
                          <span>
                            {latest.source} · {latest.timestamp}
                          </span>
                          <button
                            onClick={() =>
                              speak(latest.text, latest.language, setNotice)
                            }
                          >
                            <Volume2 size={18} /> Read aloud
                          </button>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="conversation-symbol">
                          <MessageSquare size={36} />
                          <ArrowLeftRight size={25} />
                        </div>
                        <h3>Your conversation starts here.</h3>
                        <p>Sent messages appear here for both people.</p>
                      </>
                    )}
                  </div>
                  <div className="isl-note">
                    <span className="isl-icon">
                      <Hand size={23} />
                    </span>
                    <div>
                      <h3>
                        Indian Sign Language <span>Experimental</span>
                      </h3>
                      <p>Limited recorded signs, not full ISL translation.</p>
                      <button
                        className="inline-link"
                        onClick={() => setShowPlayback(!showPlayback)}
                      >
                        {showPlayback
                          ? "Hide sign preview"
                          : "Explore recorded signs"}
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </div>
                  {showPlayback && (
                    <div className="playback-panel">
                      <p className="small-note">
                        Word matching only, not a complete ISL translation.
                        Missing words remain text.
                      </p>
                      <Suspense fallback={<p>Loading sign preview…</p>}>
                        <ScreenErrorBoundary>
                          <SignPlayback
                            text={latest?.text ?? ""}
                            messageId={latest?.id}
                          />
                        </ScreenErrorBoundary>
                      </Suspense>
                    </div>
                  )}
                  <div className="history-heading">
                    <h3>
                      Conversation <span>{messages.length}</span>
                    </h3>
                    <button
                      onClick={download}
                      disabled={!messages.length}
                      aria-label="Download conversation"
                    >
                      <Download size={18} />
                    </button>
                  </div>
                  <div
                    className="history"
                    role="log"
                    aria-label="Conversation history"
                  >
                    {!messages.length ? (
                      <p className="empty-history">
                        Your messages will stay here during this session.
                      </p>
                    ) : (
                      messages.map((message) => (
                        <article
                          className={`history-message ${message.role}`}
                          key={message.id}
                        >
                          <div>
                            <strong>
                              {message.role === "customer"
                                ? "Service user"
                                : "Service staff"}
                            </strong>
                            <span>
                              {message.service} · {message.source} ·{" "}
                              {message.timestamp}
                            </span>
                          </div>
                          <p lang={message.language}>{message.text}</p>
                        </article>
                      ))
                    )}
                    <div ref={end} />
                  </div>
                  <div className="session-footer">
                    <span>Session only · Clears on reload</span>
                    <button
                      onClick={() => setShowReset(true)}
                      disabled={!messages.length && !draft}
                    >
                      <RotateCcw size={15} /> New conversation
                    </button>
                  </div>
                </section>
              </div>
              <div className="bottom-note">
                <Info size={17} />
                <p>
                  Communication support only. Setu does not process bookings,
                  payments, account changes or applications, or connect an
                  interpreter.
                </p>
              </div>
            </>
          )}
          {view === "phrases" && (
            <section className="panel library">
              <div className="library-toolbar">
                <div className="role-switch">
                  <button
                    aria-pressed={role === "customer"}
                    className={role === "customer" ? "selected" : ""}
                    onClick={() => changeRole("customer")}
                  >
                    Service user phrases
                  </button>
                  <button
                    aria-pressed={role === "staff"}
                    className={role === "staff" ? "selected" : ""}
                    onClick={() => changeRole("staff")}
                  >
                    Staff phrases
                  </button>
                </div>
                <label>
                  Find a phrase
                  <input
                    type="search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search requests, help…"
                  />
                </label>
              </div>
              <div className="library-grid">
                {visiblePhrases.map((phrase) => (
                  <button
                    className="library-phrase"
                    key={phrase.id}
                    onClick={() => choose(phrase)}
                  >
                    <span>{phrase.category}</span>
                    <strong>{phrase.en}</strong>
                    <p lang="hi">{phrase.hi}</p>
                    <span className="inline-link">
                      Use this phrase <ArrowRight size={16} />
                    </span>
                  </button>
                ))}
              </div>
              {!visiblePhrases.length && (
                <p>
                  No matching phrases. Try another word or type your own message
                  at the service counter.
                </p>
              )}
            </section>
          )}
          {view === "access" && (
            <section className="capabilities">
              <div className="panel capability-intro">
                <Accessibility size={32} />
                <h2>Choose the way that works for you.</h2>
                <p>
                  Large text, visible speaker labels, keyboard controls, and a
                  phrase board support the shared conversation. English and
                  Hindi phrases are written in advance. Custom text is shown
                  exactly as entered.
                </p>
                <button className="primary" onClick={() => setLarge(!large)}>
                  {large ? "Use standard text" : "Use larger text"}
                </button>
              </div>
              <div className="capability-list">
                {[
                  [
                    "Phrase board & typed messages",
                    "Available",
                    "Choose, review, send, and read messages in both directions. No microphone or camera required.",
                  ],
                  [
                    "Read aloud",
                    "Device dependent",
                    "Uses installed browser voices. A missing voice or playback error is reported; visible text remains available.",
                  ],
                  [
                    "Speech to text",
                    recognitionConstructor()
                      ? "Browser API detected"
                      : "Unavailable in this browser",
                    "Requires microphone permission and may use an online service. Review the actual speech draft before sending. No simulated transcript.",
                  ],
                  [
                    "Sign recognition & ISL guidance",
                    "Experimental",
                    "Bundled model, landmark tracking and recorded skeleton playback are retained. Results need review. Full service-specific sentences and ISL grammar are not supported. Live accuracy is unverified.",
                  ],
                  [
                    "Offline use",
                    "Cached production build",
                    "Phrases and text work after the production app has been loaded and cached online. First use needs internet. Recognition and playback need their assets cached separately; speech may use online services.",
                  ],
                  [
                    "Across tertiary services",
                    "Phrase and text support",
                    "General services, banking, hospitality, retail, transport and public services have selectable written phrases. Custom text works in any service setting. No service transactions are processed.",
                  ],
                ].map(([title, status, description]) => (
                  <article className="panel capability" key={title}>
                    <div>
                      <h3>{title}</h3>
                      <span>{status}</span>
                    </div>
                    <p>{description}</p>
                  </article>
                ))}
              </div>
              <div className="panel additional-tools">
                <h2>On this device</h2>
                <InstallButton />
                <button onClick={() => setShowRecorder(!showRecorder)}>
                  {showRecorder
                    ? "Close recording tools"
                    : "Open sign recording tools"}
                </button>
                <p className="small-note">
                  Recording exports landmark data for development. It does not
                  train the model or translate your recording.
                </p>
                {showRecorder && (
                  <Suspense fallback={<p>Loading recording tools…</p>}>
                    <ScreenErrorBoundary>
                      <Recorder />
                    </ScreenErrorBoundary>
                  </Suspense>
                )}
              </div>
            </section>
          )}
          <div className="feedback" role="status">
            {notice}
          </div>
          <footer className="page-footer">
            <span>
              Setu{" "}
              <span className="muted">
                / Connecting people, one conversation at a time.
              </span>
            </span>
            <span>SIH26199 · Tertiary services</span>
          </footer>
        </main>
      </div>
      <dialog
        ref={dialog}
        onCancel={() => setShowReset(false)}
        aria-labelledby="reset-title"
      >
        <button
          className="close-dialog"
          aria-label="Close"
          onClick={() => setShowReset(false)}
        >
          <X size={20} />
        </button>
        <h2 id="reset-title">Start a new conversation?</h2>
        <p>
          This clears all messages and the current draft from this session.
          Download the conversation first if you want a copy.
        </p>
        <div className="dialog-actions">
          <button onClick={() => setShowReset(false)}>Keep conversation</button>
          <button
            className="primary"
            onClick={() => {
              stopListening();
              window.speechSynthesis?.cancel();
              resetSession();
              setStep(0);
              setRole("customer");
              setDraft("");
              setSelected(null);
              setSource("Typed text");
              setShowReset(false);
              setNotice("New conversation ready.");
            }}
          >
            Clear and start
          </button>
        </div>
      </dialog>
    </div>
  );
}
export default function App() {
  return (
    <SessionProvider>
      <Workspace />
    </SessionProvider>
  );
}
