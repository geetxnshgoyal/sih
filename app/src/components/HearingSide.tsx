import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, SendHorizontal, Square } from "lucide-react";
import SignPlayer from "./SignPlayer";
import { createRecogniser, textToGlosses, type SignLibrary } from "../lib/reverse";
import { LANGUAGES, type LangCode } from "../lib/speech";
import { asset } from "../lib/assetUrl";

/**
 * Direction B: the hearing person's half of the conversation.
 *
 * They speak; it is transcribed, matched to signs we can play, and shown to
 * the deaf user as an animated skeleton plus large text. Text is not a
 * fallback here: many deaf users read the spoken language, and it carries the
 * words we have no sign for.
 */
export default function HearingSide({
  lang,
  /** Setting-specific spoken-word mappings, layered over the shared map. */
  synonyms = {},
}: {
  lang: LangCode;
  synonyms?: Record<string, string>;
}) {
  const [library, setLibrary] = useState<SignLibrary>({});
  const [listening, setListening] = useState(false);
  const [heard, setHeard] = useState("");
  const [queue, setQueue] = useState<string[]>([]);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [at, setAt] = useState(0);
  const [supported, setSupported] = useState(true);
  const [typed, setTyped] = useState("");
  const recogRef = useRef<SpeechRecognition | null>(null);
  const timerRef = useRef<number>(0);

  useEffect(() => {
    fetch(asset("/model/_signs.json"))
      .then((r) => r.json())
      .then(setLibrary)
      .catch(() => setLibrary({}));
    setSupported(!!createRecogniser(lang));
    return () => { window.clearTimeout(timerRef.current); };
  }, [lang]);

  /** Walk the matched glosses, holding each long enough to be read. */
  const play = useCallback((glosses: string[]) => {
    window.clearTimeout(timerRef.current);
    setQueue(glosses);
    setAt(0);
    if (!glosses.length) return;
    let i = 0;
    const step = () => {
      i += 1;
      if (i >= glosses.length) return;
      setAt(i);
      timerRef.current = window.setTimeout(step, 1900);
    };
    timerRef.current = window.setTimeout(step, 1900);
  }, []);

  const submit = useCallback((text: string) => {
    setHeard(text);
    const { matched, skipped } = textToGlosses(text, library, synonyms);
    setSkipped(skipped);
    play(matched);
  }, [library, play, synonyms]);

  function listen() {
    const r = createRecogniser(lang);
    if (!r) { setSupported(false); return; }
    recogRef.current = r;
    setListening(true);
    setHeard("");
    r.onresult = (e: SpeechRecognitionEvent) => {
      const text = Array.from(e.results).map((res) => res[0].transcript).join(" ");
      setHeard(text);
      if (e.results[e.results.length - 1].isFinal) submit(text);
    };
    r.onerror = () => setListening(false);
    r.onend = () => setListening(false);
    r.start();
  }

  function stopListening() {
    recogRef.current?.stop();
    setListening(false);
  }

  const current = queue[at];
  const frames = current ? library[current] ?? null : null;
  const langLabel = LANGUAGES.find((l) => l.code === lang)?.label ?? lang;

  return (
    <section className="card hearing">
      <div className="card-h">
        <span>Speech to signs</span>
        <span className="mono">{langLabel}</span>
      </div>

      <div className="hearing-stage">
        <SignPlayer frames={frames} />
        <div className="hearing-gloss">
          {current ? (
            <>
              <span className="k">{current}</span>
              <span className="pos">{at + 1} / {queue.length}</span>
            </>
          ) : (
            <span className="idle-note">
              {heard ? "No matching sign in the current library." : "Speak or type. The response appears here as signs."}
            </span>
          )}
          {queue.length > 0 && (
            <div className="queue-pills" aria-label="Matched signs">
              {queue.map((g, i) => (
                <span key={`${g}-${i}`} className={i === at ? "on" : ""}>{g}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="card-b">
        <div className="input-row">
          {supported ? (
            <button className={listening ? "" : "go"} onClick={listening ? stopListening : listen}>
              {listening ? <Square size={16} /> : <Mic size={17} />}
              {listening ? "Stop" : "Speak"}
            </button>
          ) : null}
          <div className="type-submit">
            <input
              className="say"
              value={typed}
              placeholder="Type what you would say"
              onChange={(e) => setTyped(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && typed.trim()) { submit(typed); setTyped(""); } }}
            />
            <button
              className="icon-command send"
              onClick={() => { if (typed.trim()) { submit(typed); setTyped(""); } }}
              disabled={!typed.trim()}
              aria-label="Send typed phrase"
              title="Send"
            >
              <SendHorizontal size={17} />
            </button>
          </div>
        </div>

        {heard && <p className="heard">“{heard}”</p>}

        {skipped.length > 0 && (
          <p className="note skipped">
            No sign for: {skipped.join(", ")}. Shown as text only.
          </p>
        )}

        {!supported && (
          <p className="note">
            This browser has no speech recognition. Typing works the same way.
          </p>
        )}
      </div>
    </section>
  );
}
