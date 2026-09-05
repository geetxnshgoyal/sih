import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Info, Play, RotateCcw, Square, Volume2 } from "lucide-react";
import { useLandmarkers } from "../hooks/useLandmarkers";
import { GlossClassifier } from "../lib/classifier";
import { StabilityGate, FLOOR, NEEDED } from "../lib/gate";
import { SEQ_LEN, type PointFrame } from "../lib/features";
import { SignSegmenter } from "../lib/segment";
import { UtteranceBuilder, assembleWithSource } from "../lib/sentence";
import { loadGlossTable, sourceLabel, type TranslationSource } from "../lib/glossTranslate";
import { LANGUAGES, phraseFor, speak, refreshVoices, voiceFor, type LangCode } from "../lib/speech";
import { asset } from "../lib/assetUrl";
import { certainty, UNCERTAIN } from "../lib/calibrate";

/** Replay still fills a buffer; the live path is driven by the segmenter. */
const BUFFER = SEQ_LEN * 2;
/** Predict every Nth frame. Landmarks still run every frame so the overlay
 *  stays smooth; inference at ~10Hz is plenty for the stability gate. */
const PREDICT_EVERY = 3;

type Entry = {
  gloss: string;
  text: string;
  conf: number;
  at: string;
  /** Where the spoken text came from — shown so the UI never overstates it. */
  source: TranslationSource;
};

const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const ARM_BONES = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12]];

export default function SignBridge({
  lang: langProp,
  onLang,
  compact = false,
  showDetails = false,
  onRecognized,
}: {
  lang?: LangCode;
  onLang?: (l: LangCode) => void;
  compact?: boolean;
  showDetails?: boolean;
  onRecognized?: (text: string) => void;
} = {}) {
  const recognizedRef = useRef(onRecognized);
  useEffect(() => { recognizedRef.current = onRecognized; }, [onRecognized]);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);

  /** Size the preview box to the stream's real shape.
   *
   * Cameras hand back whatever they like -- 1280x720 is requested, 640x480 is
   * common -- and the same number feeds the model as `aspect`, so the picture
   * and the classifier stay in agreement about what is being looked at. */
  const setStageAspect = useCallback((w: number, h: number) => {
    if (stageRef.current && w > 0 && h > 0) {
      stageRef.current.style.setProperty("--stage-aspect", `${w} / ${h}`);
    }
  }, []);
  const bufferRef = useRef<PointFrame[]>([]);
  const gateRef = useRef(new StabilityGate());
  const clfRef = useRef(new GlossClassifier());
  const rafRef = useRef<number>(0);
  const tickRef = useRef(0);
  const runningRef = useRef(false);
  const frameTimes = useRef<number[]>([]);
  const handFramesRef = useRef(0);
  const segRef = useRef(new SignSegmenter());
  const uttRef = useRef(new UtteranceBuilder());

  const { state: lmState, error: lmError, detect } = useLandmarkers();
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [modelError, setModelError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [ownLang, setOwnLang] = useState<LangCode>("hi-IN");
  const lang = langProp ?? ownLang;
  const setLang = onLang ?? setOwnLang;
  /**
   * Candidates for the sign just segmented, when the top one is not certain.
   *
   * Measured on a held-out signer group: top-1 is right 40.4% of the time, but
   * the correct answer is in the TOP FIVE 65.1% of the time (84% when
   * confidence >= 0.40). Refusing to show anything below a threshold throws
   * that away — and with the calibrated bands it meant showing nothing for 65%
   * of signs, which reads as "the app cannot detect".
   *
   * So: never refuse. Offer the shortlist and let a person choose. That turns a
   * 40%-accurate model into a 65%-useful one without ever claiming certainty.
   */
  const [candidates, setCandidates] = useState<{ gloss: string; conf: number }[]>([]);
  const [live, setLive] = useState<{ gloss: string | null; conf: number; progress: number }>(
    { gloss: null, conf: 0, progress: 0 }
  );
  const [log, setLog] = useState<Entry[]>([]);
  /** Why nothing is being recognised, when the reason is actionable. */
  const [notice, setNotice] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [camError, setCamError] = useState<string | null>(null);
  const [pending, setPending] = useState<string[]>([]);
  const [replaying, setReplaying] = useState<string | null>(null);
  const [diag, setDiag] = useState<{
    pose: boolean; left: boolean; right: boolean;
    shoulder: number; verdict: string; top: { gloss: string; conf: number }[];
  } | null>(null);
  const framingRef = useRef<{ shoulder_width: { p5: number; p50: number; p95: number } } | null>(null);
  const langRef = useRef(lang);
  useEffect(() => { langRef.current = lang; }, [lang]);

  // load the trained classifier
  useEffect(() => {
    let cancelled = false;
    clfRef.current
      .load("/model/model.json", "/model/labels.json")
      .then(() => { if (!cancelled) setModelState("ready"); })
      .catch((e) => {
        if (cancelled) return;
        setModelError(e instanceof Error ? e.message : String(e));
        setModelState("error");
      });
    fetch(asset("/model/_framing.json")).then((r) => r.json())
      .then((f) => { framingRef.current = f; })
      .catch(() => { /* diagnostics are optional */ });
    // Precomputed gloss reorderings. A missing table is a supported state —
    // every utterance then falls back to the phrasebook, as it did before.
    void loadGlossTable();
    refreshVoices();
    window.speechSynthesis?.addEventListener("voiceschanged", refreshVoices);
    return () => {
      cancelled = true;
      window.speechSynthesis?.removeEventListener("voiceschanged", refreshVoices);
    };
  }, []);

  const draw = useCallback((pose: unknown, left: unknown, right: unknown) => {
    const cv = canvasRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const { width: w, height: h } = cv;
    ctx.clearRect(0, 0, w, h);

    const arms = pose as { x: number; y: number }[] | null;
    if (arms) {
      ctx.strokeStyle = "rgba(72,207,171,.45)";
      ctx.lineWidth = 4;
      for (const [a, b] of ARM_BONES) {
        if (!arms[a] || !arms[b]) continue;
        ctx.beginPath();
        ctx.moveTo(arms[a].x * w, arms[a].y * h);
        ctx.lineTo(arms[b].x * w, arms[b].y * h);
        ctx.stroke();
      }
    }
    for (const hand of [left, right] as ({ x: number; y: number }[] | null)[]) {
      if (!hand) continue;
      ctx.strokeStyle = "rgba(72,207,171,.85)";
      ctx.lineWidth = 3;
      for (const [a, b] of HAND_BONES) {
        ctx.beginPath();
        ctx.moveTo(hand[a].x * w, hand[a].y * h);
        ctx.lineTo(hand[b].x * w, hand[b].y * h);
        ctx.stroke();
      }
      ctx.fillStyle = "#2FA9C9";
      for (const p of hand) {
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, []);

  /** Compare this camera's framing to what the model trained on.
   *  INCLUDE signers stood well back: shoulders span ~0.12-0.15 of frame
   *  width. Sitting close at 0.30+ is a domain the model never saw. */
  const framing = useCallback((pose: { x: number; y: number }[] | null) => {
    const f = framingRef.current;
    if (!pose) return { shoulder: 0, verdict: "" };
    const ls = pose[11], rs = pose[12];
    const shoulder = Math.hypot(ls.x - rs.x, ls.y - rs.y);
    if (!f) return { shoulder, verdict: "" };
    const { p5, p95 } = f.shoulder_width;
    const verdict =
      shoulder > p95 * 1.6 ? "close range — model handles this"
      : shoulder > p95 ? "slightly close — fine"
      : shoulder < p5 * 0.5 ? "very far — hands may be too small to track"
      : "matches training framing";
    return { shoulder, verdict };
  }, []);

  /** Speak and log a completed utterance. */
  const emit = useCallback(
    (glosses: string[], at: string, l: LangCode, conf: number) => {
      if (!glosses.length) return;
      const { text, source } = assembleWithSource(glosses, l);
      speak(text, l);
      recognizedRef.current?.(text);
      setLog((prev) => [
        { gloss: glosses.join(" · "), text, conf, at, source },
        ...prev,
      ].slice(0, 40));
    },
    []
  );

  const loop = useCallback(() => {
    if (!runningRef.current) return;
    const video = videoRef.current;
    if (video && video.readyState >= 2) {
      // MediaPipe's coordinates are normalised by frame width and height
      // separately, so they carry the camera's aspect ratio. The model was
      // trained in an isotropic space; feeding it raw MediaPipe output means
      // classifying a body stretched by whatever shape this webcam happens to
      // be. Read it live rather than assuming 16:9 — see lib/features.ts.
      const aspect = video.videoHeight ? video.videoWidth / video.videoHeight : 16 / 9;
      const res = detect(video, performance.now());
      if (res) {
        draw(res.pose, res.left, res.right);
        tickRef.current++;
        const hasHands = !!res.left || !!res.right;

        // Segment first, classify second. The model was trained on clips
        // trimmed to one sign; classifying a rolling window that also contains
        // rest position drops accuracy from 66% to as low as 23%. The
        // segmenter watches hand motion and hands over only the sign itself.
        const segment = segRef.current.push(res.frame, hasHands);
        const seg = segRef.current;

        // A window thrown out for having no hands is worth saying out loud —
        // it is the difference between "the app is broken" and "you are framed
        // wrong", and the user cannot tell those apart from silence.
        if (segRef.current.lastReject === "no-hands") {
          setLive({ gloss: "", conf: 0, progress: 0 });
          setNotice("no hands detected — step back so both hands are in frame");
        } else if (segRef.current.lastReject === "one-hand") {
          // Every training clip is two-handed, so a one-handed window is out of
          // distribution and the model answers confidently anyway (Truck, Car,
          // Mouse at ~0.76). Say what to fix rather than announce a guess.
          setLive({ gloss: "", conf: 0, progress: 0 });
          setNotice("only one hand visible — bring both hands into frame");
        } else if (hasHands) {
          setNotice(null);
        }

        if (tickRef.current % PREDICT_EVERY === 0) {
          const { shoulder, verdict } = framing(res.pose);
          setDiag({
            pose: !!res.pose, left: !!res.left, right: !!res.right,
            shoulder, verdict,
            top: segment ? clfRef.current.predictTop(segment, aspect, 3) : [],
          });
        }

        if (segment) {
          const pred = clfRef.current.predict(segment, aspect);
          const g = gateRef.current.once(pred);
          setLive({ gloss: pred.gloss, conf: pred.conf, progress: 1 });

          // Below the confident band, show what else it considered rather than
          // discarding the sign. The right answer is in this list far more often
          // than it is the top entry.
          setCandidates(
            certainty(pred.conf) === "confident"
              ? []
              : clfRef.current.predictTop(segment, aspect, 5)
          );

          if (g.fire) {
            const l = langRef.current;
            const now = Date.now();
            const finished = uttRef.current.add(g.fire, now, g.conf);
            if (finished) emit(finished.glosses, finished.at, l, finished.conf);
            setPending(uttRef.current.pending);
          }
        } else {
          setLive({ gloss: null, conf: 0, progress: seg.progress });
        }


      }
      const done = uttRef.current.tick(Date.now());
      if (done) {
        // Real confidence, not 1. See UtteranceBuilder.add().
        emit(done.glosses, done.at, langRef.current, done.conf);
        setPending([]);
      }

      const now = performance.now();
      frameTimes.current.push(now);
      while (frameTimes.current.length && now - frameTimes.current[0] > 1000) {
        frameTimes.current.shift();
      }
      if (tickRef.current % 15 === 0) setFps(frameTimes.current.length);
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [detect, draw, framing, emit]);

  /**
   * Replay real ISL clips from the held-out group through the exact same
   * pipeline the camera uses — same buffer, same classifier, same gate, same
   * speech. Only the frame source differs. Doubles as the stage fallback.
   */
  async function replayDemo() {
    const clips: {
      true: string; pred: string; conf: number; correct: boolean;
      frames: number[][][]; aspect?: number;
    }[] = await (await fetch(asset("/model/_demo.json"))).json();

    const cv = canvasRef.current!;
    cv.width = 960; cv.height = 540;
    setStageAspect(960, 540);       // demo clips are INCLUDE's 16:9
    gateRef.current.reset();
    bufferRef.current = [];

    for (const clip of clips) {
      // These are INCLUDE clips, uniformly 1920x1080. Clips written after the
      // isotropic fix carry the field; older ones predate it and were 16:9.
      const aspect = clip.aspect ?? 16 / 9;
      setReplaying(clip.true);
      bufferRef.current = [];
      gateRef.current.reset();

      for (const raw of clip.frames) {
        const frame = raw.map(([x, y, z]) => ({ x, y, z }));
        bufferRef.current.push(frame);
        if (bufferRef.current.length > BUFFER) bufferRef.current.shift();

        drawFrame(frame);

        if (bufferRef.current.length >= SEQ_LEN) {
          const pred = clfRef.current.predict(bufferRef.current, aspect);
          const g = gateRef.current.push(pred);
          setLive({ gloss: pred.gloss, conf: pred.conf, progress: g.progress });
          if (g.fire) {
            const l = langRef.current;
            const text = phraseFor(g.fire, l);
            speak(text, l);
      recognizedRef.current?.(text);
            setLog((prev) => [{
              gloss: g.fire!, text, conf: g.conf,
              at: new Date().toLocaleTimeString(), source: "phrasebook" as const,
            }, ...prev].slice(0, 40));
          }
        }
        await new Promise((r) => setTimeout(r, 45));
      }

      // The clip has ended but the sign is complete in the buffer. Live, the
      // signer holds the final position and the gate keeps sampling; here we
      // do the same so the stability window can fill. Same gate, same floor —
      // not lowering the bar, just giving it the frames it expects.
      for (let i = 0; i < NEEDED + 4 && bufferRef.current.length >= SEQ_LEN; i++) {
        const pred = clfRef.current.predict(bufferRef.current, aspect);
        const g = gateRef.current.push(pred);
        setLive({ gloss: pred.gloss, conf: pred.conf, progress: g.progress });
        if (g.fire) {
          const l = langRef.current;
          const text = phraseFor(g.fire, l);
          speak(text, l);
      recognizedRef.current?.(text);
          setLog((prev) => [{
            gloss: g.fire!, text, conf: g.conf,
            at: new Date().toLocaleTimeString(), source: "phrasebook" as const,
          }, ...prev].slice(0, 40));
          break;
        }
        await new Promise((r) => setTimeout(r, 40));
      }
      await new Promise((r) => setTimeout(r, 600));
    }
    setReplaying(null);
    setLive({ gloss: null, conf: 0, progress: 0 });
  }

  /** Draw one assembled 65-point frame: arms from pose, both hands. */
  function drawFrame(f: { x: number; y: number }[]) {
    const cv = canvasRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const { width: w, height: h } = cv;
    ctx.clearRect(0, 0, w, h);

    // frames are shoulder-centred in unit space; map to canvas with padding
    const xs = f.map((p) => p.x), ys = f.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const sc = Math.min(w / Math.max(maxX - minX, 1e-3), h / Math.max(maxY - minY, 1e-3)) * 0.75;
    const ox = w / 2 - ((minX + maxX) / 2) * sc;
    const oy = h / 2 - ((minY + maxY) / 2) * sc;
    const P = (i: number) => ({ x: f[i].x * sc + ox, y: f[i].y * sc + oy });

    ctx.strokeStyle = "rgba(72,207,171,.45)"; ctx.lineWidth = 5; ctx.lineCap = "round";
    for (const [a, b] of ARM_BONES) {
      const A = P(a), B = P(b);
      ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke();
    }
    for (const base of [23, 44]) {
      ctx.strokeStyle = "rgba(72,207,171,.9)"; ctx.lineWidth = 3;
      for (const [a, b] of HAND_BONES) {
        const A = P(base + a), B = P(base + b);
        if (!A.x && !A.y) continue;
        ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke();
      }
      ctx.fillStyle = "#2FA9C9";
      for (let i = 0; i < 21; i++) {
        const A = P(base + i);
        ctx.beginPath(); ctx.arc(A.x, A.y, 3.4, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  async function start() {
    setCamError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      if (!videoRef.current) { stream.getTracks().forEach(t => t.stop()); return; }
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      if (!canvasRef.current) { stream.getTracks().forEach(t => t.stop()); return; }
      const cv = canvasRef.current;
      setStageAspect(video.videoWidth, video.videoHeight);
      cv.width = video.videoWidth || 1280;
      cv.height = video.videoHeight || 720;
      runningRef.current = true;
      setRunning(true);
      rafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      setCamError(
        `${e instanceof Error ? e.message : String(e)} — the page must be served over http://localhost or https://, and the browser needs camera permission.`
      );
    }
  }

  function stop() {
    runningRef.current = false;
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    const video = videoRef.current;
    (video?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
    if (video) video.srcObject = null;
    bufferRef.current = [];
    gateRef.current.reset();
    handFramesRef.current = 0;
    segRef.current.reset();
    uttRef.current.reset();
    setPending([]);
    setLive({ gloss: null, conf: 0, progress: 0 });
    setFps(0);
  }

  useEffect(() => () => { runningRef.current = false; cancelAnimationFrame(rafRef.current); streamRef.current?.getTracks().forEach(t => t.stop()); }, []);

  const ready = lmState === "ready" && modelState === "ready";
  const status =
    lmState === "error" ? `Landmarker failed: ${lmError}` :
    modelState === "error" ? `Model failed: ${modelError}` :
    !ready ? "Preparing camera tools..." :
    running ? "Listening for signs" : "Ready";

  const voice = voiceFor(lang);

  return (
    <div className={compact ? "bridge compact" : "bridge"}>
      <header>
        <div className="brand">
          <div className="mark">से</div>
          <div>
            <h1>Setu</h1>
            <p>{clfRef.current.vocabulary.length || "..."} signs available</p>
          </div>
        </div>
        <span className={`status ${lmState === "error" || modelState === "error" ? "err" : ready ? "ok" : "busy"}`}>
          <i /> {status}
        </span>
      </header>

      <div className="bridge-content">
        <section className="card stage-card">
          <div className="card-h">
            <span>Signs to speech</span>
            {showDetails && <span className="mono">{fps ? `${fps} fps` : "-"}</span>}
          </div>
          <div className="stage" ref={stageRef}>
            <video ref={videoRef} playsInline muted />
            <canvas ref={canvasRef} />
            {!running && !replaying && (
              <div className="idle">
                <b>{ready ? "Ready to start" : status}</b>
                <span>Start when the signer is framed from face to hands.</span>
              </div>
            )}
            {(running || replaying) && (
              <div className="hud">
                <div>
                  {/* "Current sign" rather than "Detecting": it names what the
                      reader is looking at instead of what the machine is doing. */}
                  <div className="hud-k">Current sign</div>
                  {/* Three bands, not one floor. The classifier no longer drops
                      low-confidence reads silently, so an uncertain one is shown
                      AND marked — the clinician can confirm it instead of the
                      app either announcing a guess or going mysteriously quiet.
                      Hiding it below a threshold meant showing nothing for 65%
                      of signs, which reads as a broken detector rather than an
                      uncertain one. The certainty class is what colours it, so
                      it must survive any restyling. */}
                  <div className={`gloss ${live.gloss ? "" : "none"} ${
                    live.gloss ? certainty(live.conf) : ""
                  }`}>
                    {live.gloss ??
                      (diag && !diag.left && !diag.right ? "hands not visible" : "no sign")}
                  </div>
                  {live.gloss && certainty(live.conf) !== "confident" && (
                    <div className="hud-confirm">
                      {Math.round(live.conf * 100)}% — not certain, pick below if wrong
                    </div>
                  )}
                  <div className="meter">
                    <i style={{
                      width: `${Math.round(live.conf * 100)}%`,
                      background:
                        live.conf >= FLOOR ? "var(--sign)"
                        : live.conf >= UNCERTAIN ? "var(--warn)"
                        : "var(--line-2)",
                    }} />
                  </div>
                  {candidates.length > 0 && (
                    <div className="cands">
                      {candidates.map((c) => (
                        <button
                          key={c.gloss}
                          className={`cand ${c.gloss === live.gloss ? "top" : ""}`}
                          onClick={() => {
                            // A human picking from the shortlist is a CONFIRMED
                            // reading, so it goes straight into the utterance
                            // rather than back through the confidence gate.
                            const l = langRef.current;
                            // 1.0 is correct here and ONLY here: a person tapped this candidate.
                            // Human confirmation is not a model estimate, so it does
                            // not inherit the model's uncertainty.
                            const finished = uttRef.current.add(c.gloss, Date.now(), 1);
                            if (finished) emit(finished.glosses, finished.at, l, c.conf);
                            setPending(uttRef.current.pending);
                            setCandidates([]);
                          }}
                        >
                          {c.gloss}
                          <span className="cand-c">{Math.round(c.conf * 100)}%</span>
                        </button>
                      ))}
                    </div>
                  )}
                  {notice && <div className="hud-notice">{notice}</div>}
                </div>
                {pending.length > 0 && (
                  <div className="pending">
                    {pending.join(" · ")}<span className="caret" />
                  </div>
                )}
                <div className="ring" style={{ ["--p" as string]: live.progress }}>
                  <span>{live.progress >= 1 ? "OK" : "HOLD"}</span>
                </div>
              </div>
            )}
          </div>
          <div className="card-b">
            <div className="row">
              <button className="go" onClick={start} disabled={!ready || running}>
                <Camera size={17} /> Start camera
              </button>
              <button onClick={stop} disabled={!running}>
                <Square size={16} /> Stop
              </button>
              {showDetails && (
                <button onClick={replayDemo} disabled={!ready || running || !!replaying}>
                  <Play size={16} /> {replaying ? replaying : "Play sample"}
                </button>
              )}
              <button onClick={() => setLog([])} disabled={!log.length}>
                <RotateCcw size={16} /> Clear
              </button>
            </div>
            {camError && <div className="err-box">{camError}</div>}
          </div>
        </section>

        <div className="side">
          <section className="card">
            <div className="card-h">
              <span>Spoken output</span>
              <Volume2 size={15} />
            </div>
            <div className="card-b">
              <label className="field">
                <span>Language</span>
                <select value={lang} onChange={(e) => setLang(e.target.value as LangCode)}>
                  {LANGUAGES.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
                </select>
              </label>
              {showDetails && (
                <p className="note">
                  {voice
                    ? <>Voice: <code>{voice.name}</code></>
                    : <>No <code>{lang}</code> voice installed. The browser will use its default.</>}
                </p>
              )}
            </div>
          </section>

          {showDetails && (
            <section className="card details-card">
              <div className="card-h">
                <span>Demo details</span>
                <Info size={15} />
              </div>
              <div className="card-b">
                {!diag && <p className="note">Start the camera to see recognition diagnostics.</p>}
                {diag && (
                  <div className="diag">
                    <div className="drow">
                      <span className={diag.pose ? "on" : "off"}>pose</span>
                      <span className={diag.left ? "on" : "off"}>left hand</span>
                      <span className={diag.right ? "on" : "off"}>right hand</span>
                    </div>
                    <div className="dline">
                      <span>shoulder span</span>
                      <b>{diag.shoulder.toFixed(3)}</b>
                      <em>training range 0.12-0.15</em>
                    </div>
                    {diag.verdict && (
                      <p className={`verdict ${diag.verdict.includes("matches") ? "good" : "bad"}`}>
                        {diag.verdict}
                      </p>
                    )}
                    <div className="dtop">
                      {diag.top.map((t) => (
                        <div key={t.gloss}>
                          <span>{t.gloss}</span>
                          <i style={{ width: `${Math.round(t.conf * 100)}%` }} />
                          <b>{Math.round(t.conf * 100)}%</b>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </section>
          )}

          <section className="card">
            <div className="card-h">Transcript</div>
            <div className="card-b">
              <div className="log">
                {log.length === 0 && <p className="empty">Spoken translations appear here.</p>}
                {log.map((e, i) => (
                  <div className="msg" key={`${e.at}-${i}`}>
                    <div className="t">{e.text}</div>
                    <div className="k">{e.gloss}</div>
                    <div className="m">
                      <span>{Math.round(e.conf * 100)}%</span>
                      {showDetails && (
                        <span
                          className={e.source === "gloss-order" ? "prov warn" : "prov"}
                          title={
                            e.source === "reordered"
                              ? "Reordered into natural spoken word order"
                              : e.source === "phrasebook"
                                ? "From the hand-verified phrase table"
                                : "Signs read in the order they were made - not reordered"
                          }
                        >
                          {sourceLabel(e.source)}
                        </span>
                      )}
                      <span>{e.at}</span>
                      <button className="replay" onClick={() => speak(e.text, lang)}>
                        <Volume2 size={13} /> Replay
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
