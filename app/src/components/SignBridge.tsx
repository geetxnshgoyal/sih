import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Info, Play, RotateCcw, Square, Volume2 } from "lucide-react";
import { useLandmarkers } from "../hooks/useLandmarkers";
import { GlossClassifier } from "../lib/classifier";
import { SignBank, type BankMatch } from "../lib/bank";
import { StabilityGate, FLOOR, NEEDED } from "../lib/gate";
import { SEQ_LEN, type PointFrame } from "../lib/features";
import { SignSegmenter, splitRecording, SIGN_GAP_FRAMES } from "../lib/segment";
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

/**
 * The rate the segmenter is fed at, in frames per second.
 *
 * Every corpus here is resampled to 15 fps before a single feature is computed
 * (TARGET_FPS in the extractors). The app was feeding the segmenter on every
 * requestAnimationFrame instead, so on a 60 Hz display it ran four times
 * faster than anything it was tuned against, and every frame-counted threshold
 * silently meant a quarter of what it says:
 *
 *     MIN_FRAMES 12    0.80 s as tuned    0.20 s at 60 Hz
 *     QUIET_FRAMES 6   0.40 s as tuned    0.10 s at 60 Hz
 *     MAX_FRAMES 90    6.0 s as tuned     1.5 s at 60 Hz
 *
 * Worse, motionEnergy measures displacement PER FRAME, so the same physical
 * movement sampled four times as often reads as a quarter of the energy, and
 * START = 0.012 became four times harder to reach. The result is a segmenter
 * that starts late, on the fastest instant of a sign, and is ended 0.1 s later
 * by any hold: it captures a position rather than a movement.
 *
 * Sampling on wall-clock time also makes a fast phone and a slow one behave
 * identically, which frame counting never could.
 */
const CAPTURE_FPS = 15;
const CAPTURE_INTERVAL_MS = 1000 / CAPTURE_FPS;

type Entry = {
  gloss: string;
  text: string;
  conf: number;
  at: string;
  /** Where the spoken text came from, shown so the UI never overstates it. */
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
  confirmBeforeSend = false,
  onRecognized,
}: {
  lang?: LangCode;
  onLang?: (l: LangCode) => void;
  compact?: boolean;
  showDetails?: boolean;
  /** Keep camera guesses out of the conversation until the signer chooses one. */
  confirmBeforeSend?: boolean;
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
  const bankRef = useRef(new SignBank());
  const rafRef = useRef<number>(0);
  const tickRef = useRef(0);
  const lastCaptureRef = useRef(0);
  const runningRef = useRef(false);
  const frameTimes = useRef<number[]>([]);
  const handFramesRef = useRef(0);
  const segRef = useRef(new SignSegmenter());
  /**
   * Record mode: the frames captured between pressing record and stopping.
   *
   * The live segmenter has to decide "has the sign ended?" from what it has
   * seen so far. Measured on five real clips of Thank you and Good Morning it
   * emitted nothing at all, while the same clips classified whole gave
   * Thank you 99% and Morning 97%. Recording removes the guess: the model was
   * trained on clips trimmed to exactly one sign, and a recording is that,
   * with a person choosing the boundaries instead of a motion threshold.
   */
  const recordRef = useRef<PointFrame[]>([]);
  const recordingRef = useRef(false);
  const uttRef = useRef(new UtteranceBuilder());

  const { state: lmState, error: lmError, detect } = useLandmarkers();
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [modelError, setModelError] = useState<string | null>(null);
  const [vocabSize, setVocabSize] = useState(0);
  // The list, not just the count. A fluent signer sitting down in front of this
  // will sign naturally and get nothing, because they will sign words outside
  // an 83-sign vocabulary and in connected sentences rather than one citation
  // form at a time. Showing "83 signs available" without showing WHICH 83 makes
  // that look like a broken detector instead of a stated limit.
  const [vocab, setVocab] = useState<string[]>([]);
  const [showVocab, setShowVocab] = useState(false);
  const [running, setRunning] = useState(false);
  const [ownLang, setOwnLang] = useState<LangCode>("hi-IN");
  const lang = langProp ?? ownLang;
  const setLang = onLang ?? setOwnLang;
  /**
   * Candidates for the sign just segmented, when the top one is not certain.
   *
   * Measured on a held-out signer group: top-1 is right 68.3% of the time, but
   * the correct answer is in the TOP FIVE 89.7% of the time. That gap IS the
   * product. Refusing to show anything below a threshold
   * throws away the 22 points between them, and reads as "the app cannot
   * detect" when in fact it knows and is merely unsure which.
   *
   * So: never refuse. Offer the shortlist and let a person choose. That turns a
   * 40%-accurate model into a 65%-useful one without ever claiming certainty.
   */
  const [candidates, setCandidates] = useState<{ gloss: string; conf: number }[]>([]);
  const [dict, setDict] = useState<BankMatch[]>([]);
  const [recording, setRecording] = useState(false);
  const [recFrames, setRecFrames] = useState(0);
  const [recHands, setRecHands] = useState<"both" | "one" | "none">("none");
  const [tracked, setTracked] = useState({ pose: 0, left: 0, right: 0, face: 0 });
  const [recResult, setRecResult] = useState<
    { gloss: string; conf: number; bothHands: boolean; dict: BankMatch[] }[] | null>(null);
  const [bankSize, setBankSize] = useState(0);
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

  // Load the classifier.
  //
  // One model, 83 signs, covering both the clinical and the travel setting.
  // Two models shipped briefly and it was a mistake: two temperatures, two
  // label sets and two caches to invalidate, and the first release pinned
  // returning users to a stale one through exactly that complexity.
  //
  // The 83 classes still cannot say pain, water or help: those words have one
  // clip each in every source that has them, and one example cannot both teach
  // and examine a class. They are reached a different way. lib/bank.ts holds
  // one reference vector per word and matches the same embedding by cosine
  // distance, which needs one clip instead of fifteen. It is offered as a
  // dictionary shortlist, never as a confident answer, because measured from a
  // corpus its references do not include it is 46% top-1 and 72% top-5.
  //
  // The phrase board remains the exact path and needs no model at all.
  useEffect(() => {
    let cancelled = false;
    clfRef.current
      .load(asset("/model/model.json"), asset("/model/labels.json"))
      .then(() => {
        if (!cancelled) {
          setVocabSize(clfRef.current.vocabulary.length);
          setVocab([...clfRef.current.vocabulary].sort((a, b) =>
            a.localeCompare(b, undefined, { sensitivity: "base" })));
          setModelState("ready");
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setModelError(e instanceof Error ? e.message : String(e));
        setModelState("error");
      });
    bankRef.current
      .load(asset("/model/_bank.json"))
      .then(() => { if (!cancelled) setBankSize(bankRef.current.size); })
      .catch(() => { /* the classifier still works without the dictionary */ });
    fetch(asset("/model/_framing.json")).then((r) => r.json())
      .then((f) => { framingRef.current = f; })
      .catch(() => { /* diagnostics are optional */ });
    // Precomputed gloss reorderings. A missing table is a supported state , 
    // every utterance then falls back to the phrasebook, as it did before.
    void loadGlossTable();
    refreshVoices();
    window.speechSynthesis?.addEventListener("voiceschanged", refreshVoices);
    return () => {
      cancelled = true;
      window.speechSynthesis?.removeEventListener("voiceschanged", refreshVoices);
    };
  }, []);

  /**
   * Draw what MediaPipe is actually tracking.
   *
   * The face mesh is drawn even though the MODEL does not consume it. Two
   * different questions get confused otherwise: "is the tracker seeing me"
   * and "is the tracker using my face". Holistic returns 468 face points every
   * frame and always has; they were simply never rendered, so the tracking
   * looked dead when it was working perfectly.
   *
   * What the model consumes is 65 points: pose 0-22, which INCLUDES the nose,
   * eyes, ears and mouth corners, plus both hands. So head position and
   * orientation do reach it. The 468-point mesh does not: FACE_MODE is
   * HEAD_ONLY because the FULL_FACE ablation measured no gain from it
   * (ARCHITECTURE.md 9), and INCLUDE's pose release carries no mesh to train
   * on in the first place.
   */
  const draw = useCallback((pose: unknown, left: unknown, right: unknown,
                            face: unknown) => {
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

    // Face mesh. Confirmation that tracking is alive, not a claim that the
    // model reads expression.
    //
    // This was drawn at alpha 0.34 and radius 1.1, which is invisible against a
    // bright room, and the tracker was reported as "not capturing face points"
    // when it was returning all 468 every frame. Legibility on a washed-out
    // camera image is the whole job here, so: every 2nd point, larger, opaque
    // enough to survive a white background, with a dark halo so it reads on
    // pale skin and pale walls alike.
    const mesh = face as { x: number; y: number }[] | null;
    if (mesh && mesh.length) {
      ctx.strokeStyle = "rgba(0,0,0,.30)";
      ctx.lineWidth = 0.6;
      ctx.fillStyle = "rgba(72,207,171,.85)";
      for (let i = 0; i < mesh.length; i += 2) {
        const p = mesh[i];
        if (!p) continue;
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, 1.7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
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
      shoulder > p95 * 1.6 ? "close range, model handles this"
      : shoulder > p95 ? "slightly close, fine"
      : shoulder < p5 * 0.5 ? "very far, hands may be too small to track"
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

  /**
   * Stop recording and read back what was signed.
   *
   * One recording can hold one sign or several. splitRecording cuts it at
   * pauses of 1.2 s or longer, which is the only threshold measured that keeps
   * a compound sign whole while still separating two signs (see
   * SIGN_GAP_FRAMES). Each piece is then exactly the shape of a training clip,
   * which is the condition the model's 68.3% top-1 was measured under.
   */
  const finishRecording = useCallback(() => {
    recordingRef.current = false;
    setRecording(false);
    const frames = recordRef.current;
    recordRef.current = [];
    setRecFrames(0);

    const video = videoRef.current;
    const aspect = video && video.videoHeight
      ? video.videoWidth / video.videoHeight : 16 / 9;

    const pieces = splitRecording(frames);
    if (!pieces.length) {
      setRecResult(null);
      setNotice(
        frames.length < SIGN_GAP_FRAMES
          ? "too short. Hold record while you sign, then stop"
          : "no hands found in that recording. Step back so both hands are in frame"
      );
      return;
    }

    const reads = pieces.map((piece) => {
      const pred = clfRef.current.predict(piece, aspect);
      const e = clfRef.current.embed(piece, aspect);
      // The SAME guard the live path applies, which record mode was missing.
      // Measured on 400 clips against this model, zeroing the hand landmarks:
      //
      //   both hands      conf 0.73, 53 distinct answers
      //   no hands        conf 0.75, 15 distinct, 'we' 176/400, 'Month' 167/400
      //   right missing   conf 0.77, 18 distinct, 'Restaurant' 210/400
      //
      // Confidence is as high or HIGHER when the hands are gone, so no
      // threshold on it can catch this. Counting hands is the only thing that
      // can, and a read that fails must never be spoken.
      const bothHands = SignSegmenter.hasBothHands(piece);
      return {
        gloss: pred.gloss ?? "",
        conf: pred.conf,
        bothHands,
        dict: e && bankRef.current.ready ? bankRef.current.lookup(e, 3) : [],
      };
    }).filter((r) => r.gloss);

    setRecResult(reads);
    const speakable = reads.filter((r) => r.bothHands);
    if (!speakable.length) {
      setNotice(
        "both hands were not in frame for that sign, so it was not read aloud. " +
        "Step back and keep both hands visible."
      );
      return;
    }
    setNotice(speakable.length < reads.length
      ? "some signs had a hand out of frame and were left out"
      : null);
    const mean = speakable.reduce((a, r) => a + r.conf, 0) / speakable.length;
    emit(speakable.map((r) => r.gloss), new Date().toLocaleTimeString(),
         langRef.current, mean);
  }, [emit]);

  const toggleRecord = useCallback(() => {
    if (recordingRef.current) { finishRecording(); return; }
    recordRef.current = [];
    setRecFrames(0);
    setRecResult(null);
    setNotice(null);
    setCandidates([]);
    setDict([]);
    segRef.current.reset();
    setRecHands("none");
    recordingRef.current = true;
    setRecording(true);
  }, [finishRecording]);

  const loop = useCallback(function loop() {
    if (!runningRef.current) return;
    const video = videoRef.current;
    if (video && video.readyState >= 2) {
      // MediaPipe's coordinates are normalised by frame width and height
      // separately, so they carry the camera's aspect ratio. The model was
      // trained in an isotropic space; feeding it raw MediaPipe output means
      // classifying a body stretched by whatever shape this webcam happens to
      // be. Read it live rather than assuming 16:9, see lib/features.ts.
      const aspect = video.videoHeight ? video.videoWidth / video.videoHeight : 16 / 9;
      const nowMs = performance.now();
      const res = detect(video, nowMs);
      if (res) {
        // Draw at the display's rate, sample at the corpus's rate. The overlay
        // should look smooth; the segmenter must see the same 15 fps every
        // training clip was resampled to. See CAPTURE_FPS.
        draw(res.pose, res.left, res.right, res.face);
        if (tickRef.current % 8 === 0) {
          setTracked({
            pose: res.pose?.length ?? 0, left: res.left?.length ?? 0,
            right: res.right?.length ?? 0, face: res.face?.length ?? 0,
          });
        }
        const due = nowMs - lastCaptureRef.current >= CAPTURE_INTERVAL_MS;
        if (!due) {
          rafRef.current = requestAnimationFrame(loop);
          return;
        }
        lastCaptureRef.current = nowMs;
        tickRef.current++;
        const hasHands = !!res.left || !!res.right;

        // Record mode owns the frames. The automatic segmenter is not merely
        // unnecessary here, it is the thing being replaced, so it does not run
        // at all and cannot emit a competing answer mid-recording.
        if (recordingRef.current) {
          recordRef.current.push(res.frame);
          setRecFrames(recordRef.current.length);
          // Live, not after the fact. Finding out that a hand was out of frame
          // once the recording is already spoiled is the actual failure: the
          // model answers anyway, at full confidence, from an attractor class.
          setRecHands(res.left && res.right ? "both"
                    : res.left || res.right ? "one" : "none");
          rafRef.current = requestAnimationFrame(loop);
          return;
        }

        // Segment first, classify second. The model was trained on clips
        // trimmed to one sign; classifying a rolling window that also contains
        // rest position drops accuracy from 66% to as low as 23%. The
        // segmenter watches hand motion and hands over only the sign itself.
        const segment = segRef.current.push(res.frame, hasHands);
        const seg = segRef.current;
        const reject = seg.lastReject;

        // A window thrown out for having no hands is worth saying out loud , 
        // it is the difference between "the app is broken" and "you are framed
        // wrong", and the user cannot tell those apart from silence.
        if (reject === "no-hands") {
          setLive({ gloss: "", conf: 0, progress: 0 });
          setNotice("no hands detected. Step back so both hands are in frame");
        } else if (reject === "one-hand") {
          setNotice("only one hand visible. Confirm the sign below or bring both hands into frame");
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
          // A segment whose shoulders were never found cannot be classified:
          // everything downstream is anchored on them, and the model answers
          // confidently about nothing when they are missing. This is an else
          // rather than an early return because requestAnimationFrame(loop)
          // is at the BOTTOM of this function: returning here would stop the
          // camera loop for good.
          if (!clfRef.current.usable(segment, aspect)) {
            setLive({ gloss: "", conf: 0, progress: 0 });
            setCandidates([]);
            setDict([]);
            setNotice("your shoulders are not in frame. Step back so your head and both shoulders are visible");
          } else {
            const pred = clfRef.current.predict(segment, aspect);
            const confirmOnly = confirmBeforeSend || reject === "one-hand";
            const g = confirmOnly ? { fire: null, conf: pred.conf, progress: 1 } : gateRef.current.once(pred);
            setLive({ gloss: pred.gloss, conf: pred.conf, progress: 1 });

            // Below the confident band, show what else it considered rather than
            // discarding the sign. The right answer is in this list far more often
            // than it is the top entry.
            const unsure = certainty(pred.conf) !== "confident" || confirmOnly;
            setCandidates(unsure ? clfRef.current.predictTop(segment, aspect, 5) : []);

            // The dictionary runs on EVERY segment, not only when the
            // classifier doubts itself.
            //
            // The first version gated it on low confidence, which sounds right
            // and is wrong. A closed-set classifier cannot answer "not one of
            // mine": asked to read a sign outside its 83 it must still pick one,
            // and softmax is perfectly capable of being certain about it.
            // Measured on real clips of words it was never trained on, it said
            // "Man" at 86% for water and "Alright" at 74% for help, both in the
            // confident band, both spoken aloud, and both with the dictionary
            // suppressed precisely because it was confident. The dictionary had
            // water and help right.
            //
            // Score cannot arbitrate either. Where the dictionary is right the
            // median top-1 cosine is 0.832 and where it is wrong it is 0.780,
            // so any threshold that keeps most correct answers is barely better
            // than a coin toss. Nothing here can tell the two apart, so nothing
            // here pretends to: both readings are shown and a person decides.
            if (bankRef.current.ready) {
              const e = clfRef.current.embed(segment, aspect);
              setDict(e ? bankRef.current.lookup(e, 4) : []);
            } else {
              setDict([]);
            }

            if (g.fire) {
              const l = langRef.current;
              const now = Date.now();
              const finished = uttRef.current.add(g.fire, now, g.conf);
              if (finished) emit(finished.glosses, finished.at, l, finished.conf);
              setPending(uttRef.current.pending);
            }
          }
        } else {
          // In confirmation mode the last completed reading is a pending
          // choice. Keep it visible with its shortlist instead of showing the
          // contradictory combination from the old UI: "no sign" beside a
          // stale predicted-word button.
          setLive((previous) => confirmBeforeSend && previous.gloss
            ? previous
            : { gloss: null, conf: 0, progress: seg.progress });
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
  }, [detect, draw, framing, emit, confirmBeforeSend]);

  /**
   * Replay real ISL clips from the held-out group through the exact same
   * pipeline the camera uses: same buffer, same classifier, same gate, same
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
      // do the same so the stability window can fill. Same gate, same floor , 
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
        `${e instanceof Error ? e.message : String(e)}. The page must be served over http://localhost or https://, and the browser needs camera permission.`
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
    lastCaptureRef.current = 0;
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
            <button
              className="vocab-toggle"
              onClick={() => setShowVocab((v) => !v)}
              aria-expanded={showVocab}
              disabled={!vocabSize}
            >
              {vocabSize || "..."} signs available{vocabSize ? (showVocab ? " (hide)" : " (see the list)") : ""}
            </button>
          </div>
        </div>
        <span className={`status ${lmState === "error" || modelState === "error" ? "err" : ready ? "ok" : "busy"}`}>
          <i /> {status}
        </span>
      </header>

      {showVocab && (
        <section className="vocab-list" aria-label="Signs this model knows">
          <p className="vocab-note">
            These are the only signs recognition can produce. Anything else,
            and any sentence signed continuously rather than one sign at a
            time, will not be recognised. For everything else, use the phrase
            board below, which is exact.
          </p>
          <div className="vocab-grid">
            {vocab.map((v) => <span key={v}>{v}</span>)}
          </div>
        </section>
      )}

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
                  <div className="hud-k">{confirmBeforeSend ? "Possible sign" : "Current sign"}</div>
                  {/* Three bands, not one floor. The classifier no longer drops
                      low-confidence reads silently, so an uncertain one is shown
                      AND marked: the clinician can confirm it instead of the
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
                  {live.gloss && confirmBeforeSend && (
                    <div className="hud-confirm">
                      Choose the correct sign below before it is sent
                    </div>
                  )}
                  {live.gloss && !confirmBeforeSend && certainty(live.conf) !== "confident" && (
                    <div className="hud-confirm">
                      {Math.round(live.conf * 100)}% confident. Pick below if this is wrong
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
                            setDict([]);
                            setLive({ gloss: null, conf: 0, progress: 0 });
                          }}
                        >
                          {c.gloss}
                          <span className="cand-c">{Math.round(c.conf * 100)}%</span>
                        </button>
                      ))}
                    </div>
                  )}
                  {dict.length > 0 && (
                    <div className="dict">
                      <div className="dict-head">
                        closest of {bankSize} dictionary signs, beyond the {vocabSize} above
                      </div>
                      <div className="cands">
                        {dict.map((d) => (
                          <button
                            key={d.word}
                            className="cand dict-cand"
                            onClick={() => {
                              const l = langRef.current;
                              const finished = uttRef.current.add(d.word, Date.now(), 1);
                              if (finished) emit(finished.glosses, finished.at, l, 1);
                              setPending(uttRef.current.pending);
                              setCandidates([]);
                              setDict([]);
                              setLive({ gloss: null, conf: 0, progress: 0 });
                            }}
                          >
                            {d.word}
                            <span className="cand-c">{Math.round(d.score * 100)}</span>
                          </button>
                        ))}
                      </div>
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
                  <span>{confirmBeforeSend && live.gloss ? "PICK" : live.progress >= 1 ? "OK" : "HOLD"}</span>
                </div>
              </div>
            )}
          </div>
          <div className="card-b">
            <div className="row">
              <button className="go" onClick={start} disabled={!ready || running}>
                <Camera size={17} /> Start camera
              </button>
              <button
                className={recording ? "rec on" : "rec"}
                onClick={toggleRecord}
                disabled={!running}
                title="Record a sign, then stop. Pause about a second between signs."
              >
                {recording
                  ? `Stop and read (${(recFrames / CAPTURE_FPS).toFixed(1)}s)`
                  : "Record a sign"}
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
            <p className="tracked">
              tracking: pose {tracked.pose} · left hand {tracked.left} ·
              right hand {tracked.right} · face {tracked.face} points
              {tracked.face > 0
                ? " (face is tracked and drawn; the model reads head position, not expression)"
                : ""}
            </p>
            {recording && (
              <p className={`rec-hint${recHands === "both" ? " ok" : " bad"}`}>
                {recHands === "both"
                  ? "Both hands visible. Sign, then press stop. Pause about a second between signs."
                  : recHands === "one"
                  ? "Only ONE hand visible. Bring the other into frame, or this will not be read."
                  : "Hands NOT visible. Step back so both hands are in the picture."}
                {` · ${(recFrames / CAPTURE_FPS).toFixed(1)}s`}
              </p>
            )}
            {recResult && !recording && (
              <div className="rec-out">
                <div className="rec-out-head">
                  read {recResult.length === 1 ? "1 sign" : `${recResult.length} signs`}
                </div>
                {recResult.map((r, i) => (
                  <div className={`rec-row${r.bothHands ? "" : " weak"}`} key={i}>
                    <span className="rec-n">{i + 1}</span>
                    <span className="rec-g">{r.gloss}</span>
                    <span className="cand-c">{Math.round(r.conf * 100)}%</span>
                    {!r.bothHands && <span className="rec-warn">one hand only, not spoken</span>}
                    {r.dict.length > 0 && (
                      <span className="rec-d">
                        or {r.dict.map((d) => d.word).join(", ")}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
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
