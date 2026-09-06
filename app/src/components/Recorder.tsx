import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Download, Dot, Square } from "lucide-react";
import { useLandmarkers } from "../hooks/useLandmarkers";
import { SEQ_LEN, type PointFrame } from "../lib/features";

/**
 * Record your own signs, in your room, on your camera.
 *
 * Everything trained so far comes from 7 signers in one Chennai school at one
 * fixed distance. That model reaches ~52% on held-out INCLUDE signers and much
 * less on a laptop webcam, and no amount of augmentation closed the gap , 
 * three attempts each made it measurably worse.
 *
 * Recording here removes the domain gap instead of modelling around it: same
 * camera, same lighting, same distance, same hands as the demo. A 20-30 sign
 * vocabulary recorded this way is a far easier problem than 264 classes of
 * someone else's footage.
 *
 * Output is a JSON file of raw unit-coordinate frames, the same thing
 * features.to_unit() produces from the INCLUDE pickles, so train/preprocess.py
 * ingests it with no new code path.
 */

const COUNTDOWN = 3;
const CAPTURE_MS = 2200;

type Take = { gloss: string; frames: PointFrame[]; at: number; aspect: number };

export default function Recorder() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);
  const countdownRef = useRef<number | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const bufRef = useRef<PointFrame[]>([]);
  const capturingRef = useRef(false);

  const { state, error, detect } = useLandmarkers();
  const [live, setLive] = useState(false);
  const [gloss, setGloss] = useState("");
  const [takes, setTakes] = useState<Take[]>([]);
  const [phase, setPhase] = useState<"idle" | "counting" | "capturing">("idle");
  const [count, setCount] = useState(0);
  const [hands, setHands] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);

  const loop = useCallback(function loop() {
    if (!runningRef.current) return;
    const video = videoRef.current;
    if (video && video.readyState >= 2) {
      const res = detect(video, performance.now());
      if (res) {
        setHands(!!res.left || !!res.right);
        if (capturingRef.current) bufRef.current.push(res.frame);

        const cv = canvasRef.current;
        const ctx = cv?.getContext("2d");
        if (cv && ctx) {
          ctx.clearRect(0, 0, cv.width, cv.height);
          for (const hand of [res.left, res.right]) {
            if (!hand) continue;
            ctx.fillStyle = "#48CFAB";
            for (const p of hand) {
              ctx.beginPath();
              ctx.arc(p.x * cv.width, p.y * cv.height, 3.4, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
      }
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [detect]);

  async function start() {
    setCamError(null);
    if (state !== "ready") {
      setCamError(error ? `Landmarker: ${error}` : "Camera tools are still preparing.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamError("Camera access is unavailable in this browser or page context.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      const v = videoRef.current;
      const cv = canvasRef.current;
      if (!v || !cv) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = stream;
      v.srcObject = stream;
      await v.play();
      cv.width = v.videoWidth || 1280;
      cv.height = v.videoHeight || 720;
      runningRef.current = true;
      setLive(true);
      rafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      setCamError(e instanceof Error ? e.message : String(e));
    }
  }

  function clearTimers() {
    if (countdownRef.current !== null) window.clearInterval(countdownRef.current);
    if (captureTimerRef.current !== null) window.clearTimeout(captureTimerRef.current);
    countdownRef.current = null;
    captureTimerRef.current = null;
  }

  function stop() {
    runningRef.current = false;
    capturingRef.current = false;
    clearTimers();
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    bufRef.current = [];
    setLive(false);
    setHands(false);
    setPhase("idle");
    setCount(0);
  }

  useEffect(() => () => { stop(); }, []);

  /** Countdown, then capture a fixed window so every take is comparable. */
  function record() {
    if (!gloss.trim()) {
      setCamError("Enter a sign label before recording.");
      return;
    }
    if (!live || !runningRef.current) {
      setCamError("Start the camera before recording.");
      return;
    }
    if (phase !== "idle") return;
    setCamError(null);
    setPhase("counting");
    setCount(COUNTDOWN);
    let n = COUNTDOWN;
    countdownRef.current = window.setInterval(() => {
      n -= 1;
      setCount(n);
      if (n > 0) return;
      if (countdownRef.current !== null) window.clearInterval(countdownRef.current);
      countdownRef.current = null;
      bufRef.current = [];
      capturingRef.current = true;
      setPhase("capturing");
      captureTimerRef.current = window.setTimeout(() => {
        captureTimerRef.current = null;
        capturingRef.current = false;
        const frames = bufRef.current.slice();
        setPhase("idle");
        if (frames.length >= SEQ_LEN) {
          // Record the camera's aspect ratio with the take. MediaPipe's
          // coordinates are aspect-dependent, so frames without it cannot be
          // put into the model's coordinate space later: see features.ts.
          const v = videoRef.current;
          const aspect = v && v.videoHeight ? v.videoWidth / v.videoHeight : 16 / 9;
          setTakes((prev) => [...prev, { gloss: gloss.trim(), frames, at: Date.now(), aspect }]);
          setCamError(null);
        } else {
          setCamError(
            `Only ${frames.length} frames captured (need ${SEQ_LEN}). Keep both hands in frame for the whole take.`
          );
        }
      }, CAPTURE_MS);
    }, 700);
  }

  function download() {
    // Frames are already in the unit coordinate space that to_unit() produces,
    // so train/preprocess.py can read this directly.
    const payload = {
      format: "setu-recordings-v2",
      points: 65,
      note: "unit coordinates, pose 0-22 + left hand 23-43 + right hand 44-64; "
        + "per-take `aspect` is the camera's width/height, required to map "
        + "these into the model's isotropic space (v2 added this field)",
      takes: takes.map((t) => ({
        gloss: t.gloss,
        recorded_at: new Date(t.at).toISOString(),
        aspect: t.aspect,
        frames: t.frames.map((f) => f.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)])),
      })),
    };
    const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `setu-recordings-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "")}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const counts = takes.reduce<Record<string, number>>((acc, t) => {
    acc[t.gloss] = (acc[t.gloss] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <section className="recorder">
      <div className="record-lead">
        <p className="eyebrow">Community data collection</p>
        <h2>Collect clean examples from the same camera used in the demo.</h2>
      </div>
      <div className="card">
        <div className="card-h">
          <span>Collect signs</span>
          <span className="mono">{takes.length} takes</span>
        </div>

        <div className="stage rec-stage">
          <video ref={videoRef} playsInline muted />
          <canvas ref={canvasRef} />
          {!live && <div className="idle"><b>Camera is off.</b><span>Start, enter a sign label, then record a take.</span></div>}
          {phase === "counting" && <div className="countdown">{count}</div>}
          {phase === "capturing" && <div className="capturing">RECORDING</div>}
          {live && (
            <div className="rec-hands">
              <span className={hands ? "on" : "off"}>{hands ? "hands visible" : "no hands"}</span>
            </div>
          )}
        </div>

        <div className="card-b">
          <div className="row">
            {!live ? (
              <button className="go" onClick={start} disabled={state !== "ready"}>
                <Camera size={17} /> {state === "ready" ? "Start camera" : "Preparing..."}
              </button>
            ) : (
              <button onClick={stop}><Square size={16} /> Stop camera</button>
            )}
            <input
              className="say"
              value={gloss}
              placeholder="Sign label, e.g. Hello"
              onChange={(e) => setGloss(e.target.value)}
            />
            <button
              className="go"
              onClick={record}
              disabled={!live || !gloss.trim() || phase !== "idle"}
            >
              <Dot size={20} /> Record take
            </button>
            <button onClick={download} disabled={!takes.length}>
              <Download size={17} /> Download {takes.length ? `(${takes.length})` : ""}
            </button>
          </div>

          {error && <div className="err-box">Landmarker: {error}</div>}
          {camError && <div className="err-box">{camError}</div>}

          <p className="note" style={{ marginTop: 14 }}>
            Aim for <b>30-50 takes per sign</b>. Vary distance, angle and lighting
            so the demo learns the room it will be shown in.
          </p>

          {Object.keys(counts).length > 0 && (
            <div className="tally">
              {Object.entries(counts).sort().map(([g, n]) => (
                <span key={g} className={n >= 30 ? "done" : ""}>
                  {g} <b>{n}</b>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
