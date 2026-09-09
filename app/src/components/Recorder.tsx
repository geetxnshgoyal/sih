import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Download, Dot, Square } from "lucide-react";
import { useLandmarkers } from "../hooks/useLandmarkers";
import { type PointFrame } from "../lib/features";
import type { FaceFrame } from "../lib/face";
import { segmentQuality } from "../lib/segment";

/**
 * Record your own signs, in your room, on your camera.
 *
 * Everything trained so far comes from 7 signers in one Chennai school at one
 * fixed distance. That model reaches ~52% on held-out INCLUDE signers and much
 * less on a laptop webcam, and no amount of augmentation closed the gap —
 * three attempts each made it measurably worse.
 *
 * Recording here removes the domain gap instead of modelling around it: same
 * camera, same lighting, same distance, same hands as the demo. A 20-30 sign
 * vocabulary recorded this way is a far easier problem than 264 classes of
 * someone else's footage.
 *
 * Output is a JSON file of raw unit-coordinate frames — the same thing
 * features.to_unit() produces from the INCLUDE pickles, so train/preprocess.py
 * ingests it with no new code path.
 */

const COUNTDOWN = 3;
// Please starts at the lips, travels down, then finishes with a short shake.
// A 2.2 s window clipped that ending motion on slower signers, producing a
// training example that looked like a static pose. Keep the full gesture.
const CAPTURE_MS = 3200;

type CaptureFrame = { body: PointFrame; face: FaceFrame | null };
type Take = { gloss: string; frames: PointFrame[]; face: (FaceFrame | null)[]; at: number; aspect: number; durationMs: number };

export default function Recorder() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const runningRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);
  const countdownRef = useRef(0);
  const endRef = useRef(0);
  const generation = useRef(0);
  const [starting, setStarting] = useState(false);
  const bufRef = useRef<CaptureFrame[]>([]);
  const capturingRef = useRef(false);

  const { state, error, detect } = useLandmarkers();
  const [live, setLive] = useState(false);
  const [gloss, setGloss] = useState("");
  const [takes, setTakes] = useState<Take[]>([]);
  const [phase, setPhase] = useState<"idle" | "counting" | "capturing">("idle");
  const [count, setCount] = useState(0);
  const [hands, setHands] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);

  const loop = useCallback(function recordFrame() {
    if (!runningRef.current) return;
    const video = videoRef.current;
    if (video && video.readyState >= 2) {
      let res;
      try { res = detect(video, performance.now()); }
      catch (error) {
        runningRef.current = false; capturingRef.current = false;
        streamRef.current?.getTracks().forEach(t => t.stop());
        clearInterval(countdownRef.current); clearTimeout(endRef.current);
        setLive(false); setPhase('idle'); setCamError(String(error)); return;
      }
      if (res) {
        setHands(!!res.left || !!res.right);
        if (capturingRef.current) bufRef.current.push({ body: res.frame, face: res.faceFrame });

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
    rafRef.current = requestAnimationFrame(recordFrame);
  }, [detect]);

  async function start() {
    if (starting || runningRef.current) return;
    const current = ++generation.current;
    setStarting(true); setCamError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      if (current !== generation.current || !videoRef.current) { stream.getTracks().forEach(t => t.stop()); return; }
      streamRef.current = stream;
      const v = videoRef.current;
      v.srcObject = stream;
      await v.play();
      if (current !== generation.current || !canvasRef.current) { stream.getTracks().forEach(t => t.stop()); return; }
      const cv = canvasRef.current;
      cv.width = v.videoWidth || 1280;
      cv.height = v.videoHeight || 720;
      runningRef.current = true;
      setLive(true); setStarting(false);
      rafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      streamRef.current?.getTracks().forEach(t => t.stop()); setStarting(false);
      setCamError(e instanceof Error ? e.message : String(e));
    }
  }

  function stop() {
    generation.current++; runningRef.current = false; capturingRef.current = false;
    clearInterval(countdownRef.current); clearTimeout(endRef.current);
    setPhase('idle'); setStarting(false);
    cancelAnimationFrame(rafRef.current);
    (videoRef.current?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setLive(false);
  }

  useEffect(() => () => {
    generation.current++; runningRef.current = false; capturingRef.current = false;
    cancelAnimationFrame(rafRef.current); clearInterval(countdownRef.current); clearTimeout(endRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
  }, []);

  /** Countdown, then capture a fixed window so every take is comparable. */
  function record() {
    if (!gloss.trim() || !live) return;
    setPhase("counting");
    setCount(COUNTDOWN);
    let n = COUNTDOWN;
    const tick = countdownRef.current = window.setInterval(() => {
      n -= 1;
      setCount(n);
      if (n > 0) return;
      window.clearInterval(tick);
      bufRef.current = [];
      capturingRef.current = true;
      setPhase("capturing");
      endRef.current = window.setTimeout(() => {
        capturingRef.current = false;
        const captured = bufRef.current.slice();
        const frames = captured.map(frame => frame.body);
        setPhase("idle");
        const rejected = segmentQuality(frames);
        if (!rejected || rejected === 'one-hand') {
          const video = videoRef.current;
          const aspect = video?.videoHeight ? video.videoWidth / video.videoHeight : 0;
          if (!aspect) { setCamError('Camera dimensions are unavailable. Record the take again.'); return; }
          setTakes((prev) => [...prev, {
            gloss: gloss.trim(), frames,
            face: captured.map(frame => frame.face),
            at: Date.now(), aspect, durationMs: CAPTURE_MS,
          }]);
          setCamError(rejected === 'one-hand' ? 'Saved with one hand tracked. Verify that this is an intentional one-handed sign.' : null);
        } else {
          setCamError(
            `Recording rejected (${rejected}, ${frames.length} frames). Keep shoulders and both hands visible, including the resting hand.`
          );
        }
      }, CAPTURE_MS);
    }, 1000);
  }

  function download() {
    // Frames are already in the unit coordinate space that to_unit() produces,
    // so train/preprocess.py can read this directly.
    const payload = {
      format: "setu-recordings-v3",
      points: 65,
      facePoints: 48,
      note: "unit coordinates; body is pose 0-22 + left hand 23-43 + right hand 44-64; face is eyebrows, eyes and lips",
      takes: takes.map((t) => ({
        gloss: t.gloss,
        aspect: t.aspect,
        durationMs: t.durationMs,
        recorded_at: new Date(t.at).toISOString(),
        faceAvailable: t.face.some(Boolean),
        body: t.frames.map((f) => f.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)])),
        face: t.face.map((f) => f?.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)]) ?? null),
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
        <p className="muted">For moving signs such as Please, keep the full path in frame: start at the lips, move down, and finish the finger shake before the capture ends.</p>
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
              <button className="go" onClick={start} disabled={state !== "ready" || starting}>
                <Camera size={17} /> {starting ? "Opening…" : state === "ready" ? "Start camera" : "Preparing..."}
              </button>
            ) : (
              <button onClick={stop}><Square size={16} /> Stop camera</button>
            )}
            <input
              className="say"
              value={gloss}
              placeholder="Sign label, e.g. Please"
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
