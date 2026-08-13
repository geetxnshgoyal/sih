import { useCallback, useEffect, useRef, useState } from "react";
import { useLandmarkers } from "../hooks/useLandmarkers";
import { SEQ_LEN, type PointFrame } from "../lib/features";

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
const CAPTURE_MS = 2200;

type Take = { gloss: string; frames: PointFrame[]; at: number };

export default function Recorder() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
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

  const loop = useCallback(() => {
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
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      const cv = canvasRef.current!;
      cv.width = v.videoWidth || 1280;
      cv.height = v.videoHeight || 720;
      runningRef.current = true;
      setLive(true);
      rafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      setCamError(e instanceof Error ? e.message : String(e));
    }
  }

  function stop() {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    (videoRef.current?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setLive(false);
  }

  useEffect(() => () => { runningRef.current = false; cancelAnimationFrame(rafRef.current); }, []);

  /** Countdown, then capture a fixed window so every take is comparable. */
  function record() {
    if (!gloss.trim() || !live) return;
    setPhase("counting");
    setCount(COUNTDOWN);
    let n = COUNTDOWN;
    const tick = window.setInterval(() => {
      n -= 1;
      setCount(n);
      if (n > 0) return;
      window.clearInterval(tick);
      bufRef.current = [];
      capturingRef.current = true;
      setPhase("capturing");
      window.setTimeout(() => {
        capturingRef.current = false;
        const frames = bufRef.current.slice();
        setPhase("idle");
        if (frames.length >= SEQ_LEN) {
          setTakes((prev) => [...prev, { gloss: gloss.trim(), frames, at: Date.now() }]);
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
      format: "setu-recordings-v1",
      points: 65,
      note: "unit coordinates, pose 0-22 + left hand 23-43 + right hand 44-64",
      takes: takes.map((t) => ({
        gloss: t.gloss,
        recorded_at: new Date(t.at).toISOString(),
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
      <div className="card">
        <div className="card-h">
          <span>Record your own signs</span>
          <span className="mono">{takes.length} takes</span>
        </div>

        <div className="stage rec-stage">
          <video ref={videoRef} playsInline muted />
          <canvas ref={canvasRef} />
          {!live && <div className="idle">Camera off — press Start to begin recording.</div>}
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
                {state === "ready" ? "Start camera" : "Loading…"}
              </button>
            ) : (
              <button onClick={stop}>Stop camera</button>
            )}
            <input
              className="say"
              value={gloss}
              placeholder="sign label, e.g. Hello"
              onChange={(e) => setGloss(e.target.value)}
            />
            <button
              className="go"
              onClick={record}
              disabled={!live || !gloss.trim() || phase !== "idle"}
            >
              Record take
            </button>
            <button onClick={download} disabled={!takes.length}>
              Download {takes.length ? `(${takes.length})` : ""}
            </button>
          </div>

          {error && <div className="err-box">Landmarker: {error}</div>}
          {camError && <div className="err-box">{camError}</div>}

          <p className="note" style={{ marginTop: 14 }}>
            Aim for <b>30–50 takes per sign</b> across 20–30 signs. Vary your
            distance, angle and lighting between takes — that variety is what the
            INCLUDE data lacks, and it is what makes the model survive your demo
            room instead of one classroom in Chennai.
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
