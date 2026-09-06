import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { assembleFrame, type Landmark, type PointFrame } from "../lib/features";

// Keep this in sync with the version in app/package.json, which is pinned
// EXACTLY (no caret) for this reason. The JS wrapper and the Wasm are
// version-coupled: loading mismatched pairs can leave the detector stuck in
// "Preparing..." on real browsers, with no error to explain it.
//
// A caret range here was a live trap. package.json said "^1.0.1" while this URL
// says 1.0.1, so any fresh npm install that resolved 1.1+ would have shipped a
// mismatched pair. The lockfile happened to hold 1.0.1, so nothing broke, but
// nothing was preventing it either.
const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const HOLISTIC_MODEL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

export type LoadState = "loading" | "ready" | "error";

export type DetectResult = {
  frame: PointFrame;
  pose: Landmark[] | null;
  left: Landmark[] | null;
  right: Landmark[] | null;
  face: Landmark[] | null;
};

/**
 * MediaPipe Holistic: one model for pose, both hands, and face.
 *
 * Why Holistic and not PoseLandmarker + HandLandmarker:
 * the INCLUDE pose release was extracted with Holistic, which produces pose and
 * hand landmarks in ONE consistent coordinate frame. Running two separate
 * landmarkers gives two different z conventions, hand z relative to the wrist,
 * pose z relative to the torso, and z carries roughly a third of the model's
 * input signal (pose z std 1.32 and hand z std 0.75, against x,y std of 1.24
 * and 2.32). Mixing conventions makes a third of the live input meaningless to
 * a model trained on Holistic. Same model in, same model out.
 *
 * Holistic also returns the 468-point face mesh, which is what FULL_FACE mode
 * in train/face.py needs for eyebrow and mouth grammar.
 */
export function useLandmarkers() {
  const ref = useRef<HolisticLandmarker | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(CDN);
        let lm: HolisticLandmarker;
        try {
          lm = await HolisticLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: HOLISTIC_MODEL, delegate: "GPU" },
            runningMode: "VIDEO",
          });
        } catch {
          lm = await HolisticLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: HOLISTIC_MODEL, delegate: "CPU" },
            runningMode: "VIDEO",
          });
        }
        if (cancelled) { lm.close(); return; }
        ref.current = lm;
        setState("ready");
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setState("error");
      }
    })();
    return () => { cancelled = true; ref.current?.close(); ref.current = null; };
  }, []);

  function detect(video: HTMLVideoElement, tMs: number): DetectResult | null {
    if (!ref.current) return null;
    const r: HolisticLandmarkerResult = ref.current.detectForVideo(video, tMs);

    const pose = (r.poseLandmarks?.[0] as Landmark[] | undefined) ?? null;
    const left = (r.leftHandLandmarks?.[0] as Landmark[] | undefined) ?? null;
    const right = (r.rightHandLandmarks?.[0] as Landmark[] | undefined) ?? null;
    const face = (r.faceLandmarks?.[0] as Landmark[] | undefined) ?? null;

    return { frame: assembleFrame(pose, left, right), pose, left, right, face };
  }

  return { state, error, detect };
}
