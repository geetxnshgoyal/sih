import { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision';
import wasmLoader from '@mediapipe/tasks-vision/vision_wasm_internal.js?url';
import wasmBinary from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url';
import fallbackLoader from '@mediapipe/tasks-vision/vision_wasm_nosimd_internal.js?url';
import fallbackBinary from '@mediapipe/tasks-vision/vision_wasm_nosimd_internal.wasm?url';
import { assembleFrame, type Landmark, type PointFrame } from '../lib/features';

export type LoadState = 'loading' | 'ready' | 'error';
export type DetectResult = {
  frame: PointFrame; pose: Landmark[] | null; left: Landmark[] | null;
  right: Landmark[] | null; face: Landmark[] | null;
};

/** One Holistic coordinate contract for both training and camera input. */
export function useLandmarkers() {
  const ref = useRef<HolisticLandmarker | null>(null);
  const previousVideo = useRef<HTMLVideoElement | null>(null);
  const previousTime = useRef(-1);
  const timestamp = useRef(0);
  const [state, setState] = useState<LoadState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let cancelled = false;
    let instance: HolisticLandmarker | null = null;
    setState('loading'); setError(null);
    void (async () => {
      try {
        const simd = await FilesetResolver.isSimdSupported();
        const fileset = {
          wasmLoaderPath: simd ? wasmLoader : fallbackLoader,
          wasmBinaryPath: simd ? wasmBinary : fallbackBinary,
        };
        if (cancelled) return;
        const create = (delegate: 'GPU' | 'CPU') => HolisticLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/vision/holistic_landmarker.task', delegate },
          runningMode: 'VIDEO',
        });
        try { instance = await create('GPU'); }
        catch (gpuError) {
          if (cancelled) return;
          console.warn('[camera] GPU initialization failed; trying CPU', gpuError);
          instance = await create('CPU');
        }
        if (cancelled) { instance.close(); return; }
        ref.current = instance;
        previousTime.current = -1; previousVideo.current = null; timestamp.current = 0;
        setState('ready');
      } catch (e) {
        if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); setState('error'); }
      }
    })();
    return () => {
      cancelled = true;
      if (ref.current === instance) ref.current = null;
      instance?.close();
    };
  }, [attempt]);

  const detect = useCallback((video: HTMLVideoElement, tMs: number): DetectResult | null => {
    if (!ref.current || video.readyState < 2) return null;
    // RAF can run at 60/120 Hz while the camera supplies only 15/30 frames.
    // Repeated images used to look like stillness, prematurely ending every sign.
    if (previousVideo.current === video && previousTime.current === video.currentTime) return null;
    previousVideo.current = video; previousTime.current = video.currentTime;
    timestamp.current = Math.max(tMs, timestamp.current + 0.001);
    const result = ref.current.detectForVideo(video, timestamp.current);
    const points = (value: Landmark[][] | undefined) => value?.[0]?.length ? value[0] : null;
    const pose = points(result.poseLandmarks);
    const left = points(result.leftHandLandmarks);
    const right = points(result.rightHandLandmarks);
    const face = points(result.faceLandmarks);
    return { frame: assembleFrame(pose, left, right), pose, left, right, face };
  }, []);
  return { state, error, detect, retry: () => setAttempt(n => n + 1) };
}
