"""
Parity test: train/features.py  ==  app/src/lib/features.ts

The single highest-value test in this project. If these two extractors drift,
the model trains on one representation and runs on another. Offline accuracy
stays high, live accuracy collapses, and nothing in the logs says why.

Runs both implementations on identical random input and asserts the outputs
match to float32 precision.

    python train/test_parity.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TS_FEATURES = ROOT / "app" / "src" / "lib" / "features.ts"

# Frame counts chosen to exercise the resampler: shorter than SEQ_LEN, equal to
# it, longer, and lengths that land exactly on .5 boundaries where numpy's
# round-half-to-even differs from naive rounding.
CASES = [1, 7, 32, 33, 63, 64, 69, 120]

# Aspect ratios the contract must agree on. 1.0 is the square CISLR format,
# 16/9 is INCLUDE and most webcams, 4/3 is the rest of them, and 9/16 is a
# phone held upright. A test that only ever passed 1.0 would not have caught
# the isotropic bug at all, because y/1 is the identity.
ASPECTS = [1.0, 16 / 9, 4 / 3, 9 / 16]

NODE_HARNESS = r"""
import { readFileSync } from "node:fs";
const tsSrc = readFileSync(process.argv[2], "utf8");

// strip TypeScript syntax the JS engine will not accept
const js = tsSrc
  .replace(/^\s*export\s+type[\s\S]*?;\s*$/gm, "")
  .replace(/:\s*Float64Array\b/g, "").replace(/:\s*Float32Array\b/g, "")
  .replace(/:\s*PointFrame\[\]/g, "").replace(/:\s*PointFrame\b/g, "")
  .replace(/:\s*Landmark\[\]\s*\|\s*null/g, "").replace(/:\s*Landmark\b/g, "")
  .replace(/:\s*number\[\]/g, "").replace(/:\s*number\b/g, "")
  .replace(/\bexport\s+const\b/g, "const")
  .replace(/\bexport\s+function\b/g, "function")
  .replace(/new Array\(N_POINTS\)/g, "new Array(N_POINTS)");

const mod = await import("data:text/javascript," + encodeURIComponent(
  js + "\nexport { extractFeatures, N_POINTS, FEATURE_SIZE };"));

const input = JSON.parse(readFileSync(process.argv[3], "utf8"));
const frames = input.frames.map(f => f.map(([x, y, z]) => ({ x, y, z })));
const out = mod.extractFeatures(frames, input.aspect);
process.stdout.write(JSON.stringify(Array.from(out)));
"""


def run_ts(seq: np.ndarray, aspect: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.mjs"
        harness.write_text(NODE_HARNESS)
        data = Path(tmp) / "input.json"
        data.write_text(json.dumps({"frames": seq.tolist(), "aspect": aspect}))
        proc = subprocess.run(
            ["node", str(harness), str(TS_FEATURES), str(data)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"node harness failed:\n{proc.stderr}")
        return np.array(json.loads(proc.stdout), dtype=np.float32)


def main() -> int:
    rng = np.random.default_rng(20260827)
    failures = 0

    print(f"contract: {features.N_POINTS} points x {features.SEQ_LEN} frames "
          f"-> {features.FEATURE_SIZE} features\n")

    total = 0
    for aspect in ASPECTS:
        print(f"aspect {aspect:.4f}")
        for t in CASES:
            # plausible unit-coordinate landmarks, with a real shoulder separation
            seq = rng.uniform(0.1, 0.9, size=(t, features.N_POINTS, 3))
            seq[:, features.L_SHOULDER, :2] = [0.40, 0.35]
            seq[:, features.R_SHOULDER, :2] = [0.60, 0.35]

            py = features.extract(seq, aspect)
            ts = run_ts(seq, aspect)
            total += 1

            if py.shape != ts.shape:
                print(f"  FAIL  T={t:3d}  shape {py.shape} vs {ts.shape}")
                failures += 1
                continue

            delta = np.abs(py - ts).max()
            ok = delta <= 1e-5
            if not ok:
                failures += 1
            print(f"  {'PASS' if ok else 'FAIL'}  T={t:3d}  max|py-ts| = {delta:.3e}")
        print()

    if failures:
        print(f"{failures}/{total} FAILED: extractors have drifted. "
              f"Do not train until this passes.")
        return 1
    print(f"{total}/{total} passed: Python and TypeScript agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
