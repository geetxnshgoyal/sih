# Setu — handoff

Two-way bridge between Indian Sign Language and India's spoken languages.
Target: SIH 2026, Student Innovation, **SIH26196** (MedTech · Software).
Idea submission deadline **20 September 2026**.

Read this before changing anything in `train/` or `app/src/lib/`.

---

## 1. Run it

```bash
cd app && npx vite --port 5174      # http://localhost:5174
```

Three modes in the top bar:

| mode | state |
|---|---|
| **Conversation** | works reliably, no camera needed — this is the demo |
| **Record signs** | works — capture your own vocabulary |
| **Signs to speech** | works, ~52% accuracy over 264 classes |

---

## 2. Environments — three venvs, on purpose

| venv | Python | why |
|---|---|---|
| `.venv` | 3.14 | data work, numpy only |
| `.venv-tf` | 3.11 | **TensorFlow does not support Python 3.14** |
| `.venv-mp` | 3.11 | **MediaPipe 1.x's Python HolisticLandmarker is broken** (`Check failed: service_ Service is unavailable`). Pinned to 0.10.14 for the legacy Holistic solution |

---

## 3. The pipeline

```bash
.venv/bin/python    train/test_parity.py        # MUST pass before training
.venv/bin/python    train/profile_signers.py    # recover signer groups
.venv/bin/python    train/preprocess.py         # 4,284 clips -> dataset.npz
.venv-tf/bin/python train/train.py              # far + close scores, auto-promote
.venv-tf/bin/python train/export_tfjs.py        # -> app/public/model (2.4 MB)
```

Your own recordings:

```bash
.venv/bin/python    train/ingest_recordings.py setu-recordings-*.json
.venv-tf/bin/python train/eval_on_takes.py      setu-recordings-*.json
```

---

## 4. Current results

| split | far camera | close (laptop) |
|---|---|---|
| random (same signers both sides) | 90.3% | 88.4% |
| **held-out group (mean)** | **52.0%** | **51.6%** |

264 classes, chance 0.4%. **Report the held-out number.** The random split
inflates by ~40 points because the same person appears in train and test.

---

## 5. Three bugs that made the camera path fail

Each was hidden behind the previous one. All fixed.

**1. Two landmarkers, incompatible z conventions.**
The app ran `PoseLandmarker` + `HandLandmarker` separately. INCLUDE was
extracted with *Holistic*, one unified depth frame. Hand z is relative to the
wrist, pose z to the torso — and z carries about a third of the input signal
(pose z std 1.32, hand z std 0.75, against x,y std 1.24 and 2.32). Fixed by
switching to `HolisticLandmarker` in `useLandmarkers.ts`.

**2. The model was never trained for close range.**
Measured by re-projecting held-out clips through a pinhole model:

| condition | top-1 |
|---|---|
| as recorded | 67.3% |
| scaled ×2.6, no perspective | 67.3% |
| perspective at laptop distance | **2.1%** |

Scale is a **no-op** — standardisation divides it out exactly (verified 4e-16).
Perspective is the whole gap. Fixed with perspective augmentation in
`train/augment.py`.

**3. Rolling window instead of trimmed signs.**
Every training clip is trimmed to one sign. The app fed a continuous 64-frame
buffer that was mostly rest position:

| input | top-1 |
|---|---|
| trimmed clip (as trained) | 66.0% |
| rolling window + 10 rest frames | 54.8% |
| rolling window + 40 rest frames | **23.2%** |

Fixed with `app/src/lib/segment.ts` — motion-energy segmentation that starts on
a movement burst, ends after 6 still frames, trims trailing stillness, and
classifies once per sign. `StabilityGate.once()` replaces the N-of-M frame vote,
which could never fire on a single prediction.

---

## 6. The mistake worth remembering

I built perspective augmentation, evaluated it on the **far-camera** held-out
set, saw 45.0% against 51.6%, and reverted it as a regression. That test set
contains no close-range footage, so it was structurally incapable of measuring
what the augmentation was for. I threw away the correct fix using a metric that
could not see it.

`train.py` now reports **far and close scores separately and promotes on
close**, because that is the condition the app runs in.

**If you change augmentation, check that a metric exists which can detect the
change.** Otherwise you will confidently undo the right work.

---

## 7. Invariants — do not break these

**`train/features.py` ↔ `app/src/lib/features.ts` must stay bit-identical.**
If they drift, the model trains on one representation and runs on another;
offline accuracy stays high and live accuracy collapses with no visible cause.

```bash
.venv/bin/python train/test_parity.py     # 8/8, exact zero difference
```

The test is verified to catch real drift — injecting a z-axis term into the
shoulder span, or a wrong anchor landmark, both fail it loudly.

**Model round-trip** is checked in-browser by `app/src/devParity.ts` (dev only):
TF.js reproduces Python to 1.19e-7.

**Checkpoints are versioned.** `train.py` writes
`gloss_classifier_<stamp>_close<X>_far<Y>.keras` and only promotes to
`gloss_classifier.keras` if it beats the recorded best. An earlier version
overwrote a single file and destroyed the best model across three runs.

---

## 8. Data

| source | what | licence |
|---|---|---|
| `data/Pose_Signs/` | **INCLUDE**, 4,284 clips, 264 signs — the only ISL data | CC BY 4.0 |
| `data/video_landmarks/` | re-extracted with **468-point face mesh** | CC BY 4.0 |
| `data/poses/` | AUTSL, MS-ASL, WLASL, GSL, LSA64 — other sign languages | CC BY 4.0 |

Schema of the INCLUDE pickles (verified):

```
{'keypoints': (T, 75, 3) float64, 'confidences': (T,75), 'vid_shape': (W,H)}
75 = 33 pose + 21 left hand + 21 right hand
x,y are PIXEL coords -> divide by vid_shape.  z is already relative.
Pose 23-32 are legs: extrapolated outside frame, mean confidence 0.25 -> dropped.
```

Load them with the restricted unpickler used throughout `train/` — they are
third-party files and arbitrary pickles execute code.

### Things that do not exist

- **Dialect labels.** No published ISL corpus tags dialect. Do not promise this.
- **A conversation vocabulary.** INCLUDE is a *lexicon* — it has "Actor",
  "Election", "Monsoon" and no *yes*, *no*, *please*, *help*, *where*.
- **FDMSE-ISL** (40k clips) — no public download, author request only.
- **CISLR** — gated on Hugging Face, someone must accept the terms.

### Known bias

INCLUDE is 7 signers from one school in Chennai, one room, one distance.
Signer groups in `data/signer_index.json` are **body-type clusters recovered
from pose-stable ratios**, not identified individuals — the official split is
offline (Google Drive 404). Stricter than a random split, weaker than a true
signer-disjoint one. Say so in the pitch.

---

## 9. Background jobs

```bash
data/fetch_video.sh        # 46 video parts, resumable, single-instance lock
data/fetch_poses_all.sh    # other sign-language pose datasets
run_extract_loop.sh        # extracts each part as it lands, deletes the video
```

The extract loop deletes each 1.3 GB archive after pulling landmarks (104 clips
→ 38 MB), so disk stays flat. Markers: `.done` = downloaded, `.extracted` =
processed, and the downloader skips both.

When most parts are through, set `FACE_MODE = "FULL_FACE"` in
`train/features.py` (65 → 113 points) and retrain to pick up eyebrow, eye and
lip landmarks — the non-manual grammar ISL uses for questions and negation.
Verified present in the extracted data: eyebrow height varies by 0.077 within a
single clip, which the 11-point pose release could not represent at all.

---

## 10. Gotchas that cost real time

- **Keras 3 → TF.js layers models do not work.** `batch_shape` vs
  `batch_input_shape`, object-form `inbound_nodes`, `sequential/` weight
  prefixes. Use `tfjs_graph_model` via SavedModel and `tf.loadGraphModel`.
- **`tensorflowjs` needs `protobuf==6.31.1` and `setuptools<81`** (84 removed
  `pkg_resources`, which `tensorflow_hub` imports).
- **macOS has no `flock`** — the download scripts use a `mkdir` lock.
- **Devanagari needs `\p{M}`** in tokenisers. Every Indic vowel sign is a
  combining mark; without it `नमस्ते` becomes `नमस त` and matches nothing.
- **Zenodo is unreliable** — expect stalls, failed resumes, and at least one
  archive that silently truncated. All fetchers retry indefinitely.

---

## 11. What is left

1. **Record your own 20–30 signs.** Same camera, same room, same distance as
   the demo. This is the difference between 52% and something you would put on
   stage. Everything downstream already works.
2. **Finish the face-mesh extraction**, then `FULL_FACE` and retrain.
3. **LLM gloss reordering.** ISL word order differs from spoken order.
   `reverse.ts` currently does word matching and says so honestly — it is not
   translation.
4. **Bhashini** for real translation beyond the precomputed phrase table.
   Free tier is PoC-only; cache everything demo-critical.
5. Ask the SPOC: 2026 per-college nomination quota, whether one team may submit
   more than one idea, and the real deadline (portal says 20 Sept, secondary
   sources say 30 Sept).

---

## 12. Pitch framing that is actually true

> Thirty ministries submitted 226 problem statements to SIH 2026. Not one
> addresses the ~5 million deaf Indians whose access the RPwD Act already
> mandates.

- Google's **SL2T** shipped Aug 2026 — ASL only, Pixel 11 only, no API, no ISL,
  and it outputs text, not regional speech. It uses MediaPipe pose landmarks
  on-device, the same architecture as this. Good validation, no overlap.
- **SignGemma** was announced May 2025 and still has not shipped; Google said
  publicly they missed their own quality bar.
- Report **52%**, not 90%. Explain the difference — it is the most credible
  thing you can say.
