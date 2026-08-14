# Setu — architecture and system state

Two-way bridge between Indian Sign Language and India's spoken languages.
Target: **SIH 2026, Student Innovation** — submitted under **two themes**,
**MedTech** (SIH26196) and **Travel & Tourism**.

> **One system, two settings — not two products.** The recognition stack is
> entirely domain-neutral: one 264-sign classifier, one feature contract, one
> segmenter. INCLUDE is a general lexicon, and "Doctor" and "Train Station" are
> the same kind of sign to the model. Everything below the UI is unaware of which
> setting it is running in. See §11.

This document is the precise reference: where every artefact lives, what shape it
has, and what is verified versus assumed. `HANDOFF.md` covers *why* decisions were
made and which bugs cost time — read that for rationale, this for facts.

Last audited: **30 August 2026**.

---

## 1. Repository map

```
~/sih/
├── data/                   6.6 GB   all corpora and the preprocessed tensor
│   ├── Pose_Signs/         640 MB   INCLUDE pose release — TRAINS THE CURRENT MODEL
│   ├── video_landmarks/    231 MB   468-pt face-mesh re-extraction — IN PROGRESS
│   ├── video/              143 MB   raw INCLUDE video, transient (deleted after extract)
│   ├── poses/              5.0 GB   AUTSL, MS-ASL, WLASL, GSL, LSA64 — NOT ISL, unused
│   ├── INCLUDE-poses.zip   632 MB   original archive of Pose_Signs
│   ├── dataset.npz          83 MB   preprocessed training tensor
│   ├── signer_index.json   384 KB   clip -> {class, signer group}
│   └── signer_profiles.npz 504 KB   body-ratio clusters used to derive groups
├── models/                  39 MB   checkpoints, labels, metrics, holistic .task
├── train/                  156 KB   the Python pipeline (13 scripts)
├── app/                    438 MB   Vite + React 19 front end (node_modules included)
├── phase0/                          early spike, superseded
├── HANDOFF.md                       rationale, bug history, pitch framing
└── ARCHITECTURE.md                  this file
```

### Three virtualenvs, deliberately

| venv | Python | Why it must be separate | Verified |
|---|---|---|---|
| `.venv` | 3.14 | data work, numpy 2.5.2 only | ✅ |
| `.venv-tf` | 3.11 | TensorFlow 2.19 has no 3.14 wheels | ✅ |
| `.venv-mp` | 3.11 | MediaPipe pinned **0.10.14** — 1.x's Python `HolisticLandmarker` raises `Check failed: service_ Service is unavailable` | ✅ |

`.venv-mp` prints a harmless `MessageFactory has no attribute GetPrototype`
protobuf warning on import; `run_extract_loop.sh` filters it. `mp.solutions.holistic.Holistic`
constructs correctly — confirmed 30 Aug.

---

## 2. Data

### 2.1 What actually trains the model

**`data/Pose_Signs/`** — the OpenHands INCLUDE pose release. This is the *only*
ISL data in the project. CC BY 4.0.

- **4,284 clips, 264 classes, 15 categories**

| category | clips | category | clips | category | clips |
|---|---|---|---|---|---|
| Adjectives | 791 | Home | 379 | Pronouns | 168 |
| People | 513 | Society | 324 | Animals | 166 |
| Places | 399 | Days_and_Time | 298 | Electronics | 140 |
| Jobs | 225 | Clothes | 198 | Seasons | 85 |
| Colours | 222 | Greetings | 190 | Means_of_Transportation | 186 |

Per-clip pickle schema (verified):

```python
{'keypoints':   (T, 75, 3) float64,
 'confidences': (T, 75)    float64,
 'vid_shape':   (W, H)}
```

- `75 = 33 pose + 21 left hand + 21 right hand` (MediaPipe **Holistic**)
- `x, y` are **pixel** coordinates → divide by `vid_shape`. `z` is already relative.
- Pose landmarks **23–32 are legs**: extrapolated outside frame, mean confidence
  0.25 → dropped. Keeps 0–22.

> These are third-party pickles. Load them only with the restricted unpickler used
> throughout `train/` — arbitrary pickles execute code on load.

### 2.2 The face-mesh re-extraction (blocks FULL_FACE)

INCLUDE's pose release carries **no face mesh** — only 11 coarse pose landmarks
(nose, 6 eye, 2 ear, 2 mouth-corner). That is enough for head nod/shake/tilt and
nothing else. ISL marks **yes/no questions with raised eyebrows, wh-questions with
furrowed brows, and negation with a head shake** — a hands-only model renders a
question as a statement, which is a correctness failure, not a polish one.

So the raw video (Zenodo 4010759, **56.8 GB, 46 parts**) is being re-extracted with
full Holistic to recover the 468-point mesh.

**Status as of 30 Aug:**

| | |
|---|---|
| Parts extracted | **8 of 46** (all Adjectives) |
| Clips with face mesh | **687 of 4,284 — 16%** |
| Currently downloading | `Adjectives_1of8.zip` (partial), then `Animals_1of2.zip` |

Disk is safe: `run_extract_loop.sh` deletes each ~1.3 GB archive immediately after
pulling landmarks (104 clips → ~38 MB), so peak usage stays at roughly one part.
**69 GB free** at audit time, against a transient footprint of ~2 GB.

### 2.3 Other sign languages — present but unused

`data/poses/` holds AUTSL, MS-ASL, WLASL, GSL, LSA64 (5.0 GB). These are *not*
ISL. They are kept for a possible cross-lingual pretraining experiment and are
not part of any current result. Do not cite them as ISL data.

### 2.4 What does not exist — do not promise these

- **Dialect labels.** No published ISL corpus tags dialect.
- **A conversation vocabulary.** INCLUDE is a *lexicon*: "Actor", "Election",
  "Monsoon". It has no *yes*, *no*, *please*, *help*, *where*. This is why the
  demo's conversation mode uses a separate curated phrase set.
- **FDMSE-ISL** (40k clips) — no public download, author request only.
- **CISLR** — gated on Hugging Face, requires accepting terms.

### 2.5 Known bias — state this in the pitch

INCLUDE is **7 signers from one school in Chennai, one room, one camera distance**.
`data/signer_index.json` groups are **body-type clusters recovered from pose-stable
ratios**, not identified individuals — the official split is offline (Google Drive
404). This is stricter than a random split and weaker than a true signer-disjoint
one. Say exactly that.

---

## 3. Feature pipeline

`train/features.py` ↔ `app/src/lib/features.ts` are a **bit-identical contract**.
If they drift, the model trains on one representation and runs on another; offline
accuracy stays high while live accuracy collapses with no visible cause.

```bash
.venv/bin/python train/test_parity.py    # 8/8, exact zero difference — MUST pass
```

The test is verified to catch real drift: injecting a z-term into the shoulder
span, or using a wrong anchor landmark, both fail it loudly.

### Current configuration

```python
SEQ_LEN   = 32
POSE_KEEP = 23                  # pose 0..22, includes the 11 head points
FACE_MODE = "HEAD_ONLY"         # ← flips to FULL_FACE when extraction completes
N_FACE    = 0                   # 48 under FULL_FACE
N_POINTS  = 23 + 21 + 21 + 0    # 65 now, 113 under FULL_FACE
N_DIMS    = 3
FEATURE_SIZE = 32 * 65 * 3      # 6,240 floats per clip
```

### The transform, in order

1. **`select_points`** — `(T,75,3) → (T,65,3)`, dropping legs.
2. **`to_unit`** *(dataset only)* — divide `x` by `vid_shape[0]`, `y` by `vid_shape[1]`.
   The browser's MediaPipe already returns unit coordinates, so this step only
   brings the dataset into the same space. **Everything below is the shared contract.**
3. **`anchor`** — subtract the **shoulder midpoint** (landmarks 11, 12), then divide
   by 2-D shoulder span. Position- and scale-invariant.
   *Anchoring at the shoulders rather than the head is deliberate: head nod, shake,
   and tilt survive normalisation and stay visible to the model.*
4. **`resample`** — any `T` → exactly 32 frames by nearest-index striding.
5. **`standardise`** — flatten, then `(v - mean) / std` over the whole clip.

Scale is provably a no-op after standardisation (verified to 4e-16). **Perspective
is not** — see §5.

### FULL_FACE subset (`train/face.py`)

Using all 468 mesh points would swamp 65 body points, so FULL_FACE adds a curated
**48**: 10+10 eyebrow, 6+6 eye aperture (EAR-style), 8+8 lip outer/inner.

---

## 4. Model

1-D CNN over the temporal axis. **Not** a transformer — TFLite and TF.js have no
native `MultiHeadAttention`, and at 264 classes attention buys little over
convolutions.

```
Input (32, 195)                     # 32 frames × (65 points × 3 dims)
  Conv1D(128, k=5, no bias) → BN → ReLU
  Conv1D(256, k=5, no bias) → BN → ReLU → MaxPool(2) → Dropout(0.2)
  Conv1D(256, k=3, no bias) → BN → ReLU
  GlobalAveragePooling1D → Dropout(0.4)
  Dense(264, softmax)
```

`SEED = 20260827`, `EPOCHS = 60`, `BATCH = 64`.

### Artefacts

| path | what |
|---|---|
| `models/gloss_classifier.keras` | promoted best |
| `models/gloss_classifier_<stamp>_close<X>_far<Y>.keras` | every run, versioned |
| `models/labels.json` | 264 class names |
| `models/metrics.json` | the numbers below |
| `models/holistic_landmarker.task` | MediaPipe model |
| `app/public/model/` | TF.js graph model, **2.4 MB** (`group1-shard1of1.bin`) |

**Checkpoints are versioned on purpose.** `train.py` promotes to
`gloss_classifier.keras` only if the run beats the recorded best. An earlier
version overwrote a single file and destroyed the best model across three runs.

---

## 5. Results — and which number to report

| split | far camera | close (laptop) |
|---|---|---|
| random (same signers both sides) | 90.3% top-1 / 97.4% top-5 | 88.4% |
| **held-out group (mean)** | **52.0% top-1 / 75.7% top-5** | **51.6% / 76.2%** |

Per-group held-out top-1: **42.7% / 67.4% / 45.8%** — the spread across body-type
groups is itself a finding worth showing.

264 classes, chance **0.4%**.

> **Report 52%, not 90%.** The random split inflates by ~38 points because the same
> person appears on both sides. Explaining that gap is the most credible thing the
> team can say — most competing projects quote the inflated figure.

### Three bugs that made the live camera path fail

Each was hidden behind the previous one. All fixed; do not reintroduce them.

1. **Two landmarkers, incompatible z conventions.** The app ran `PoseLandmarker` +
   `HandLandmarker` separately, but INCLUDE was extracted with *Holistic* — one
   unified depth frame. Hand `z` is wrist-relative, pose `z` torso-relative, and
   `z` carries roughly a third of the signal (pose z std 1.32, hand z std 0.75,
   against x,y std 1.24 / 2.32). Fixed by switching to `HolisticLandmarker`.

2. **Never trained for close range.** Re-projecting held-out clips through a
   pinhole model:

   | condition | top-1 |
   |---|---|
   | as recorded | 67.3% |
   | scaled ×2.6, no perspective | 67.3% |
   | perspective at laptop distance | **2.1%** |

   Scale is a no-op; **perspective is the entire gap**. Fixed with perspective
   augmentation in `train/augment.py`. `train.py` now reports far and close
   separately and **promotes on close**, because that is where the app runs.

3. **Rolling window instead of trimmed signs.** Every training clip is one trimmed
   sign; the app fed a continuous 64-frame buffer that was mostly rest position:

   | input | top-1 |
   |---|---|
   | trimmed clip (as trained) | 66.0% |
   | rolling window + 10 rest frames | 54.8% |
   | rolling window + 40 rest frames | **23.2%** |

   Fixed by `app/src/lib/segment.ts` — motion-energy segmentation that starts on a
   movement burst, ends after 6 still frames, trims trailing stillness, and
   classifies once per sign.

---

## 6. Pipeline commands

```bash
# preconditions
.venv/bin/python    train/test_parity.py       # MUST pass before training
.venv/bin/python    train/profile_signers.py   # recover signer groups

# build + train + ship
.venv/bin/python    train/preprocess.py        # 4,284 clips -> data/dataset.npz
.venv-tf/bin/python train/train.py             # far + close scores, auto-promote
.venv-tf/bin/python train/export_tfjs.py       # -> app/public/model (2.4 MB)

# your own recordings
.venv/bin/python    train/ingest_recordings.py setu-recordings-*.json
.venv-tf/bin/python train/eval_on_takes.py     setu-recordings-*.json
```

`data/dataset.npz` currently holds:

```
X      (4284, 32, 65, 3) float32
y      (4284,)           int32
signer (4284,)           int32
labels (264,)            <U16
```

### Background jobs

```bash
cd data && ./fetch_video.sh    # 46 parts, resumable, mkdir-based single-instance lock
./run_extract_loop.sh          # extracts each part as it lands, deletes the archive
```

Markers: `.done` = downloaded, `.extracted` = processed. The downloader skips both.
macOS has no `flock`, hence the `mkdir` lock.

---

## 7. Application

Vite 8 + React 19 + TypeScript 6, TF.js 4.22, MediaPipe tasks-vision 0.10.14.

```bash
cd app && npx vite --port 5174     # http://localhost:5174
```

| mode | file | state |
|---|---|---|
| **Conversation** | `HearingSide.tsx` | works reliably, no camera — **this is the demo** |
| **Record signs** | `Recorder.tsx` | works — capture your own vocabulary |
| **Signs to speech** | `SignBridge.tsx` | works, ~52% over 264 classes |

### Runtime chain

```
webcam → HolisticLandmarker (GPU, VIDEO mode, CDN wasm 0.10.14)
       → segment.ts        motion-energy: burst starts, 6 still frames end
       → features.ts       the §3 contract, mirrored from Python
       → classifier.ts     tf.loadGraphModel, input [1, 32, 195]
       → gate.ts           StabilityGate.once() — one decision per segment
       → sentence.ts       UtteranceBuilder, 2600 ms pause closes an utterance
       → speech.ts         TTS in hi/ta/te/bn/mr/en-IN
```

Reverse direction (`reverse.ts` → `SignPlayer.tsx`): spoken text → gloss match →
animated pose skeleton from `app/public/model/_signs.json` (96 signs). Skeletons
rather than video because the pose release is already on disk at 662 MB where raw
video is 56.8 GB, and a skeleton avoids jarring cuts between different signers.

**Model round-trip verified in-browser** by `app/src/devParity.ts` (dev only):
TF.js reproduces Python to **1.19e-7**.

### Why `tfjs_graph_model` and not a layers model

Keras 3 → TF.js layers conversion is broken: `batch_shape` vs `batch_input_shape`,
object-form `inbound_nodes`, `sequential/` weight prefixes. Export via SavedModel
and load with `tf.loadGraphModel`. `tensorflowjs` needs **`protobuf==6.31.1`** and
**`setuptools<81`** (84 removed `pkg_resources`, which `tensorflow_hub` imports).

---

## 8. Honest limitations

These are real and should be stated rather than hidden:

1. **`reverse.ts` is word matching, not translation.** It keeps spoken word order.
   ISL has its own grammar — "STATION GO WHERE", not "where do I go for the
   station". The code says so in its own comments and the UI labels output as a
   *gloss sequence*.
2. **`sentence.ts` does not reorder either.** It speaks known multi-word phrases
   from a 3-entry phrasebook and otherwise reads glosses in signed order.
   Inventing plausible word order would produce confident mistranslation, which is
   worse than none.
3. **No non-manual grammar yet.** HEAD_ONLY cannot see eyebrows or mouth shape, so
   questions and negation are invisible to the model. This is what FULL_FACE fixes.
4. **52% is a lexicon score on 264 isolated signs**, not continuous sentence
   translation.

---

## 9. FULL_FACE — the ablation

`FACE_MODE = "FULL_FACE"` widens the input 65 → 113 points. There is a
linguistic reason to expect a gain (eyebrows and mouth carry ISL's question and
negation marking), but that is an argument, not evidence — and `HANDOFF.md` §6
records what happens when a change like this is judged by a metric that cannot
see it.

So it is measured directly, on the 8 parts already extracted:

```bash
.venv/bin/python    train/preprocess_face.py   # -> data/dataset_face.npz
.venv-tf/bin/python train/train_ablation.py    # -> models/ablation_face.json
```

`preprocess_face.py` emits **both** representations over the same clips:

```
X_head (685, 32,  65, 3)    23 pose + 21 + 21 hands
X_full (685, 32, 113, 3)    the same, + 48 face points
y, signer, labels           51 classes, 3 groups
```

Face points are appended **after** the body block, so landmarks 11/12 remain the
shoulders and `features.anchor` works unchanged on both arms. `train_ablation.py`
then trains both with the same architecture, seed, augmentation draws, and
held-out-group protocol. The only difference between the arms is the face block,
so the delta is attributable to it.

> **Scope.** 685 clips, 51 classes, chance 2.0% — the Adjectives categories only.
> This is **not** the 264-class result and must never be quoted as one. It answers
> the narrower question of whether the face block earns its place, early enough to
> act on. Note also that adjectives carry relatively little non-manual grammar; the
> question-heavy categories (Pronouns, Society) are still downloading, so a null
> result here is weak evidence, not a refutation.

### Result (31 Aug) — the decisive re-run, and it does not support FULL_FACE

Ran on **1,441 clips / 98 classes** across Adjectives, Animals, **Pronouns** and
**Society** — the question-heavy categories the first run lacked. Chance 1.0%.

| arm | far camera | close (laptop) |
|---|---|---|
| HEAD_ONLY (65 pt) | **60.1%** | **58.5%** |
| FULL_FACE (113 pt) | 59.7% | 57.1% |
| delta | −0.4 pp | −1.4 pp |

Per-group close delta: `g0 +2.1pp`, `g1 −1.3pp`, `g2 −5.1pp` — spread 3.0 pp
against a mean of 1.4 pp. Still formally **INCONCLUSIVE**: three signer groups
cannot resolve a difference this small.

**But compare the two runs, because that is the real finding:**

| run | data | far Δ | close Δ |
|---|---|---|---|
| 30 Aug | 685 clips, 51 cls, Adjectives | −2.4 pp | **+1.5 pp** |
| 31 Aug | 1,441 clips, 98 cls, **+Pronouns/Society** | −0.4 pp | **−1.4 pp** |

**The close-range delta flipped sign** — the direction that looked promising
reversed once the categories that should *favour* the face block were added. A
real effect does not invert when you strengthen the conditions for it. This is
noise, not signal.

That kills the reading the first run invited. It is no longer the
perspective-augmentation shape from `HANDOFF.md` §6, where far lost and close
gained; FULL_FACE now loses on both arms.

**Conclusion: do not set `FACE_MODE = "FULL_FACE"`.** Not because it is proven
harmful — it is not — but because two runs, the second on the categories designed
to show its benefit, produce no measurable gain. The 48 face points cost 74% more
input width for nothing detectable.

**What would still change the answer** (in order of expected value):

1. **More signer groups.** The blocker is now power, not vocabulary: 3 groups
   against a ~1 pp effect. Recording your own signers adds groups directly.
2. **The 48-point subset may be wrong.** Eyebrow height varies 0.077 within a
   clip, so signal exists — but a 32-frame resample may simply smear a brief
   eyebrow raise away. Worth testing temporal resolution before discarding the
   face entirely.
3. **Augmentation was tuned for body landmarks.** Perspective warping may be
   actively damaging face points.

### Result (30 Aug) — the first run, superseded above

| arm | far camera | close (laptop) |
|---|---|---|
| HEAD_ONLY (65 pt) | **69.6%** | 62.6% |
| FULL_FACE (113 pt) | 67.3% | **64.1%** |
| delta | **−2.4 pp** | **+1.5 pp** |

Per-group close delta: `g0 +0.8pp`, `g1 −4.8pp`, `g2 +8.6pp` — **spread 5.5 pp
against a mean of 1.5 pp.**

**The mean is smaller than the spread, so this subset cannot separate the two
arms.** Three groups give no power to call a 1.5-point difference, and reporting
it as a win would be exactly the overclaiming this project avoids everywhere
else. `train_ablation.py` prints `INCONCLUSIVE` for this reason.

What *is* worth noting is the **direction**: FULL_FACE loses on far camera and
gains at close range. That is the same shape as the perspective-augmentation
result in §5 — and the same trap. Judged on the far-camera number alone,
FULL_FACE looks like a 2.4-point regression and would be rejected. The app runs
at close range.

So: do not switch `FACE_MODE` on this evidence, and do not discard the change
either. Re-run when Pronouns and Society are extracted — those are the categories
where non-manual grammar actually lives.

Full numbers in `models/ablation_face.json` (`conclusive: false`).

### Fetch order was the hidden blocker (fixed 30 Aug)

Zenodo lists the 46 parts alphabetically, which put **Pronouns at 39–40 and
Society at 43–45** — the decisive categories arriving only after ~50 GB. The
ablation could not be settled until essentially the whole corpus was down, by
which point the decision would have been made by default.

`fetch_video.sh` now orders by linguistic priority instead:

```
Pronouns → Society → Greetings → People → (everything else, alphabetical)
```

Same 46 files, same total bytes; only the order changes, and `.done` /
`.extracted` markers make it safe to reorder mid-run. This moves a conclusive
ablation from ~35 parts away to ~5.

**Re-run the ablation once Pronouns and Society are extracted** — that is the
result worth quoting, and it is the one that decides `FACE_MODE`.

This is automated. `run_ablation_when_ready.sh` waits for the five decisive parts
and then runs both stages unattended:

```bash
./run_ablation_when_ready.sh          # logs to /tmp/setu-ablation.log
```

It polls once a minute with an 8-hour ceiling, prints a progress line every 15
minutes, and fails loudly naming the missing parts rather than hanging if the
fetcher dies. Safe to re-run — `preprocess_face.py` globs whatever is in
`data/video_landmarks`, so it always uses everything extracted so far.

Started 30 Aug alongside the reordered fetch; 6.3 GB of decisive data at roughly
1.7 GB/h.

---

## 10. Gloss reordering — implemented, offline

ISL word order is not spoken word order. This is now handled, and the design
constraint that shaped it is worth stating: **the stage demo must never depend on
a live network call**, and **an API key must never reach the browser**.

Both are satisfied by generating the reorderings ahead of time.

```bash
# once, with your own key — never the browser's
ANTHROPIC_API_KEY=sk-... node --experimental-strip-types app/tools/buildGlossTable.ts
```

| file | role |
|---|---|
| `app/tools/glossCorpus.ts` | 73 gloss sequences (medical / travel / civic / social) |
| `app/tools/buildGlossTable.ts` | generator — Claude Opus 5, structured output, batched, resumable |
| `app/public/model/_utterances.json` | the shipped table |
| `app/src/lib/glossTranslate.ts` | runtime lookup + provenance |

Every corpus entry is validated against `models/labels.json` before generation, so
the corpus cannot drift from what the classifier can actually emit. The generator
writes after each batch, so an interrupted run keeps its progress.

### The provenance contract

The runtime returns *where the sentence came from*, and the UI shows it:

| source | meaning | may be called a translation? |
|---|---|---|
| `reordered` | precomputed reordering | **yes** |
| `phrasebook` | hand-verified phrase table | as a phrase |
| `gloss-order` | signs read in signed order | **no** — badged amber |

This preserves the original caution rather than discarding it. Reordering happens
only where a verified entry exists; everywhere else the app still refuses to
pretend, and says so on screen.

### What the vocabulary cannot express

Validation surfaced this concretely — `UNREACHABLE` in `glossCorpus.ts`:

> **water · help · yes · no · where · please · pain · name**

None of these are INCLUDE classes, so no amount of translation work makes them
reachable. "I need water" is not a hard sentence; it is simply unsayable with this
lexicon. That is the checkable form of "INCLUDE is a lexicon, not a conversation
vocabulary" — and the strongest argument for recording your own signs.

---

## 11. Deployment domains — Healthcare and Travel & Tourism

The app ships both SIH themes from one codebase. A switch in the top bar changes
the setting; the choice persists per device (`setu.domain`).

**What a domain is allowed to change — and nothing else:**

1. which phrases lead in the tap-to-say rail
2. what spoken words map onto (`platform` → Train Station, `ward` → Hospital)
3. what the kiosk calls itself (`RECEPTION · OPD` / `ENQUIRY · HELP DESK`)

That is the whole abstraction, and keeping it that small is the point: the Travel
submission is honestly the same system rather than a fork maintained twice, and a
third setting (civic services, railway enquiry) is a data entry, not a rewrite.

| file | role |
|---|---|
| `app/src/lib/domains.ts` | both domain definitions + the persisted store |
| `app/src/components/QuickPhrases.tsx` | the tap-to-say rail |
| `app/tools/checkDomains.ts` | validates every gloss against `labels.json` |

```bash
node --experimental-strip-types app/tools/checkDomains.ts   # 0 errors required
```

Run it after any retrain. Every quick phrase and synonym target names a gloss the
classifier is supposed to emit; a typo or a dropped class turns a demo button
into a silent no-op that still renders and still looks tappable.

### Direction matters, and the two are not symmetric

- **Quick phrases run FORWARD** — tapped by the Deaf user, spoken aloud. They
  need only to be things a person wants to say; no pose frames required.
- **Synonyms run REVERSE** — spoken word → played ISL skeleton. These *do* need
  bundled pose frames.

`checkDomains.ts` reports 19 warnings for glosses that are recognisable but not
playable — most importantly **`Location`**, the stand-in for WHERE that appears
in nearly every "where is X" phrase. `textToGlosses` already guards on the
library, so an unplayable match is reported as *skipped* rather than queueing a
sign that renders as an empty player. The warnings are informational, not bugs —
but they are the list of signs worth recording first for the reverse direction.

---

## 12. Remaining work

### Running in background (started 30 Aug)

- **Face-mesh extraction** — 8/46 parts. `fetch_video.sh` + `run_extract_loop.sh`
  relaunched. Logs: `/tmp/setu-fetch.log`, `/tmp/setu-extract.log`.
  Expect stalls: Zenodo returns 200 instead of 206 on resume, so curl truncates
  and restarts a part. It does make net progress, but slowly and not monotonically.

### Blocked on that

- **Full FULL_FACE retrain.** When most parts are through: set
  `FACE_MODE = "FULL_FACE"` in `train/features.py`, **mirror it in
  `app/src/lib/features.ts`**, re-run `test_parity.py`, then `preprocess.py` →
  `train.py` → `export_tfjs.py`. Signal is verified present: eyebrow height varies
  by **0.077 within a single clip**, which 11 coarse pose points cannot represent.

### Do now — highest value per hour

1. **Record 20–30 of your own signs** in the demo room, camera, and distance.
   Per `HANDOFF.md` this is the single largest gap between 52% and something
   stage-worthy, and everything downstream already works
   (`ingest_recordings.py` → `eval_on_takes.py`). Prioritise the eight
   `UNREACHABLE` glosses above — they are what a patient actually needs.
2. **Generate the gloss table** with your API key (one command, above).
3. **Bhashini** for coverage beyond the table. Free tier is PoC-only — cache
   everything demo-critical.

### Admin

- Confirm with the SPOC: the 2026 per-college nomination quota, **whether one team
  may submit more than one idea** (this gates reusing the idea for Travel & Tourism),
  and the real deadline — the portal says 20 Sept, secondary sources say 30 Sept.

### Admin

- Confirm with the SPOC: the 2026 per-college nomination quota, **whether one team
  may submit more than one idea** (this gates reusing the idea for Travel & Tourism),
  and the real deadline — the portal says 20 Sept, secondary sources say 30 Sept.

---

## 13. Invariants — breaking these silently destroys accuracy

1. **`train/features.py` ↔ `app/src/lib/features.ts` stay bit-identical.**
   Run `test_parity.py` after touching either.
2. **Never overwrite `gloss_classifier.keras` directly.** Let `train.py` promote.
3. **If you change augmentation, verify a metric exists that can detect the change.**
   Perspective augmentation was once built, evaluated on the *far-camera* held-out
   set (which contains no close-range footage), measured as a regression, and
   reverted — the correct fix was thrown away using a metric structurally incapable
   of seeing it.
4. **Devanagari needs `\p{M}`** in any tokeniser. Every Indic vowel sign is a
   combining mark; without it `नमस्ते` becomes `नमस त` and matches nothing.
5. **Zenodo is unreliable** — expect stalls, failed resumes, and at least one
   archive that silently truncated. All fetchers retry indefinitely; two curls
   writing one path once collapsed 1.1 GB into 14 MB, hence the single-instance lock.
