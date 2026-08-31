# Setu: architecture and system state

Two-way bridge between Indian Sign Language and India's spoken languages.
Target: **SIH 2026, Student Innovation**, submitted under **two themes**,
**MedTech** (SIH26196) and **Travel & Tourism**.

> **One system, two settings: not two products.** The recognition stack is
> entirely domain-neutral: one 264-sign classifier, one feature contract, one
> segmenter. INCLUDE is a general lexicon, and "Doctor" and "Train Station" are
> the same kind of sign to the model. Everything below the UI is unaware of which
> setting it is running in. See §11.

This document is the precise reference: where every artefact lives, what shape it
has, and what is verified versus assumed. `HANDOFF.md` covers *why* decisions were
made and which bugs cost time, read that for rationale, this for facts.

Last audited: **2 September 2026**.

---

## 1. Repository map

```
~/sih/
├── data/                   8.5 GB   all corpora and the preprocessed tensor
│   ├── Pose_Signs/         640 MB   INCLUDE pose release, TRAINS THE CURRENT MODEL
│   ├── video_landmarks/    231 MB   468-pt face-mesh re-extraction, IN PROGRESS
│   ├── video/              143 MB   raw INCLUDE video, transient (deleted after extract)
│   ├── poses/              5.0 GB   AUTSL, MS-ASL, WLASL, GSL, LSA64, NOT ISL, unused
│   ├── INCLUDE-poses.zip   632 MB   original archive of Pose_Signs
│   ├── cislr/              1.5 GB   CISLR v1.5-a, SECOND CORPUS, gated (§2.2)
│   ├── cislr_landmarks/    319 MB   612 CISLR clips re-extracted to landmarks
│   ├── dataset.npz          83 MB   preprocessed training tensor (INCLUDE only)
│   ├── dataset_merged.npz   98 MB   INCLUDE + CISLR, carries a `corpus` array
│   ├── signer_index.json   384 KB   clip -> {class, signer group}
│   └── signer_profiles.npz 504 KB   body-ratio clusters used to derive groups
├── models/                  39 MB   checkpoints, labels, metrics, holistic .task
├── run/                             job state and results (cislr_eval.json)
├── train/                  180 KB   the Python pipeline (21 scripts)
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
| `.venv-mp` | 3.11 | MediaPipe pinned **0.10.14**, 1.x's Python `HolisticLandmarker` raises `Check failed: service_ Service is unavailable` | ✅ |

`.venv-mp` prints a harmless `MessageFactory has no attribute GetPrototype`
protobuf warning on import; `run_extract_loop.sh` filters it. `mp.solutions.holistic.Holistic`
constructs correctly: confirmed 30 Aug.

---

## 2. Data

### 2.1 What actually trains the model

**`data/Pose_Signs/`**: the OpenHands INCLUDE pose release. This is the *only*
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
> throughout `train/`: arbitrary pickles execute code on load.

### 2.2 CISLR: the second corpus (added 5 Sept)

**`data/cislr/`**: CISLR v1.5-a, Exploration-Lab. Gated on Hugging Face,
AFL-3.0, licence accepted per-account. `data/fetch_cislr.sh` (`FETCH_ALL=1`)
pulls the 1.1 GB video zip; `data/` is gitignored, so the fetch script is the
reproduction path.

- **7,050 clips, 4,765 glosses, 58 categories**
- **clips per gloss: min 1, median 1, max 13**

That median is the whole constraint. CISLR is a *retrieval* corpus, "is this
sign in that video?": not a classification corpus. A 4,765-class model cannot
be trained on one example each, and adding 4,545 single-example glosses to
`labels.json` would manufacture words the model claims to know and cannot
recognise. We import **none** of them.

What we do import is the overlap:

- **612 clips carry 220 of our 264 labels** (median 2 clips per class)
- all eight `UNREACHABLE` glosses of §2.4 are present in CISLR
- 610 survive preprocessing -> **4,894 clips total, 7 signer groups**

Two ingestion details that are load-bearing:

**Trimming.** CISLR is framed tighter than INCLUDE, hands at rest sit *below
the crop*: and clips run longer, so roughly half of each is lead-in and
lead-out at rest. `features.resample` strides over the whole clip, so untrimmed
CISLR spends half its 32 frames on stillness where INCLUDE spends almost none,
and the model would learn "long still lead-in" as a corpus tell rather than a
sign. `preprocess_cislr.py` cuts to the hand-visible span:

| | untrimmed | trimmed | INCLUDE |
|---|---|---|---|
| frames per clip | 86 | 48 | 61 |
| hand-present fraction | 0.44 | 0.79 | 0.89 |

Hand presence rather than motion energy, because the target is hands *out of
frame*, not stillness: motion energy would also cut the hold at the end of a
sign, which carries meaning.

**Signer IDs.** CISLR clips take groups 3-6, recovered by `signers.py`
proportions and never merged into INCLUDE's 0-2. Held-out-group evaluation is
only honest if a group is one set of people; letting a CISLR clip land in
INCLUDE group 0 would put the same corpus on both sides of the split, the same
class of error as the 99.8% calibration run (§12).

A `corpus` array rides along in `dataset_merged.npz` so evaluation can train on
one corpus and test on the other. That is §5.1, and it is the most important
measurement in this document.

### 2.3 The face-mesh re-extraction (blocks FULL_FACE)

INCLUDE's pose release carries **no face mesh**, only 11 coarse pose landmarks
(nose, 6 eye, 2 ear, 2 mouth-corner). That is enough for head nod/shake/tilt and
nothing else. ISL marks **yes/no questions with raised eyebrows, wh-questions with
furrowed brows, and negation with a head shake**, a hands-only model renders a
question as a statement, which is a correctness failure, not a polish one.

So the raw video (Zenodo 4010759, **56.8 GB, 46 parts**) is being re-extracted with
full Holistic to recover the 468-point mesh.

**Status: COMPLETE.**

| | |
|---|---|
| Downloaded | **56.8 / 56.8 GB, all 44 archives** |
| Clips with face mesh | **~4,280 of 4,284** (a couple of source videos are unreadable) |
| Categories | all 15 |

Having finished it, the honest note is that it did not change the answer: §9
shows the face mesh does not improve recognition. The extraction was still worth
doing: the question could not be settled without it, but `FACE_MODE` stays
`HEAD_ONLY`.

Disk is safe: `run_extract_loop.sh` deletes each ~1.3 GB archive immediately after
pulling landmarks (104 clips → ~38 MB), so peak usage stays at roughly one part.
**69 GB free** at audit time, against a transient footprint of ~2 GB.

### 2.4 Other sign languages: present but unused

`data/poses/` holds AUTSL, MS-ASL, WLASL, GSL, LSA64 (5.0 GB). These are *not*
ISL. They are kept for a possible cross-lingual pretraining experiment and are
not part of any current result. Do not cite them as ISL data.

### 2.5 What does not exist: do not promise these

- **Dialect labels.** No published ISL corpus tags dialect.
- **A conversation vocabulary.** INCLUDE is a *lexicon*: "Actor", "Election",
  "Monsoon". It has no *yes*, *no*, *please*, *help*, *where*. This is why the
  demo's conversation mode uses a separate curated phrase set.
- **FDMSE-ISL** (40k clips): no public download, author request only.
- **CISLR**: gated on Hugging Face (`gated: auto`, AFL-3.0). **71 signers**,
  ~4,700 words, only 1.59 GB, and it ships pre-extracted I3D features alongside
  the video. This is the single most valuable dataset available to this project,
  because §9 concluded the bottleneck is signer diversity and CISLR is ~10x
  INCLUDE's. `data/fetch_cislr.sh` is written and tested; it needs a human to
  accept the licence once (which includes agreeing to share contact details) and
  supply `HF_TOKEN`.

### 2.6 Known bias: state this in the pitch

INCLUDE is **7 signers from one school in Chennai, one room, one camera distance**.
`data/signer_index.json` groups are **body-type clusters recovered from pose-stable
ratios**, not identified individuals, the official split is offline (Google Drive
404). This is stricter than a random split and weaker than a true signer-disjoint
one. Say exactly that.

---

## 3. Feature pipeline

`train/features.py` ↔ `app/src/lib/features.ts` are a **bit-identical contract**.
If they drift, the model trains on one representation and runs on another; offline
accuracy stays high while live accuracy collapses with no visible cause.

```bash
.venv/bin/python train/test_parity.py    # 8/8, exact zero difference, MUST pass
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

1. **`select_points`**, `(T,75,3) → (T,65,3)`, dropping legs.
2. **`to_unit`** *(dataset only)*, divide `x` by `vid_shape[0]`, `y` by `vid_shape[1]`.
   The browser's MediaPipe already returns unit coordinates, so this step only
   brings the dataset into the same space. **Everything below is the shared contract.**
3. **`anchor`**: subtract the **shoulder midpoint** (landmarks 11, 12), then divide
   by 2-D shoulder span. Position- and scale-invariant.
   *Anchoring at the shoulders rather than the head is deliberate: head nod, shake,
   and tilt survive normalisation and stay visible to the model.*
4. **`resample`**: any `T` → exactly 32 frames by nearest-index striding.
5. **`standardise`**: flatten, then `(v - mean) / std` over the whole clip.

Scale is provably a no-op after standardisation (verified to 4e-16). **Perspective
is not**: see §5.

### FULL_FACE subset (`train/face.py`)

Using all 468 mesh points would swamp 65 body points, so FULL_FACE adds a curated
**48**: 10+10 eyebrow, 6+6 eye aperture (EAR-style), 8+8 lip outer/inner.

---

## 4. Model

1-D CNN over the temporal axis. **Not** a transformer, TFLite and TF.js have no
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

## 5. Results: and which number to report

Two models ship. The app picks by setting: health loads the clinical one, travel
keeps the general one.

| model | classes | clips | held-out signer, close range |
|---|---|---|---|
| **clinical** (health) | **38** | 740 | **73.9% top-1 / 96.3% top-5** |
| general (travel) | 264 | 4,894 | 64.8% top-1 / 86.6% top-5 |

Chance is 2.6% and 0.38% respectively. Per-group top-1 for the clinical model:
**72.9 / 80.5 / 68.2**.

> **Report the held-out-signer number, never a random split.** A random split
> puts the same person on both sides and inflates by roughly 38 points. Being
> able to explain that gap is the most credible thing this project can say;
> most competing work quotes the inflated figure.

**Quote 73.9% for the clinical model, and say what it cannot do in the same
breath.** It has no sign for *pain*, *water*, *help*, *yes* or *no*: those are
absent from every ISL corpus available to us, and they live on the phrase board
instead. A headline accuracy without that sentence is misleading.

### 5.0 How these numbers were earned, and what did not work

Cutting the vocabulary did nearly all of it. Holding data, recipe and protocol
fixed and varying only the class count, on held-out INCLUDE group 0:

| classes | top-1 | top-5 |
|---|---|---|
| 20 | 76.1% | 94.8% |
| 80 | 66.0% | 92.8% |
| 264 | 56.1% | 81.4% |

Each extra class is another way to be wrong across ~19 clips from about ten
signers, and most of the 264 are words a consultation never needs.

Everything tried on the modelling side, measured against a control:

| change | effect |
|---|---|
| **38-class clinical vocabulary** | **+8.4** |
| **stacked encoder** (SSL then supervised ASL) | **+3.1** on identical arms |
| supervised ASL pretraining (38,758 clips) | +18.4 over no pretraining |
| self-supervised on 13,662 ISL clips, alone | +1.3 |
| hand-dropout augmentation | -0.8 |
| test-time augmentation | -8.1 |
| face mesh (FULL_FACE), SL-GCN | ~0, see §9 |

Five consecutive negative results are what prompted the vocabulary experiment.
When five methods do nothing, the method is not the constraint.

The stacked encoder is the only use in which the ISL dictionary paid for
itself. Trained on alone it was worth +1.3; used as the INITIALISER for the ASL
stage it is worth +3.1 on top of ASL, because the two signals compound instead
of one overwriting the other.

### 5.0.1 Superseded figures

Earlier revisions of this document led with **52.0%** and **40.4%**. Both came
from a protocol that selected the stopping epoch on the test set, so both were
optimistic by an unmeasured amount. They are recorded here only so an old slide
or README can be matched to what replaced it, and neither should be quoted.

### 5.1 Cross-corpus: the number that reframes the other numbers (5 Sept)

Every figure above is **within-corpus**. INCLUDE's held-out signer is a
different person in the same room, on the same camera, under the same
recording protocol, at the same school. Holding out the signer removes one of
the five things that vary in the field.

CISLR is the first data we have where all five differ. `train/eval_cislr.py`
runs four arms: same model, same augmentation, same 60-epoch budget, same
validation signer: differing only in which clips train:

| arm | train -> test | close top-1 | top-5 |
|---|---|---|---|
| A | INCLUDE -> INCLUDE group 0 | 28.2% | 51.0% |
| B | INCLUDE + CISLR -> INCLUDE group 0 | **29.6%** | 53.8% |
| C | INCLUDE -> **CISLR** | **2.1%** | 7.4% |
| D | INCLUDE + CISLR -> CISLR group 6 | 4.0% | 21.5% |

264 classes; chance is 0.38% top-1, 1.9% top-5.

**Arm C is the finding.** A model trained on INCLUDE scores 2.1% on a different
corpus: about five times chance, and functionally nothing. Set against the
28.2% of arm A and the ~100% of a random split, the ladder is:

    held-out clip, same signers      ~100%
    held-out signer, same corpus      28.2%
    held-out corpus                    2.1%

Most of what the model knows is INCLUDE, not ISL. That single fact explains
why three independent architecture experiments, face mesh (§9), SL-GCN (§9),
calibration (§12): all failed to move the number: none of them addressed what
the model is actually keying on.

**Arm B says more signers do help, and quantifies how slowly.** 610 CISLR
clips, a 36% increase in training data, bought +1.4 points. Reaching a
deployable score is not a few hundred more clips; on this slope it is thousands,
from many more people and many more rooms.

**Arm D is weak evidence, and is labelled as such.** Training on some CISLR
roughly doubles top-1 on held-out CISLR (2.1% -> 4.0%) and triples top-5
(7.4% -> 21.5%), so the domain gap is at least partly learnable. But its test
set is 149 clips with only 100 of 264 classes represented in training, and it
early-stopped at 22 epochs. Treat the direction as real and the magnitude as
noise.

#### A protocol correction that applies to every number above

`train.py:118` passes the **test set** as `validation_data` with
`restore_best_weights=True`, so the stopping epoch is selected on the test set.
Every figure `train.py` and `train_ablation.py` have printed is optimistic by an
unmeasured amount: 52.0%, 56.8% and 40.4% included.

`eval_cislr.py` carves validation out of TRAIN instead and touches the test set
once, at the end. That, plus training on group 2 alone (group 1 is held for
validation), is why arm A reads 28.2% where the superseded figure for a similar
split was 40.4% (§5.0.1). **Arm A is the honest re-measurement; compare B
against A, never against the old number.**

#### What to say publicly

Quote the ladder, not a single number. "52% on held-out signers within one
corpus, 2% across corpora, and here is why that gap exists" is a stronger and
more defensible claim than any single figure, and it is the claim the evidence
supports.

### Three bugs that made the live camera path fail

Each was hidden behind the previous one. All fixed; do not reintroduce them.

1. **Two landmarkers, incompatible z conventions.** The app ran `PoseLandmarker` +
   `HandLandmarker` separately, but INCLUDE was extracted with *Holistic*, one
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

   Fixed by `app/src/lib/segment.ts`, motion-energy segmentation that starts on a
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

# CISLR: second corpus (§2.2). Needs a HF account with the licence accepted.
FETCH_ALL=1 HF_TOKEN=... ./data/fetch_cislr.sh   # 1.1 GB video zip
.venv-mp/bin/python train/extract_cislr.py       # 612 clips -> landmarks, ~2 h
.venv/bin/python    train/preprocess_cislr.py    # -> data/dataset_merged.npz
.venv-tf/bin/python train/eval_cislr.py          # 4 arms, ~9 h -> run/cislr_eval.json
```

`data/dataset.npz` currently holds:

```
X      (4284, 32, 65, 3) float32
y      (4284,)           int32
signer (4284,)           int32
labels (264,)            <U16
```

`data/dataset_merged.npz` adds CISLR (§2.2). Same layout plus a `corpus` array,
`0 = INCLUDE, 1 = CISLR`, which is what lets §5.1 train on one and test on the
other:

```
X      (4894, 32, 65, 3) float32
y      (4894,)           int32
signer (4894,)           int32     0-2 INCLUDE, 3-6 CISLR, never mixed
corpus (4894,)           int32
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
| **Conversation** | `HearingSide.tsx` | works reliably, no camera, **this is the demo** |
| **Record signs** | `Recorder.tsx` | works, capture your own vocabulary |
| **Signs to speech** | `SignBridge.tsx` | works, ~52% over 264 classes |

### Runtime chain

```
webcam → HolisticLandmarker (GPU, VIDEO mode, CDN wasm 0.10.14)
       → segment.ts        motion-energy: burst starts, 6 still frames end
       → features.ts       the §3 contract, mirrored from Python
       → classifier.ts     tf.loadGraphModel, input [1, 32, 195]
       → gate.ts           StabilityGate.once(), one decision per segment
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
   ISL has its own grammar, "STATION GO WHERE", not "where do I go for the
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
5. **The model does not generalise across corpora.** Trained on INCLUDE and
   tested on CISLR it scores **2.1%** against 0.38% chance (§5.1). Held-out
   *signer* accuracy overstates field accuracy by more than an order of
   magnitude, because a held-out INCLUDE signer still shares the room, camera,
   protocol and school. Recognition must be presented as experimental. The
   phrase board: 100% correct, offline, every time, is the surface that
   actually works, and the UI already treats it as primary.
6. **Published figures were selected on the test set.** `train.py` uses the test
   set as validation with `restore_best_weights` (§5.1), so 52.0%, 56.8% and
   40.4% are all optimistic by an unmeasured amount.

---

## 9. The ablation: three arms, two settled questions

Two architecture changes were proposed and both were measured against the same
baseline, on the same clips, signers, classes, seed and augmentation draws. Only
the thing under test varied.

### Final result (3,003 clips, 188 classes, chance 0.5%)

| arm | far | close | vs baseline (close) |
|---|---|---|---|
| **HEAD_ONLY** Conv1D, 65pt | 58.2% | **56.8%** | baseline |
| **FULL_FACE** Conv1D, 113pt | 56.2% | 56.5% | **-0.3 pp** |
| **SL-GCN** graph, 65pt | 51.3% | 49.5% | **-7.3 pp** |

**FULL_FACE: inconclusive, and that is now a strong signal.** The delta is
smaller than the 2.0 pp between-group spread, so three groups cannot separate
the arms. But this run *includes* Pronouns and Society, the categories where
ISL marks questions and negation, so it is no longer a vocabulary gap. Across
three runs as data grew 685 -> 1,441 -> 3,003 clips the close delta went
**+1.5 -> -1.4 -> -0.3 pp**: the sign flipped twice and the magnitude shrank.
A real effect does not behave that way. **Do not set `FACE_MODE = "FULL_FACE"`.**

**SL-GCN: conclusive, and it loses.** -7.3 pp against a 3.1 pp spread, and it
lost on *every* group (-3.6, -7.2, -11.1 pp). Held at 0.90x the baseline's
parameters, so this is not a capacity artefact. AI4Bharat report 93.5% for
SL-GCN on INCLUDE, but on a same-signer split; under a signer-disjoint protocol
on this data the graph architecture does not transfer. **Keep the Conv1D.**

Both arms point the same way, and the harness says so itself:

> *More GROUPS, not more categories, is what would settle it.*

That is the finding of this whole section: the bottleneck is signer diversity,
not architecture. Two independent attempts to improve the model failed; the
measurement kept pointing at the data.

---

### Original scope note: FULL_FACE

`FACE_MODE = "FULL_FACE"` widens the input 65 → 113 points. There is a
linguistic reason to expect a gain (eyebrows and mouth carry ISL's question and
negation marking), but that is an argument, not evidence, and `HANDOFF.md` §6
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

> **Scope.** 685 clips, 51 classes, chance 2.0%, the Adjectives categories only.
> This is **not** the 264-class result and must never be quoted as one. It answers
> the narrower question of whether the face block earns its place, early enough to
> act on. Note also that adjectives carry relatively little non-manual grammar; the
> question-heavy categories (Pronouns, Society) are still downloading, so a null
> result here is weak evidence, not a refutation.

### Result (31 Aug): the decisive re-run, and it does not support FULL_FACE

Ran on **1,441 clips / 98 classes** across Adjectives, Animals, **Pronouns** and
**Society**: the question-heavy categories the first run lacked. Chance 1.0%.

| arm | far camera | close (laptop) |
|---|---|---|
| HEAD_ONLY (65 pt) | **60.1%** | **58.5%** |
| FULL_FACE (113 pt) | 59.7% | 57.1% |
| delta | −0.4 pp | −1.4 pp |

Per-group close delta: `g0 +2.1pp`, `g1 −1.3pp`, `g2 −5.1pp`, spread 3.0 pp
against a mean of 1.4 pp. Still formally **INCONCLUSIVE**: three signer groups
cannot resolve a difference this small.

**But compare the two runs, because that is the real finding:**

| run | data | far Δ | close Δ |
|---|---|---|---|
| 30 Aug | 685 clips, 51 cls, Adjectives | −2.4 pp | **+1.5 pp** |
| 31 Aug | 1,441 clips, 98 cls, **+Pronouns/Society** | −0.4 pp | **−1.4 pp** |

**The close-range delta flipped sign**, the direction that looked promising
reversed once the categories that should *favour* the face block were added. A
real effect does not invert when you strengthen the conditions for it. This is
noise, not signal.

That kills the reading the first run invited. It is no longer the
perspective-augmentation shape from `HANDOFF.md` §6, where far lost and close
gained; FULL_FACE now loses on both arms.

**Conclusion: do not set `FACE_MODE = "FULL_FACE"`.** Not because it is proven
harmful: it is not, but because two runs, the second on the categories designed
to show its benefit, produce no measurable gain. The 48 face points cost 74% more
input width for nothing detectable.

**What would still change the answer** (in order of expected value):

1. **More signer groups.** The blocker is now power, not vocabulary: 3 groups
   against a ~1 pp effect. Recording your own signers adds groups directly.
2. **The 48-point subset may be wrong.** Eyebrow height varies 0.077 within a
   clip, so signal exists: but a 32-frame resample may simply smear a brief
   eyebrow raise away. Worth testing temporal resolution before discarding the
   face entirely.
3. **Augmentation was tuned for body landmarks.** Perspective warping may be
   actively damaging face points.

### Result (30 Aug): the first run, superseded above

| arm | far camera | close (laptop) |
|---|---|---|
| HEAD_ONLY (65 pt) | **69.6%** | 62.6% |
| FULL_FACE (113 pt) | 67.3% | **64.1%** |
| delta | **−2.4 pp** | **+1.5 pp** |

Per-group close delta: `g0 +0.8pp`, `g1 −4.8pp`, `g2 +8.6pp`, **spread 5.5 pp
against a mean of 1.5 pp.**

**The mean is smaller than the spread, so this subset cannot separate the two
arms.** Three groups give no power to call a 1.5-point difference, and reporting
it as a win would be exactly the overclaiming this project avoids everywhere
else. `train_ablation.py` prints `INCONCLUSIVE` for this reason.

What *is* worth noting is the **direction**: FULL_FACE loses on far camera and
gains at close range. That is the same shape as the perspective-augmentation
result in §5: and the same trap. Judged on the far-camera number alone,
FULL_FACE looks like a 2.4-point regression and would be rejected. The app runs
at close range.

So: do not switch `FACE_MODE` on this evidence, and do not discard the change
either. Re-run when Pronouns and Society are extracted, those are the categories
where non-manual grammar actually lives.

Full numbers in `models/ablation_face.json` (`conclusive: false`).

### Fetch order was the hidden blocker (fixed 30 Aug)

Zenodo lists the 46 parts alphabetically, which put **Pronouns at 39–40 and
Society at 43–45**: the decisive categories arriving only after ~50 GB. The
ablation could not be settled until essentially the whole corpus was down, by
which point the decision would have been made by default.

`fetch_video.sh` now orders by linguistic priority instead:

```
Pronouns → Society → Greetings → People → (everything else, alphabetical)
```

Same 46 files, same total bytes; only the order changes, and `.done` /
`.extracted` markers make it safe to reorder mid-run. This moves a conclusive
ablation from ~35 parts away to ~5.

**Re-run the ablation once Pronouns and Society are extracted**, that is the
result worth quoting, and it is the one that decides `FACE_MODE`.

This is automated. `run_ablation_when_ready.sh` waits for the five decisive parts
and then runs both stages unattended:

```bash
./run_ablation_when_ready.sh          # logs to /tmp/setu-ablation.log
```

It polls once a minute with an 8-hour ceiling, prints a progress line every 15
minutes, and fails loudly naming the missing parts rather than hanging if the
fetcher dies. Safe to re-run, `preprocess_face.py` globs whatever is in
`data/video_landmarks`, so it always uses everything extracted so far.

Started 30 Aug alongside the reordered fetch; 6.3 GB of decisive data at roughly
1.7 GB/h.

---

## 10. Gloss reordering: implemented, offline

ISL word order is not spoken word order. This is now handled, and the design
constraint that shaped it is worth stating: **the stage demo must never depend on
a live network call**, and **an API key must never reach the browser**.

Both are satisfied by generating the reorderings ahead of time.

```bash
# once, with your own key: never the browser's
ANTHROPIC_API_KEY=sk-... node --experimental-strip-types app/tools/buildGlossTable.ts
```

| file | role |
|---|---|
| `app/tools/glossCorpus.ts` | 73 gloss sequences (medical / travel / civic / social) |
| `app/tools/buildGlossTable.ts` | generator, Claude Opus 5, structured output, batched, resumable |
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
| `gloss-order` | signs read in signed order | **no**, badged amber |

This preserves the original caution rather than discarding it. Reordering happens
only where a verified entry exists; everywhere else the app still refuses to
pretend, and says so on screen.

### What the vocabulary cannot express

Validation surfaced this concretely, `UNREACHABLE` in `glossCorpus.ts`:

> **water · help · yes · no · where · please · pain · name**

None of these are INCLUDE classes, so no amount of translation work makes them
reachable. "I need water" is not a hard sentence; it is simply unsayable with this
lexicon. That is the checkable form of "INCLUDE is a lexicon, not a conversation
vocabulary": and the strongest argument for recording your own signs.

---

## 11. Deployment domains: Healthcare and Travel & Tourism

The app ships both SIH themes from one codebase. A switch in the top bar changes
the setting; the choice persists per device (`setu.domain`).

**What a domain is allowed to change, and nothing else:**

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

- **Quick phrases run FORWARD**: tapped by the Deaf user, spoken aloud. They
  need only to be things a person wants to say; no pose frames required.
- **Synonyms run REVERSE**: spoken word → played ISL skeleton. These *do* need
  bundled pose frames.

`checkDomains.ts` reports 19 warnings for glosses that are recognisable but not
playable: most importantly **`Location`**, the stand-in for WHERE that appears
in nearly every "where is X" phrase. `textToGlosses` already guards on the
library, so an unplayable match is reported as *skipped* rather than queueing a
sign that renders as an empty player. The warnings are informational, not bugs , 
but they are the list of signs worth recording first for the reverse direction.

---

## 12. Confidence calibration: the app was confidently wrong

Softmax confidence on this model is **not a probability**. Measured on 1,566
clips from a signer group the model never trained on, at close range:

| the model says | it is actually right |
|---|---|
| 0.90 | **48%** |
| 0.99 | **70%** |

Expected Calibration Error: **34.1 pp**. At the old gate (`FLOOR = 0.75`,
uncalibrated) 54% of segments passed and only 57.9% of those were correct, so
**42% of everything spoken aloud was wrong**, stated confidently, to a patient.

### Temperature scaling

Divide the logits by a single scalar `T` before softmax, fitted offline by
minimising NLL on held-out data. `T = 2.69`.

| | before | after |
|---|---|---|
| ECE | 34.1 pp | **5.0 pp** |
| says 0.90, is right | 48% | **91%** |
| top-1 | 40.4% | **40.4%**: unchanged |

Accuracy cannot change: dividing every logit by the same positive number
preserves their order. **The model is no better; it now admits it.** That is
what makes a threshold meaningful.

The shipped graph model has softmax baked into its final layer, so raw logits
are unavailable at runtime. `log(p)` recovers them up to a constant and softmax
is invariant to that constant, so re-softmaxing `log(p)/T` is exactly
equivalent. Verified against the Python implementation on 200 samples: max
confidence error 7.3e-3 (float32 quantisation) and **zero argmax mismatches**.

### Three bands, not one floor

| calibrated conf | behaviour | speaks | and is right |
|---|---|---|---|
| >= 0.70 | spoken aloud | ~10% | 84.3% |
| >= 0.40 | shown, "please confirm" |, |, |
| < 0.40 | "unclear: sign again" |, |, |

A single floor forces a bad trade: high enough to trust and the app is silent
nine times in ten; low enough to feel responsive and two in five are wrong.

`app/src/lib/calibrate.ts`. **Refit `T` after any retrain**, it is a property of
the trained weights, not of the architecture.

---

## 13. Remaining work

Reordered 5 Sept. §5.1 changed the priorities: the bottleneck is not the model,
and it is not the face mesh. It is that the model has learned one corpus.

### Do now: highest value per hour

1. **Record your own signs, in the room the app will run in.** This is now
   unambiguously first. Arm C says a model trained elsewhere scores 2.1% here;
   the only data guaranteed to match deployment conditions is data recorded in
   them. Everything downstream already works (`ingest_recordings.py` ->
   `eval_on_takes.py`). Prioritise the eight `UNREACHABLE` glosses, they are
   what a patient actually needs, and CISLR has clips of all eight to check
   against.
2. **Label recognition as experimental in the UI.** At 2.1% cross-corpus, a
   confident-looking transcript is a liability in a hospital. The phrase board
   is already primary; the recognition panel should say what it is.
3. **Fix the evaluation protocol before running any more experiments.** Move
   `train.py`'s validation off the test set (§5.1). Until that is done, no
   ablation this repo runs can be trusted to the precision it reports.

### The real fix, and its size

Arm B: 610 new clips, +36% training data, **+1.4 points**. Extrapolating, closing
the gap to a deployable score needs thousands of clips from many signers in many
rooms: not a weekend of recording. Options, cheapest first:

- **More corpora.** ISLTranslate/iSign are continuous ISL; segmenting them into
  isolated signs is work but they are free and already identified (§2.4).
- **Domain-adversarial or corpus-balanced training.** Arm D shows the gap is at
  least partly learnable. A model penalised for predicting *which corpus* a clip
  came from is the standard remedy and costs no new data.
- **Self-recorded data at scale.** Highest quality per clip, lowest throughput.

### Deprioritised by §5.1

- **FULL_FACE retrain.** Face-mesh extraction stalled at 8/46 Zenodo parts, and
  §9 already measured FULL_FACE as a 2.4-point regression on the honest split.
  The signal is real (eyebrow height varies 0.077 within a clip) but it is not
  what is limiting the model. Do not spend the compute until cross-corpus
  accuracy is off the floor.
- **SL-GCN.** Same reasoning. §9 settled that capacity is not the constraint.

### Admin

- Confirm with the SPOC: the 2026 per-college nomination quota, **whether one team
  may submit more than one idea** (this gates reusing the idea for Travel & Tourism),
  and the real deadline: the portal says 20 Sept, secondary sources say 30 Sept.

---

## 14. Invariants: breaking these silently destroys accuracy

1. **`train/features.py` ↔ `app/src/lib/features.ts` stay bit-identical.**
   Run `test_parity.py` after touching either.
2. **Never overwrite `gloss_classifier.keras` directly.** Let `train.py` promote.
3. **If you change augmentation, verify a metric exists that can detect the change.**
   Perspective augmentation was once built, evaluated on the *far-camera* held-out
   set (which contains no close-range footage), measured as a regression, and
   reverted: the correct fix was thrown away using a metric structurally incapable
   of seeing it.
4. **Devanagari needs `\p{M}`** in any tokeniser. Every Indic vowel sign is a
   combining mark; without it `नमस्ते` becomes `नमस त` and matches nothing.
5. **Zenodo is unreliable**: expect stalls, failed resumes, and at least one
   archive that silently truncated. All fetchers retry indefinitely; two curls
   writing one path once collapsed 1.1 GB into 14 MB, hence the single-instance lock.
