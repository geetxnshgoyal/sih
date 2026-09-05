# Setu: ISL ↔ Regional Language Bridge

> **New here? Read [HANDOFF.md](HANDOFF.md) first.** It covers the environment
> setup (three venvs, and why), the three bugs that made the camera path fail,
> the invariants you must not break, and what is left to do.

Two-way bridge between Indian Sign Language and India's spoken languages.
Target: SIH 2026, Student Innovation, **SIH26196** (MedTech · Software).

## Phase 0: proof of concept (this directory)

Camera → MediaPipe hand landmarks → 3 ISL signs → spoken Hindi/Tamil/Telugu/Bengali/Marathi/English.
No build step, no training, no API key, no network after first load.

### Run

```bash
python3 -m http.server 8000 --directory phase0
```

Open <http://localhost:8000>. Must be `localhost`, `file://` cannot access the camera.

First load pulls the MediaPipe WASM + hand model from a CDN, then caches them.

### Vocabulary

| Gloss | Form |
|---|---|
| HELLO | one open palm, fingers spread (wave adds confidence) |
| HELP | thumb up, four fingers curled; flat other palm beneath = full ISL form |
| THANK YOU | both flat palms brought together |

Chosen to be maximally separable: one open hand vs one closed hand vs two hands.

### Reliability

- confidence floor **0.75**: below it, nothing fires
- **10 of last 14** frames must agree
- **1.8s** cooldown so a held sign does not repeat
- silence beats a wrong word

### What Phase 1 changes

`classify()` in `phase0/index.html` is the only piece that gets replaced, a trained
model over the same landmark stream, same `hands -> {gloss, conf}` signature.
Everything upstream and downstream stays.

## Stack

| Layer | Choice |
|---|---|
| Landmarks | `@mediapipe/tasks-vision` HandLandmarker (WASM + GPU) |
| Gloss model | own 1D-CNN, Keras → TF.js *(Phase 1)* |
| Pretrain data | OpenHands INCLUDE poses, 662 MB, CC BY 4.0 |
| Translation | Bhashini NMT, precomputed table, live only for novel speech |
| TTS | browser `speechSynthesis` |
| Sign playback | INCLUDE videos, CC BY 4.0 *(Phase 2)* |

---

## Phase 1: trained model (current)

264 ISL signs, trained on the OpenHands INCLUDE pose release (4,284 clips).

### Results

| split | top-1 | top-5 |
|---|---|---|
| random (same signers in train and test) | **90.7%** | 98.1% |
| held-out body-type group (mean of 3) | **51.6%** | 75.3% |

**Report the held-out number.** The random split inflates top-1 by **+39.1
points** because the same person appears on both sides. This is why published
sign-language accuracy figures should be read carefully.

Per-group top-1: 43.7% / 67.5% / 43.5%. Group 1 scores highest partly because
holding out the smallest group leaves the most training data.

Chance is 0.4%.

### Verified invariants

Two tests hold the system together:

1. `train/test_parity.py`: Python and TypeScript feature extractors produce
   **bit-identical** output across 8 sequence lengths. Verified to catch real
   drift by injecting two bugs (z-axis in shoulder span, wrong anchor point).
2. `app/src/devParity.ts`: TF.js in the browser reproduces Python's
   prediction to **1.19e-7** on WebGL.

Together they mean the browser computes exactly what the model trained on.

### Pipeline

```bash
.venv/bin/python    train/profile_signers.py   # recover signer groups
.venv/bin/python    train/preprocess.py        # 4,284 clips -> dataset.npz
.venv-tf/bin/python train/train.py             # random + held-out splits
.venv-tf/bin/python train/export_tfjs.py       # -> app/public/model (2.4 MB)
.venv/bin/python    train/test_parity.py       # must pass before training
```

Two venvs because TensorFlow does not support Python 3.14; `.venv-tf` is 3.11.

### Known limitations

- **Signer groups are body-type clusters, not identified individuals.** The
  official INCLUDE split is offline (Google Drive 404), so groups were recovered
  from pose-stable body ratios. Stricter than a random split, weaker than a
  true signer-disjoint one.
- **Training data is South-India-skewed**, 7 signers from one Chennai school.
- **No facial grammar.** The pose release has 11 coarse face points and no face
  mesh, so head movement is trainable but eyebrow and mouth morphemes are not.
  See `train/face.py`; `FULL_FACE` mode needs re-extraction from raw video.
- **Model export goes through a graph model,** not a layers model, Keras 3
  topology is not loadable by tfjs's layers parser.
