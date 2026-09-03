# Contributing to Setu

Setu is a two-way bridge between Indian Sign Language and India's spoken
languages. Built for SIH 2026 (Student Innovation), submitted under both the
MedTech and Travel & Tourism themes from one codebase.

## Run it in two minutes

```bash
git clone <repo> && cd sih/app
npm install
npx vite --port 5174
```

Open `https://localhost:5174`. **The trained model is committed**, so the app
works immediately — no dataset download, no training, no API key. Three modes:

| mode | needs a camera? |
|---|---|
| **Signs to speech** | yes — or click *Replay held-out clips* to run real clips through the live pipeline |
| **Conversation** | no — there is a text box beside the mic |
| **Record signs** | yes — this is how you contribute data |

To use the camera from a phone you need HTTPS on your LAN address, which the
dev server already provides — use the `Network:` URL it prints, not localhost.

## The one thing that would help most

**Record signs, as a new signer.**

Accuracy is ~90% when the same people appear in training and test, and **56.8%**
when they do not. That gap is the whole problem, and it is a data problem:
INCLUDE is 7 signers from one school in Chennai, one room, one camera distance.

We measured what would close it. Two architecture experiments (a 468-point face
mesh, and an SL-GCN graph network) both came back negative, and the ablation's
own conclusion was:

> *More GROUPS, not more categories, is what would settle it.*

So: open **Record signs**, record 30–50 takes each of 20–30 signs, vary your
distance, angle and lighting between takes, and send the downloaded JSON. One
new signer is worth more than any model change we have been able to measure.

Highest value are the eight things a patient needs that INCLUDE simply cannot
express — there is no gloss for them in the 264, so the classifier can never
emit them (`app/tools/glossCorpus.ts`, `UNREACHABLE`):

> **Water · Help · Yes · No · Pain · Name · Please · Hotel**

```bash
.venv/bin/python    train/ingest_recordings.py setu-recordings-*.json
.venv-tf/bin/python train/eval_on_takes.py     setu-recordings-*.json
```

## Rules that are not style preferences

**1. `train/features.py` and `app/src/lib/features.ts` must stay bit-identical.**
They are one contract expressed twice. If they drift, the model trains on one
representation and runs on another: offline accuracy stays high while live
accuracy collapses, with no error anywhere.

```bash
.venv/bin/python train/test_parity.py     # 8/8, exact zero difference
```

Run it after touching either file. It is verified to catch real drift.

**2. Never overwrite `models/gloss_classifier.keras` by hand.** `train.py`
promotes a checkpoint only if it beats the recorded best. An earlier version
wrote a single file and destroyed the best model across three runs.

**3. If you change augmentation, check a metric exists that can SEE the change.**
Perspective augmentation was once built, evaluated on a far-camera held-out set
containing no close-range footage, measured as a regression, and reverted. The
correct fix was thrown away by a metric structurally incapable of detecting it.

**4. Report the held-out number, not the random-split one.** 56.8%, not 90.3%.
Explaining that gap is the most credible thing this project can say.

## Environments

Three Python virtualenvs, deliberately:

| venv | Python | why separate |
|---|---|---|
| `.venv` | 3.14 | data work, numpy only |
| `.venv-tf` | 3.11 | TensorFlow has no 3.14 wheels |
| `.venv-mp` | 3.11 | MediaPipe pinned 0.10.14; 1.x's Python HolisticLandmarker is broken |

The front end needs only Node.

## Getting the data (optional)

Nothing below is needed to run the app or contribute recordings.

```bash
cd data && ./fetch_video.sh      # INCLUDE video, 56.8 GB, CC BY 4.0
./supervise.sh                   # keeps fetch + extract alive across failures
HF_TOKEN=hf_xxx ./data/fetch_cislr.sh   # CISLR, 71 signers, gated AFL-3.0
```

Zenodo does not honour byte ranges, so there is no resume — every interrupted
part restarts from zero. The fetchers verify each file against the exact byte
count from the API before promoting it, because a truncated archive otherwise
fails much later, in unzip, after the download cost is already paid.

## Where to read next

- **`ARCHITECTURE.md`** — every artefact, shape, and measured number
- **`HANDOFF.md`** — why decisions were made, and which bugs cost real time

Both are kept honest about what does not work. Please keep them that way:
a negative result recorded clearly is worth more here than an optimistic claim.
