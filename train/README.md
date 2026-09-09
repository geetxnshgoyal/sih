# Training and runtime repair

`train/` contains Python programs. It does not contain the corpus, recordings,
preprocessed tensors, or trained Keras checkpoints. Those files are ignored by
Git. Pulling the repository downloads the exported browser model, not its
training data.

The app performs inference locally with TensorFlow.js. There is no separate
inference API server to start. Camera tracking uses the bundled Holistic task
and the WASM files from the same installed MediaPipe package.

## Inspect this checkout

From the repository root:

```sh
python3 train/status.py
python3 train/test_parity.py
python3 train/test_pipeline.py
```

Use `train/status.py` for current counts. The deployed browser model remains
separate from every candidate training run; no data command promotes weights.

## Download and extract in resumable batches

Use the coordinator for future collection. It processes clinical signs first,
publishes landmark files atomically, records provenance under `data/meta`,
and resumes when the same command is run again:

```sh
bash train/setup_pipeline.sh
.venv-mp/bin/python train/pipeline.py sync --priority clinical --batch-size 100
.venv-mp/bin/python train/pipeline.py status
.venv-mp/bin/python train/pipeline.py validate
.venv-mp/bin/python train/pipeline.py retry-failed --batch-size 100
```

Each adapter keeps source video only in a temporary directory. It extracts
pose, both hands, the 48-point eyebrow/eye/lip subset when available, aspect
ratio and FPS, then deletes the video. The manifest retains URL, source,
licence declaration, checksum, label and extraction version. INCLUDE's
published pose archive has no full face mesh and remains body-only.

After one or more batches, build the optional-face tensor and lazy playback
shards:

```sh
.venv-mp/bin/python train/preprocess_face_motion.py
.venv-mp/bin/python train/pipeline.py export-playback
```

Long training remains a separate command:

```sh
.venv-tf/bin/python train/train_face_motion.py
.venv-tf/bin/python train/train_face_motion.py --personal
.venv-tf/bin/python train/export_face_motion.py
```

It writes `models/face-motion-candidate` only. Direct classes require at
least ten clips across three conservative source/signer groups. Sparse classes
remain dictionary shortlist entries; this trainer cannot overwrite deployment.

## Bring recordings into the model

Use **System checks → Open training recorder**. Label each complete ISL sign,
record examples, and download the Setu recordings JSON. Keep the signing hand
and shoulders visible. Intentional one-handed signs can be saved; tracking
warnings should be reviewed before labeling. Record across people, distances,
and lighting. These files contain body landmarks and should be shared deliberately.

Then, using the Python environments described in `HANDOFF.md`:

```sh
python3 train/ingest_recordings.py /absolute/path/to/setu-recordings-file.json
python3 train/test_parity.py
.venv-tf/bin/python train/train.py
```

Ingestion counts only valid, unique takes and requires at least ten per sign.
It creates `data/own.npz`. Training automatically merges it with
`data/dataset.npz` when available, remapping labels and keeping source signer
groups distinct. With recordings alone, training reports a personal holdout,
not unseen-signer accuracy. All takes from the recorder are currently grouped
as one signer: do not claim signer-disjoint evaluation for mixed personal files.

Validation and test subsets are separate. After evaluation, the deployed model
is refitted on **all imported clips**, rather than exporting a fold that omits
an entire signer group. Run checkpoints and their label maps are versioned.
A rejected training run cannot overwrite the promoted label map.

A changed dataset or evaluation protocol is not directly comparable with old
metrics. After reviewing its data and results, run with `--promote` to explicitly
replace the deployed checkpoint. `--data` and `--own` accept custom tensor paths.

Stop the dev server while replacing its model bundle, then export and restart:

```sh
.venv-tf/bin/python train/export_tfjs.py
cd app
npm run dev -- --port 5173
```

Export validates checkpoint dimensions, stages conversion before installation,
uses content-addressed weight files, embeds the label order in the graph metadata,
regenerates the Python reference, and exports playback from every available
training label. It keeps existing playback if the source corpus is absent.
You can also export just playback:

```sh
python3 train/export_playback.py --data data/dataset.npz --own data/own.npz
```

## Verify what is actually loaded

In the app, **System checks → Run system checks** loads real weights and runs
inference against the exported Python reference. **Test bundled examples**
shows recorded labels beside actual predictions, including mismatches.
**Test a Setu recording file** evaluates new takes without sending them anywhere
or claiming they have trained the model.

In the consultation, the camera proposes signs for a review draft. Remove
incorrect detections and use **Send signed message** to add the intended words
to the shared transcript. Stopping the camera preserves the current draft.
Use **Capture one sign → Finish capture** when automatic boundaries miss a sign.
Low-confidence and partial-hand results remain suggestions, not confirmed messages.

Playback finishes each clip before advancing, preserves intentional repeated
words, and appears in both doctor and patient views. Unknown words stay visible
as text. The skeleton playback is a vocabulary preview, not full ISL grammar.

## Automated regression checks

```sh
cd app
npm test
npm run build
npm run lint
```

The Node tests require Node 22.15+ (native TypeScript and module hooks).
Python checks require numpy; full training/export also requires TensorFlow and
tensorflowjs in the compatible Python environment. Browser verification uses
synthetic camera input and real exported model fixtures; it does not replace
an evaluation with real signers.
