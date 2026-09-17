# Bilingual benchmark workflow

This workflow turns synthetic, explicitly consented, or appropriately licensed
recordings into a governed local benchmark. Audio and per-case results are
ignored by Git.

## 1. Collect or generate source audio

Use the frozen templates in
`src/finvoice_ai/evaluation/data/bilingual_benchmark_taxonomy.json`. Record the
exact provider, voice identifier/revision, generation settings, date, license,
and operating system. A synthetic provider's output is not assumed to be
representative of human speech.

Use pseudonymous speaker IDs and allocate each speaker to exactly one split.
Do not select speakers, wording, or transformations after observing test
results.

The reproducible local synthetic starter set uses eSpeak NG and FFmpeg. It
creates 60 files under ignored `data/`, records exact voice variants and hashes,
and explicitly labels the English-voice limitation:

```bash
brew install espeak-ng ffmpeg
python scripts/generate_synthetic_benchmark.py --output-root data
```

## 2. Apply deterministic transforms

The standard transform resamples to 16 kHz mono PCM16 and normalizes to -3 dBFS.
Noise mixing uses a named local WAV, target SNR, and seed. The command appends a
hash-addressed record to the private reproducibility ledger:

```bash
python -m finvoice_ai.evaluation.audio_transforms \
  data/source/en-pin-short.wav \
  data/audio/en-pin-short-clean.wav \
  --sample-rate 16000 \
  --peak-dbfs -3 \
  --seed 17 \
  --ledger data/transform-ledger.jsonl
```

For a noise condition:

```bash
python -m finvoice_ai.evaluation.audio_transforms \
  data/source/en-pin-short.wav \
  data/audio/en-pin-short-cafe.wav \
  --noise data/noise/cafe.wav \
  --noise-snr-db 10 \
  --seed 17 \
  --ledger data/transform-ledger.jsonl
```

Use only noise recordings with documented provenance and compatible licenses.
The tool emits the source, noise, and output hashes plus all transform settings.

## 3. Create and audit the manifest

Copy the example manifest into ignored `data/`, record every required governance
field, and copy each transform output hash into `audio_sha256`. Then run:

```bash
make audit-speech \
  SPEECH_MANIFEST=data/speech-manifest.json \
  SPEECH_DATASET_ROOT=data
```

The audit verifies file presence, path safety, SHA-256, the bounded WAV
contract, development/test speaker separation, and transcript/audio leakage.
Warnings for repeated text within one split must be explained in the dataset
card. Errors block evaluation.

## 4. Select on development only

Compare model sizes or decoding configurations only on `development`. Record
the exact model identifiers, revisions, beam sizes, language settings, seeds,
hardware, package versions, and commands. Keep generated per-case reports in
ignored `reports/`.

```bash
FINVOICE_TRANSCRIPTION_PROVIDER=faster_whisper \
FINVOICE_WHISPER_MODEL_SIZE=small \
SPEECH_MANIFEST=data/speech-manifest.json \
SPEECH_DATASET_ROOT=data \
SPEECH_EVALUATION_ARGS="--split development --output reports/development-small.json" \
make evaluate-speech
```

Freeze one configuration and document the rationale before running `test` once.
Never revise the frozen choice based on held-out results.

The published v1 baseline compared `tiny` and `base` with CPU int8 and beam size
1, froze `base` from development evidence, and then ran the held-out test split
once. See `docs/bilingual-asr-baseline.md` for the results and limitations.

## 5. Publish safe aggregates

Commit only aggregate tables that cannot reconstruct restricted utterances.
Report micro and macro WER/CER, failure rate, language-identification accuracy,
p50/p95 latency, real-time factor, slices, paired differences, confidence
reliability, and bootstrap intervals when sample size supports them. Label
small-sample intervals and synthetic-only findings as exploratory.
