# Multimodal intent baseline study

## Abstract

This study compares majority, transcript-only, acoustic-only, and two fused
intent classifiers on the governed FinVoice synthetic speech benchmark. The
held-out result does **not** demonstrate a multimodal improvement. Transcript
TF-IDF reached 1.00 macro F1 on 15 test utterances. Validation selected a late
fusion weight of 1.00 on text, so late fusion was identical to text. Feature
concatenation reached 0.808 macro F1 and 0.80 accuracy; its paired accuracy
difference from text was -0.20 with a bootstrap 95% interval of [-0.40, 0.00].

## Methods

The local dataset contains 60 synthetic eSpeak utterances: 30 from `voice-a`
and `voice-b` for training, 15 from `voice-c` for validation, and 15 from the
held-out `voice-d` for testing. Splits are grouped by voice. The five labels are
PIN reset, statement access, card security, money transfer, and unrelated
request. Escalation is derived from explicit policy labels, never from inferred
emotion or speaker traits.

The compared models are:

- a majority-class baseline;
- character TF-IDF plus balanced logistic regression on ASR transcripts;
- eight waveform summary features plus scaled balanced logistic regression;
- validation-selected late fusion of text and acoustic probabilities; and
- concatenated TF-IDF/acoustic features with logistic regression.

Regularization, fusion weight, temperature, and abstention threshold are chosen
using validation data only. The abstention selector maximizes coverage among
thresholds achieving at least 0.80 validation accuracy. The test split is used
once after selection. The code also provides a lazy frozen WavLM adapter and an
audio-hash/model-revision cache, but this run does not report WavLM results.

Reproduce the ignored detailed report from the governed local dataset with:

```bash
make research-baselines SPEECH_MANIFEST=data/speech-manifest.json SPEECH_DATASET_ROOT=data
```

The publishable aggregate result is stored in
`src/finvoice_ai/research/results/multimodal-intent-baseline-v1.json`.

## Held-out results

| Model | Accuracy | Macro F1 | Brier | ECE | Escalation P/R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.20 | 0.067 | 1.600 | 0.800 | 0.60 / 1.00 |
| Acoustic | 0.40 | 0.390 | 0.712 | 0.161 | 0.64 / 1.00 |
| Text | 1.00 | 1.000 | 0.681 | 0.737 | 1.00 / 1.00 |
| Late fusion | 1.00 | 1.000 | 0.681 | 0.737 | 1.00 / 1.00 |
| Concatenated fusion | 0.80 | 0.808 | 0.343 | 0.186 | 0.89 / 0.89 |

Perfect text accuracy should not be confused with calibrated confidence: its
ECE remained 0.737 on this tiny test set. The validation-selected text and late
fusion thresholds therefore stayed at 0.0, retaining full coverage. The
concatenated model selected 0.6 and retained 0.667 test coverage at 0.90
selective accuracy. The acoustic model also selected 0.6 but accepted no test
cases, showing that a threshold can fail to transfer across speakers.

The report includes risk-coverage curves, confusion matrices, per-class
precision/recall/F1, and slices by language mode, noise, device, and speaker.
Those slices are descriptive only: many contain very few cases.

## Ablations and robustness

On validation, ASR and reference transcript variants both reached 1.00 macro
F1. This does not establish ASR robustness; the synthetic utterance templates
are highly separable. The acoustic baseline reached 0.274 macro F1 with all
features, versus 0.190 without timing, 0.174 without energy, and 0.171 without
frame statistics. These point estimates suggest that all three feature groups
carry signal, but the sample is too small for causal or population claims.

## Interpretation

The result is negative/inconclusive for multimodal benefit. Late fusion learned
to ignore acoustics, while concatenation degraded accuracy. Its uncertainty
interval includes a tie but no positive test effect. A credible improvement
claim would require consistent validation and test gains, a meaningful effect,
and an interval supporting that effect; none is present here.

## Limitations and threats to validity

- The audio uses four synthetic English eSpeak variants and does not represent
  Filipino speakers, accents, natural code-switching, or real environments.
- There is one held-out test voice and three examples per intent, so estimates,
  calibration bins, slices, and bootstrap intervals are unstable.
- Repeated intent templates can make transcript classification artificially
  easy and limit construct validity.
- Summary waveform features are engineering measurements, not emotion,
  personality, intent, or trustworthiness measurements.
- The implemented frozen WavLM path was not evaluated; no learned acoustic
  representation claim follows from this study.
- Runtime and memory measurements depend on the local ARM64 macOS environment
  and are not production serving benchmarks.
- Model and threshold selection used validation only, but repeated development
  against this small benchmark can still create researcher overfitting.

The next study should use consented or appropriately licensed multilingual
speech, more speakers per split, naturally occurring variation, preregistered
selection rules, and an external test set before considering deployment.
