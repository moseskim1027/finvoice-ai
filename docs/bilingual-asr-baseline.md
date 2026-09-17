# Bilingual synthetic ASR baseline

## Result status

This is an exploratory systems baseline on 60 locally generated synthetic
utterances. It is evidence that the governed workflow runs end to end; it is
not evidence of performance on Filipino speakers, natural code-switching, or
production audio.

The dataset contains 45 development and 15 held-out test cases. Four eSpeak NG
English voice variants each read 15 safe intent templates. Speaker/voice IDs and
normalized transcripts are disjoint across development and test. Conditions
cover clean and synthetic household/street/cafe-like noise plus high-quality
and 8 kHz round-trip phone simulation. The integrity audit verified all 60 WAV
files and hashes with zero errors or warnings.

## Development selection

Both configurations used Faster Whisper 1.2.1 on CPU with int8 compute, beam
size 1, automatic language identification, identical VAD, and the same frozen
development cases.

| Configuration | Micro WER | Macro WER | Micro CER | LID accuracy | p50 ms | p95 ms | Mean RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `faster-whisper/tiny` | 1.012 | 1.021 | 0.532 | 0.644 | 850.9 | 5891.7 | 0.509 |
| `faster-whisper/base` | 0.794 | 0.797 | 0.415 | 0.667 | 710.2 | 4851.0 | 0.489 |

The paired mean per-utterance WER delta for base minus tiny was -0.224. A 2,000
sample paired bootstrap with seed 20260918 produced a 95% interval of
[-0.377, -0.079]. Base also improved CER and measured latency, so it was frozen
before test evaluation. This interval describes only this small synthetic
sample and does not establish population-level superiority.

## Held-out test result

The frozen configuration was `faster-whisper/base`, CPU, int8, beam size 1,
automatic language identification. It was run once on the 15 held-out cases.

| Cases | Failure rate | Micro WER | Macro WER | Micro CER | Macro CER | LID accuracy | p50 ms | p95 ms | Mean RTF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.000 | 0.846 | 0.818 | 0.458 | 0.432 | 0.667 | 826.3 | 8239.8 | 0.596 |

Test WER by declared language was 0.364 for English, 0.826 for code-switched
English–Filipino, and 1.227 for Filipino. Filipino LID accuracy was 0.0. These
results primarily expose the severe mismatch created by using English-only
synthetic voices for Filipino text; they must not be interpreted as a property
of Filipino speech or speakers.

Confidence reliability was poor: expected calibration error was 0.379 on
development and 0.382 on test for the selected configuration. Faster Whisper
confidence is treated as an uncalibrated diagnostic measurement and must not
control automation or escalation from these results.

## Reproducibility record

- Dataset: `finvoice-local-synthetic-bilingual-v1` version 1.0.0
- Taxonomy/transform: 1.0.0 / 1.0.0
- Manifest SHA-256: `db715da471a0a8dccce6da7589d14ad42366751bb0bd855334f44050cda7110a`
- Generation seed: 20260918
- Runtime: Python 3.11.0, Faster Whisper 1.2.1, CTranslate2 4.8.2, NumPy 2.4.6
- Hardware/OS: Apple Silicon arm64, macOS 27.0, CPU inference
- Synthetic source: eSpeak NG 1.52.0 voice variants
- Conversion: FFmpeg 9.0.1, mono 16 kHz PCM16

Exact generation, audit, and evaluation commands are documented in the
benchmark workflow. Machine-readable safe aggregates are versioned alongside
the evaluation package. Raw audio, manifests, transform ledgers, reference and
hypothesis pairs, and per-utterance reports remain in ignored local storage.

## Limitations and threats to validity

- Every voice is synthetic and English-based; no human Filipino voice is
  represented.
- eSpeak pronunciation of Filipino and mixed-language text is not natural.
- Four voice variants are not four independent human speakers.
- Procedural noise is only a controlled perturbation, not realistic household,
  street, or cafe acoustics.
- Phone simulation is an 8 kHz resampling proxy and omits codecs, packet loss,
  microphones, networks, and room impulse responses.
- Fifteen test cases cannot support demographic, fairness, or broad robustness
  claims.
- Noise, device, language, voice, and wording effects are not fully separable
  at this sample size.
- The published error taxonomy includes categories that are not estimable
  without richer annotations or paired recordings.

The next dataset revision should add explicitly consented or appropriately
licensed human speech from multiple Filipino regions and devices, realistic
licensed noise and room transforms, more speakers per split, annotated
code-switch boundaries, and amount/entity test cases that remain entirely
synthetic.
