# Dataset card: [dataset name]

Status: **draft — do not evaluate until all required fields are complete**

## Identity and version

- Dataset name:
- Version:
- Taxonomy version: `1.0.0`
- Transform version: `1.0.0`
- Manifest SHA-256:
- Frozen date:
- Responsible maintainer:

## Intended use

This dataset is intended only for evaluating English, Filipino, and
English–Filipino code-switched automatic speech recognition and downstream
support-intent research in the FinVoice portfolio environment. It may expose
engineering failure modes but cannot establish population-level performance.

## Prohibited use

- identifying, authenticating, profiling, or ranking people;
- inferring emotion, honesty, trustworthiness, health, or protected traits;
- making financial eligibility, fraud, employment, or access decisions;
- training or evaluating speaker-recognition systems;
- production deployment without a separate representative-data review;
- reconstructing identities or linking pseudonyms to real people.

## Provenance and consent

For every source or synthetic voice, record:

- pseudonymous speaker or voice identifier;
- synthetic provider and voice revision, explicit consent record, or license;
- collection/generation date and method;
- permitted uses and withdrawal procedure where applicable;
- source and transformed audio hashes.

Do not include names, account details, real transactions, addresses, phone
numbers, or any other personal financial information in recordings or text.

## Composition

Complete this table from the frozen manifest.

| Dimension | Values | Development count | Test count |
|---|---|---:|---:|
| Language | English / Filipino / English–Filipino | | |
| Noise | clean / household / street / cafe | | |
| Device | high-quality / phone-simulated | | |
| Utterance type | short command / conversational request | | |
| Intent | five taxonomy intents | | |
| Speaker/voice | pseudonymous IDs | | |

## Collection and transforms

- Original format:
- Resampling command:
- Normalization target:
- Noise sources and licenses:
- Noise-mixing SNR values and seeds:
- Phone simulation method:
- Software and dependency versions:
- Hardware and operating system:

Store the JSONL transform ledger with the private dataset. Verify every ledger
output hash against the evaluation manifest before running ASR.

## Splits and leakage controls

Development and test speakers must be disjoint. Model sizes, decoding settings,
VAD parameters, prompts, transforms, thresholds, and calibration methods are
selected using development data only. The test split is evaluated once after
the manifest and experiment configuration are frozen.

Record the integrity-audit command and output:

```bash
make audit-speech \
  SPEECH_MANIFEST=data/speech-manifest.json \
  SPEECH_DATASET_ROOT=data
```

## Known limitations and missing representation

Document at minimum:

- synthetic-versus-human speech coverage;
- regional language and accent coverage;
- age, gender, disability, and device gaps, without guessing attributes;
- vocabulary and financial-intent limitations;
- noise realism and room-acoustics limitations;
- dataset size and uncertainty;
- ASR and language-identification failure modes.

## Retention and access

- Storage location and access controls:
- Retention period:
- Deletion and consent-withdrawal process:
- Backup policy:
- Raw-audio publication status:

Raw audio and per-utterance reports remain outside Git. Publish only schemas,
safe templates, aggregate metrics, and sufficiently anonymized error counts.

## Evaluation record

- Development configurations compared:
- Frozen configuration and selection rationale:
- Test execution date:
- Exact commands:
- Aggregate report paths and hashes:
- Reviewer:
