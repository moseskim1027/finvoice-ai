# FinVoice AI

[![CI](https://github.com/moseskim1027/finvoice-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/moseskim1027/finvoice-ai/actions/workflows/ci.yml)

FinVoice AI is a research-to-production portfolio project for a bilingual
financial-support voice and text agent. It is designed to demonstrate speech
AI, grounded LLM applications, safe tool use, human escalation, evaluation,
observability, and inference-cost engineering in one coherent system.

The project is intentionally more than a chatbot demo. Its goal is to produce
credible evidence for both production ML engineering and applied research:

- a deployable, observable support service with explicit safety boundaries;
- a reproducible speech and language evaluation suite;
- a publication-style study of robustness, calibration, and escalation;
- measurable quality, latency, reliability, and cost tradeoffs.

## Research question

> Does combining acoustic embeddings with transcript features improve support
> intent classification and escalation decisions under noisy, accented, and
> code-switched speech, and can calibrated abstention reduce harmful errors?

This question keeps the research contribution tied to a real operational
decision. A model should not merely classify a conversation; it should know
when confidence is too low and transfer the case safely.

The first reproducible baseline study found no multimodal improvement on its
small synthetic held-out set: late fusion selected text only, while feature
concatenation underperformed the transcript model. See the
[multimodal intent study](docs/multimodal-intent-study.md) for methods, results,
calibration findings, and limitations.

## System architecture

```text
 Voice caller                         Text user
      |                                   |
      v                                   v
 [VAD + streaming ASR]              [Chat gateway]
      |                                   |
      +----------------+------------------+
                       v
              [Conversation service]
              - session state
              - policy checks
              - PII redaction
              - confidence/abstention
                       |
          +------------+-------------+
          |                          |
          v                          v
 [RAG over approved docs]      [MCP tool gateway]
 - hybrid retrieval           - account simulator
 - citations                  - ticket creation
 - versioned policies         - authenticated actions
          |                          |
          +------------+-------------+
                       v
                 [LLM router]
          small model <-> large model
                       |
          +------------+-------------+
          |                          |
          v                          v
 [Text response / TTS]       [Human escalation]
                              + summary/reason

 Every component -> OpenTelemetry -> metrics, traces, logs
 Evaluation suite -> CI quality gate -> canary -> production
```

The current implementation provides a transport-independent conversation
service, typed ports for retrieval, generation, safety policy, and conversation
storage, a repository-controlled Markdown knowledge base, deterministic BM25
retrieval, auditable response metadata, citation validation, and safe escalation
for missing context or unavailable providers. Real model providers, vector
retrieval and telemetry remain future milestones. The initial speech foundation
validates PCM WAV uploads, detects voice activity, exposes a replaceable ASR
provider contract, and reports WER/CER metrics. A deterministic streaming layer
adds ordered PCM chunk ingestion, endpointing, partial/final transcript events,
bounded session state, and response interruption. A standalone MCP server exposes
synthetic demo tools through the official stable
[MCP Python SDK](https://py.sdk.modelcontextprotocol.io/).

## Current orchestration boundary

```text
[FastAPI route]
      |
      v
[Conversation service]
      |
      +----> [Safety policy]
      |
      +----> [Retriever port]
      |             |
      |             v
      |       approved context
      |
      +----> [Generator port]
      |             |
      |             v
      |       grounded answer
      |
      +----> [Citation metadata]
      |
      +----> [Conversation store]
      |
      v
[Typed API response]
 request ID / decision / reason / confidence
 citations / provider metadata
```

All four provider boundaries are structural Python protocols. Deterministic
local implementations allow the full flow to run in tests without network
access or an external model. Provider implementations may signal a bounded
`ProviderUnavailableError`; the service converts it to a safe human escalation
instead of leaking an exception through the API.

## Grounded retrieval

The current baseline keeps approved content as version-controlled Markdown and
ranks it with BM25. It is deliberately small and inspectable so later vector or
hybrid implementations can be evaluated against identical contracts and cases.

```text
[Approved Markdown]
        |
        v
[Document loader]
 title / body / stable ID / source path
        |
        v
[BM25 index]
 tokenization / term frequency / length normalization
        |
        v
[Ranked context]
 score threshold / top-k
        |
        v
[Response generator]
 answer + cited document IDs
        |
        v
[Citation validator]
 cited IDs must be present in retrieved context
        |
   +----+----+
   |         |
 valid     invalid
   |         |
   v         v
[response] [safe escalation]
```

Run the offline benchmark with:

```bash
make evaluate-retrieval
```

The versioned six-case starter dataset currently produces hit rate `1.0` and
mean reciprocal rank `1.0`. These numbers only verify the tiny baseline corpus;
they are not evidence of production retrieval quality. The dataset must grow
with paraphrases, ambiguous questions, hard negatives, multilingual queries,
and policy-version conflicts before comparing production candidates.

## MCP tool gateway

The MCP server demonstrates protocol integration without granting an LLM direct
access to repositories, credentials, or real financial systems.

```text
[MCP host / model]
        |
        v
[Official MCP server]
 typed schemas / stdio transport
        |
        v
[Tool gateway]
 fixed registry / unknown-tool rejection
        |
        v
[Authorization policy]
 authenticated principal
 required scopes
 explicit confirmation for side effects
        |
   +----+----------------+
   |                     |
 denied              authorized
   |                     |
   v                     v
[audit event]      [synthetic handler]
                         |
                         v
                    [audit event]
```

Two tools are exposed:

- `get_demo_account_status`: read-only, requires `accounts:read`, and accepts
  identifiers matching `DEMO-[0-9]{3}` only;
- `create_demo_support_ticket`: synthetic side effect, requires
  `tickets:write` and explicit confirmation.

Run the stdio MCP server with:

```bash
make run-mcp
```

Security properties in the current implementation:

- callers cannot self-grant scopes or confirmation through tool arguments;
- tool names come from a fixed registry;
- only synthetic demo identifiers and records are accepted;
- audit events record argument names, never argument values;
- successful results include an audit ID;
- denied and failed calls are also recorded.

The current wrapper uses a fixed synthetic demo principal because no real
identity provider exists in this portfolio environment. This is not production
authentication. A deployed version must derive identity and scopes from a
verified token, bind confirmation to the initiating user and exact action, use
durable append-only audit storage, and apply rate limits and replay protection.

## Speech foundation

The first speech slice establishes deterministic input and evaluation contracts
before a production ASR model is selected.

```text
[WAV upload]
      |
      v
[Input validation]
 mono / signed 16-bit PCM / supported rate
 duration limit / upload-size limit
      |
      v
[Energy VAD]
 20 ms frames / RMS threshold
 minimum speech / silence bridging
      |
      v
[Speech segments]
 start / end / mean RMS
      |
      v
[Transcription port]
      |
      +---- current: deterministic stub
      `---- optional: Faster Whisper adapter
                    lazy model loading / CPU or CUDA
      |
      v
[Transcript metadata]
 text / confidence / language / model
      |
      v
[Offline evaluation]
 WER / CER / robustness slices
```

Analyze a local WAV file after starting the API:

```bash
curl -s http://localhost:8000/v1/audio/analyze \
  -F 'file=@sample.wav;type=audio/wav'
```

Supported input is mono, uncompressed signed 16-bit PCM WAV at 8, 16, 24, or
48 kHz, up to 60 seconds and 6.5 MB. Unsupported or malformed audio is rejected
before provider execution.

The default `deterministic-asr-stub-v1` response is intentionally not real
speech recognition and must not be presented as one. Its purpose is to verify
the provider boundary, API schema, abstention on silence, and evaluation code.

To run the Faster Whisper adapter in Docker:

```bash
docker compose --profile asr up --build asr
```

The adapter extracts VAD-selected speech, converts PCM16 to normalized samples,
and resamples it to 16 kHz before inference. Faster Whisper accepts float32
NumPy audio and downloads named models from the Hugging Face Hub on first use;
pin or pre-stage model artifacts before a reproducible or offline deployment.
The ASR extra is based on the official
[Faster Whisper package](https://pypi.org/project/faster-whisper/) and its
[model API](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py).

`confidence` is currently a duration-weighted transformation of Whisper segment
average log probabilities. It is a useful diagnostic score, not a calibrated
probability. Before it controls automation or escalation, fit and validate a
calibrator on held-out, representative English, Filipino, and code-switched
speech.

The example evaluation manifest at
`src/finvoice_ai/evaluation/data/speech_manifest.example.json` makes language
mode, noise, device, pseudonymous speaker, consent basis, license, and split
explicit. Its audio paths are placeholders—not bundled recordings. Replace
them only with synthetic, consented, or appropriately licensed WAV files, keep
test speakers separate from development speakers, and never tune on the test
split.

```text
[Governed manifest + WAV files]
              |
              v
       [Offline ASR run]
              |
       +------+-------+
       |              |
       v              v
   [WER / CER]   [Latency / failures]
       |              |
       +------+-------+
              v
 [Slices: language / code-switch / noise / device]
              |
              v
[Error taxonomy + confidence calibration report]
```

### Speech evaluation runner

The manifest-driven runner measures the whole bounded speech path—WAV parsing,
voice activity detection, preprocessing, and transcription. It emits JSON with
per-case results, micro-averaged WER and CER, failure rate, p50/p95 latency,
mean real-time factor, and slices by language, language mode, noise condition,
and device.

```text
[Manifest] + [WAV dataset]
          |
          v
 [Select development or test split]
          |
          v
 [WAV -> VAD -> ASR] ----failure----+
          |                          |
          v                          v
 [Hypothesis + timing]       [Typed failure record]
          |                          |
          +------------+-------------+
                       v
       [Per-case + aggregate JSON report]
          WER / CER / failure rate
          p50 + p95 latency / real-time factor
          language / mode / noise / device slices
```

The example manifest contains metadata placeholders but no recordings. Copy it
into a private or ignored `data/` directory, add only synthetic, consented, or
appropriately licensed WAV files, and update each relative `audio_path`.

Run the development split locally with Faster Whisper:

```bash
python -m pip install -e '.[dev,asr]'
mkdir -p reports
export FINVOICE_TRANSCRIPTION_PROVIDER=faster_whisper
SPEECH_MANIFEST=data/speech-manifest.json \
SPEECH_DATASET_ROOT=data \
SPEECH_EVALUATION_ARGS="--split development --output reports/development.json" \
make evaluate-speech
```

Or run the same evaluation in the architecture-neutral ASR container:

```bash
mkdir -p reports
docker compose --profile asr run --rm \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/reports:/reports" \
  evaluate-speech-asr \
  /data/speech-manifest.json \
  --dataset-root /data \
  --split development \
  --output /reports/development.json
```

Use `--split test` only after model, decoding, VAD, and calibration choices are
frozen. A failed or missing recording remains visible as a typed failed case,
contributes an empty-hypothesis error rate, and makes the command exit nonzero.
The runner reports descriptive measurements; it does not claim statistical
significance or calibrated confidence.

The governed 60-case local synthetic benchmark workflow, safe aggregate
Faster Whisper results, and strong limitations are documented in the
[bilingual ASR baseline](docs/bilingual-asr-baseline.md). The committed results
demonstrate reproducibility and expose an English-synthetic-voice mismatch;
they are not claims about Filipino speakers or real support calls.

## Streaming conversation mechanics

The WebSocket endpoint at `/v1/audio/stream/{session_id}` demonstrates bounded,
transport-thin streaming behavior without a microphone or external service.
It accepts ordered mono signed 16-bit little-endian PCM chunks and emits an
event for every state change:

```text
[base64 PCM chunks]
        |
        v
[session + sequence checks] -- duplicate --> [idempotent accepted]
        |                     out of order --> [error]
        v
[bounded audio buffer]
        |
        v
[energy endpointing]
 speech_started / trailing silence / maximum duration
        |
        +-------------------+
        |                   |
        v                   v
[simulated partial]    [final transcript]
 offline ASR rerun       response generation
                              |
                       new speech / barge-in
                              |
                              v
                       [cancel response once]
```

Send each chunk as JSON. `pcm_s16le_base64` contains raw PCM samples, not a WAV
container:

```json
{
  "type": "audio_chunk",
  "utterance_id": "utterance-1",
  "sequence": 0,
  "sample_rate_hz": 16000,
  "pcm_s16le_base64": "AAAAAAAAAAAAAAAAAAAAAA=="
}
```

Events use stable `type` discriminators: `accepted`, `speech_started`,
`partial`, `finalized`, `interrupted`, and `error`. Sequence numbers begin at
zero for each utterance. Repeated sequence numbers are acknowledged as
duplicates without appending audio; missing or out-of-order numbers are
rejected. A new utterance also begins at sequence zero.

The default endpoint detector requires 60 ms of speech, finalizes after 300 ms
of trailing silence, and forcibly finalizes at 30 seconds. Buffers are capped at
35 seconds, inactive sessions expire after 60 seconds, and the process admits
at most 100 sessions. Disconnecting removes session state immediately. These
defaults live in `StreamingConfig` and tests use a fake clock.

The current partial implementation periodically invokes the existing offline
transcription provider over accumulated audio. It is simulated partial
transcription, not genuine token streaming. Its purpose is to prove ordering,
state, endpoint, cancellation, and transport semantics before choosing a live
streaming ASR provider. Timing hooks expose time to speech start, time to first
partial, endpoint delay, final transcript latency, and interruption latency;
the production-observability milestone will connect them to a metrics backend.

Run the deterministic WebSocket demonstration through its chunked PCM fixture:

```bash
.venv/bin/pytest -q -o addopts='' tests/test_streaming_api.py
```

The core test suite separately verifies initial silence, short pauses, forced
finalization, buffer/session limits, disconnect cleanup, barge-in idempotency,
stale completion rejection, and isolation between concurrent sessions.

## Research pipeline

```text
 [Audio + transcript + labels]
             |
      [Data statement]
 consent / language / noise / subgroup coverage
             |
     +-------+---------+
     |                 |
     v                 v
 [SSL speech model] [Text transformer]
 WavLM/HuBERT       transcript encoder
     |                 |
     +-------+---------+
             v
      [Fusion + prediction]
             |
     +-------+---------+
     |                 |
     v                 v
 [Calibration]   [Abstain/escalate]
     |                 |
     +-------+---------+
             v
 [Robustness and fairness evaluation]
 accent / noise / device / subgroup / shift
             |
             v
 [Ablations + confidence intervals + report]
```

## Evaluation strategy

The project will report system and research metrics together.

### Speech and research metrics

- word error rate and intent accuracy;
- expected calibration error and Brier score;
- escalation precision, recall, and risk-coverage curves;
- performance across language, code-switching, accent, noise, and device slices;
- ablations for acoustic-only, transcript-only, and fused representations;
- bootstrap confidence intervals and a documented error taxonomy.

### Product and service metrics

- task-completion and grounded-answer rates;
- containment, human-transfer, repeat-contact, and safe-abstention rates;
- p50 and p95 end-to-end latency;
- availability, timeout, and tool-failure rates;
- cost per conversation and cost per successfully resolved case.

### Safety and responsible AI

- use synthetic, consented, or appropriately licensed data only;
- minimize and redact personally identifiable information;
- treat acoustic behavior signals as uncertain features, not psychological fact;
- include subgroup, calibration, and distribution-shift evaluation;
- use explicit policies and human review for sensitive financial actions;
- preserve tool-call provenance and auditable escalation reasons.

## Portfolio milestones

### Milestone 1 - Service foundation

- FastAPI application with typed request and response schemas;
- deterministic risk-policy and confidence-based escalation;
- configuration, structured package layout, health endpoint, and tests;
- container and local developer workflow.

### Milestone 2 - Text support agent

- approved-document retrieval with citations;
- policy-constrained LLM orchestration;
- MCP tools for read-only account simulation and ticket creation;
- prompt-injection tests and human-handoff summaries.

### Milestone 3 - Speech pipeline

- streaming or simulated-streaming ASR and TTS;
- voice activity detection, endpointing, and interruption handling;
- English/Filipino and code-switched evaluation data;
- self-supervised speech embeddings and calibrated confidence.

### Milestone 4 - Research study

- transcript-only, acoustic-only, and multimodal baselines;
- robustness and subgroup evaluation;
- ablations, confidence intervals, limitations, and reproducibility artifacts;
- publication-style technical report.

### Milestone 5 - Production evidence

- OpenTelemetry traces, structured logs, and operational dashboards;
- SLOs, alerts, fault injection, canary release, and incident runbook;
- model routing, caching, prompt compression, and cost benchmarking;
- quality-cost-latency Pareto analysis.

## Repository layout

```text
finvoice-ai/
|-- src/finvoice_ai/
|   |-- api/             # HTTP transport
|   |-- application/     # Orchestration service and provider ports
|   |-- domain/          # Request models and deterministic policy
|   |-- evaluation/      # Retrieval benchmark and versioned cases
|   |-- infrastructure/  # Local provider implementations
|   |-- speech/          # WAV, VAD, ASR contracts, and response schemas
|   |-- streaming/       # Chunk events, endpointing, sessions, and interruption
|   |-- tools/           # Tool policy, gateway, handlers, and audit models
|   `-- mcp_server.py    # Official SDK protocol adapter
|-- tests/
|   `-- scenarios/       # Versioned conversation behavior cases
|-- .env.example         # Safe local configuration template
|-- Dockerfile           # Reproducible API container
|-- Makefile             # Common development commands
|-- compose.yaml         # Local service orchestration
`-- pyproject.toml       # Python project, tooling, and dependencies
```

## Quick start

Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
make test
make run
```

The API will be available at `http://localhost:8000` with interactive OpenAPI
documentation at `http://localhost:8000/docs`.

Example request:

```bash
curl -s http://localhost:8000/v1/conversations/respond \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo-1","message":"How do I reset my PIN?","confidence":0.91}'
```

The response includes the decision evidence needed for evaluation and tracing:

```json
{
  "request_id": "2ee8cbaa-969d-4f1a-8109-72676c9bec1f",
  "session_id": "demo-1",
  "decision": "respond",
  "message": "Open Settings, choose Security, and select Reset PIN.",
  "reason": null,
  "confidence": 1.0,
  "citations": [
    {
      "document_id": "pin-reset",
      "title": "Resetting your PIN",
      "source": "knowledge/pin-reset.md",
      "score": 2.13,
      "excerpt": "Open Settings, choose Security, and select Reset PIN..."
    }
  ],
  "provider": {
    "model": "local-template-v1",
    "retrieved_documents": 1
  }
}
```

Run the versioned behavior suite with `make test`. Its cases cover grounded
responses, low-confidence abstention, sensitive actions, missing approved
context, malformed input, and provider failure.

### Docker alternative

Docker Engine or Docker Desktop with Compose provides an architecture-neutral
alternative to the local Python workflow. The official Python base image is
multi-platform, so these commands are the same on ARM64 and x86-64.

Build and start the API:

```bash
cp .env.example .env
docker compose up --build api
```

Run the complete validation workflow without installing Python dependencies on
the host:

```bash
docker compose run --rm lint
docker compose run --rm test
docker compose run --rm evaluate-retrieval
```

The equivalent MCP command is `docker compose run --rm mcp`. To run the
optional offline ASR image, use `docker compose --profile asr up --build asr`.

The test image installs development dependencies and contains the test suite;
the API image contains runtime dependencies only. Faster Whisper uses a
separate opt-in image because its native inference stack and model cache are
substantially larger. The named `whisper-cache` volume avoids downloading model
weights on every ASR container start.

## Current scope and limitations

This repository currently uses deterministic retrieval and generation so the
service contract and safety decisions can be tested before external models are
added.
It does not yet connect to a bank, process real customer data, perform financial
transactions, or infer emotions. It is a portfolio and research environment,
not a production financial service.

## Development principles

- Prefer explicit contracts and swappable provider interfaces.
- Keep safety policy deterministic and independently testable.
- Evaluate every model change against a versioned scenario set.
- Optimize for quality, latency, reliability, and cost together.
- Document data provenance, assumptions, limitations, and failure modes.
- Use conventional commits and focused pull requests.

## License

No license has been selected yet. All rights are reserved until a license is
added explicitly.
