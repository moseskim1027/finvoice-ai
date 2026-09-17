# FinVoice AI

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
retrieval, speech services, MCP tools, and telemetry remain future milestones.

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
|   `-- infrastructure/  # Local provider implementations
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

Container workflow:

```bash
docker compose up --build
```

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
