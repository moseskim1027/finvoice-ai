# Production-readiness evidence

## Scope and architecture

This evidence demonstrates operability of a local research prototype. It is not
evidence of bank integration or production readiness for real customer data.

```text
HTTP / WebSocket / MCP
          |
 correlation context
 request / session / utterance / audit
          |
   +------+------+ 
   |      |      |
 JSON   metrics  OpenTelemetry
 logs    /metrics spans
   |      |      |
   +------+------+ 
          |
 SLO, fault, load, cost, and quality evidence
```

Logs use an allowlist and never accept bodies, unrestricted transcripts, raw
audio, secrets, or account identifiers. Metrics use bounded labels. Sanitized
spans cover HTTP, policy, retrieval, generation, grounding, persistence, VAD,
ASR, streaming, and MCP authorization/execution. A representative trace is
stored in `src/finvoice_ai/evaluation/data/results/sanitized_trace_example.json`.

## Measured local load result

Command: `make production-evidence`. Hardware: ARM64 macOS, 8 logical CPUs,
Python 3.11. Workload: 400 requests, concurrency 8, 75% grounded deterministic
responses and 25% policy escalations, with no network or model service.

| Scenario | p50 ms | p95 ms | p99 ms | req/s | Failures | CPU s | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared retriever | 0.049 | 0.062 | 0.297 | 19,491 | 0 | 0.022 | 33.1 |
| Rebuild per request | 2.121 | 3.740 | 4.540 | 3,458 | 0 | 0.211 | 33.5 |

Sharing the immutable retriever improved observed throughput by 5.64x and
reduced p95 by 3.68 ms in this microbenchmark. The illustrative cost model uses
$0.05 per CPU-hour and estimates $0.000000000757 per warm conversation. It
excludes network, storage, telemetry backend, GPU, and external-model charges;
the number is useful only for validating accounting mechanics.

## Fault evidence

| Injected fault | Expected and observed behavior | Traceability |
| --- | --- | --- |
| ASR timeout | HTTP 503; no fabricated transcript | request ID and `asr.transcribe` error |
| Retriever unavailable | safe human escalation | `provider_unavailable` and error span |
| Generator unavailable | safe human escalation | `provider_unavailable` and error span |
| MCP unauthenticated call | fail-closed denial | `denied_or_failed` audit outcome |
| Malformed WAV | HTTP 422 | request ID and bounded error metric |
| Downstream provider unavailable | safe escalation | operation error and escalation counter |

Existing streaming tests additionally verify buffer/session limits, ordering,
disconnect cleanup, cancellation, and stale-completion suppression.

## Quality, latency, and cost

| Candidate | Quality evidence | Latency evidence | Cost implication |
| --- | --- | --- | --- |
| Deterministic conversation path | retrieval hit rate/MRR 1.0 on six starter cases | p95 0.062 ms in local microbenchmark | lowest measured CPU cost; no external model |
| Faster Whisper base | frozen test WER 0.346, CER 0.179 | p95 8.240 s on synthetic benchmark | materially higher local CPU time |
| Intent text baseline | macro F1 1.0 on 15 synthetic held-out cases; ECE 0.737 | test inference about 2.9 ms | small local artifact; calibration remains weak |

These rows are not directly substitutable products and do not establish a
universal Pareto frontier. They make the tradeoffs explicit and prevent a
quality-only or latency-only claim.

## Deployment controls and gaps

The container runs as a non-root system user, uses `no-new-privileges`, has a
readiness healthcheck, PID 1 init, and a graceful-stop interval. Python package
ranges and the base image family are constrained, while deployed environments
should additionally lock resolved dependencies and pin the base image digest.
Configuration uses `FINVOICE_` environment variables; secrets must come from a
secret manager or mounted secret and must never be committed or logged.

Known gaps include no external telemetry backend/dashboard, no distributed
collector, no durable audit store, no multi-process metric aggregation, no
network load generator, no GPU measurement, and no canary/rollback controller.
The local in-memory span exporter is bounded and exists for evidence and tests,
not production retention.
