# Portfolio service objectives

These objectives apply only to the local deterministic portfolio workload. They
are not commitments for a bank, real customers, external model providers, or a
production deployment. Service reliability and model quality are measured
separately so a fast incorrect answer cannot satisfy the quality gate.

## Reliability objectives

Measured over a rolling 30-day window for valid requests to the local service:

| Signal | Objective | Error budget |
| --- | --- | --- |
| Conversation availability | 99.5% non-5xx | 216 minutes / 30 days |
| Conversation latency | 95% below 250 ms | 5% of valid requests |
| Readiness | 99.9% successful probes | 43.2 minutes / 30 days |
| Streaming session admission | 99.0% below configured capacity | 1% of attempts |

Client validation errors, deliberate policy escalation, and authorization
denials are not service failures. Unexpected exceptions, provider timeouts
without a safe response, and corrupt or missing persisted outcomes are failures.
Alert evaluation requires a minimum traffic volume to avoid unstable ratios.

## Model-quality acceptance thresholds

The governed test sets remain release gates rather than uptime SLOs:

- retrieval hit rate and MRR must not regress from the versioned baseline;
- ASR WER/CER and utterance failure rate must stay within the frozen benchmark
  tolerance before changing a transcription model;
- escalation recall for explicitly sensitive intent labels must be 1.00 in the
  starter intent test set;
- any calibration or abstention threshold must be selected on development data,
  never the held-out test split.

## Telemetry and budget policy

`/metrics` exposes bounded-label counters, gauges, and duration aggregates.
OpenTelemetry spans cover HTTP, policy, retrieval, generation, grounding,
persistence, speech, streaming, and MCP paths. JSON logs include correlation
IDs and bounded metadata but exclude request bodies, transcripts, audio,
secrets, and account identifiers.

Burn-rate response for this portfolio uses two windows: investigate when the
one-hour failure rate exceeds 5% or the six-hour rate exceeds 2%. Disable the
affected automation when safe fallback is unavailable or sensitive operations
cannot be audited.
