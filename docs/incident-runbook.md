# Incident response runbook

This runbook covers the local FinVoice portfolio service. It has no connection
to a bank, customer accounts, or production customer data.

## First response

1. Confirm scope: affected route, environment, first occurrence, request ID,
   session ID, deployment revision, and whether safe escalation still works.
2. Check `/ready`, container status, recent JSON logs, RED metrics, and the
   correlated trace. Do not copy request bodies, audio, transcripts, secrets,
   or identifiers into an incident ticket.
3. Stop automated responses immediately if policy enforcement, authorization,
   citation validation, audit recording, or safe fallback is unreliable.
4. Preserve aggregate metrics, sanitized traces, configuration fingerprints,
   model/data versions, and relevant audit IDs.

## Symptom guide

| Symptom | Triage | Safe action |
| --- | --- | --- |
| Elevated 5xx/latency | Compare route and operation spans; check saturation and provider errors | Reduce concurrency or disable failing provider; keep escalation available |
| ASR timeouts | Check `asr.transcribe`, audio duration, model revision, CPU/memory | Return 503; do not guess a transcript |
| Retrieval failure/no context | Check corpus availability and retrieval result counter | Escalate with `provider_unavailable` or `missing_approved_context` |
| Invalid citations | Compare retrieved IDs with cited IDs | Suppress answer and escalate |
| MCP denial spike | Check authentication, scopes, confirmation, and audit outcome | Keep denial fail-closed; never weaken scopes during incident |
| Streaming rejections | Check active-session gauge, buffer limits, cleanup, disconnects | Shed load with explicit rejection; do not remove bounds |
| Quality regression | Freeze model/config, verify dataset hash and split, rerun gates | Roll back model/config independently of service availability |

## Rollback and recovery

- Roll back to the last image digest and configuration fingerprint that passed
  native, Docker, quality, and fault tests. Do not rebuild an old tag in place.
- Rotate a secret if exposure is suspected; never include its value in logs.
- For data/model drift, quarantine new artifacts, verify hashes and provenance,
  and rerun the frozen test split without tuning on it.
- Restore automation only after readiness, fault fallback, authorization,
  citation validation, and audit traceability pass.

## Disable-automation criteria

Disable automated responses if sensitive-action recall falls below its gate,
citations cannot be verified, audit events are missing, authorization fails
open, persistence is corrupt, or a provider error bypasses escalation. A human
handoff is the intended safe state, not an availability failure.

After recovery, record the timeline, detection gap, contributing conditions,
customer/data impact (expected to be none in this portfolio), rollback, and
specific test or monitor added to prevent recurrence.
