# Bilingual speech benchmark design

Version 1.0 freezes the starter taxonomy before audio collection or generation.
The benchmark is intended to expose engineering failures, not support claims
about Filipino speakers or any broader population.

## Target composition

The governed local dataset should contain 60–120 utterances and balance:

- English, Filipino, and English–Filipino code-switching;
- clean, household, street, and cafe-like conditions;
- high-quality and phone-simulated capture;
- short commands and longer conversational requests;
- development and held-out test speakers.

Each of the 15 safe templates should be rendered by at least four distinct
synthetic or consented speaker identities. Variants may change polite wording
but must preserve the frozen intent and escalation-policy label. No recording
may contain a real name, account identifier, transaction, address, phone
number, or other personal financial information.

## Labels

`pin_reset` and `statement_access` are approved informational support topics.
`card_security` requires escalation because it may indicate compromise.
`money_transfer` requires escalation because the agent must not execute a
sensitive financial transaction. `unrelated_request` is out of scope because
no approved support context exists.

Escalation labels are derived only from these explicit policies. Acoustic
features and ASR confidence must never be interpreted as emotion, intent,
honesty, or trustworthiness.

## Split policy

Speaker identities must be disjoint between development and test. Development
is used for model, decoding, VAD, transform, threshold, and calibration choices.
The test split is run once after those choices and the manifest hashes are
frozen. Duplicate normalized transcripts may not cross splits.

## Safe storage

Raw audio, local manifests containing restricted text, and per-utterance
reports stay under ignored `data/` or `reports/` directories. Git contains only
the taxonomy, schemas, tiny synthetic test fixtures, aggregate results, and
documentation safe for publication.
