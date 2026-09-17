# Speech benchmark error taxonomy

Use these mutually compatible tags during aggregate error review. Do not commit
restricted utterances or per-case annotations.

- **deletion** — a reference word is absent from the hypothesis;
- **insertion** — a hypothesis word has no aligned reference word;
- **substitution** — an aligned reference word is replaced;
- **named-entity or amount error** — an annotated entity or numeric amount is
  deleted, inserted, or substituted; the starter templates contain no real
  entities or amounts, so this category is not currently estimable;
- **code-switch boundary error** — an error occurs at an annotated language
  transition rather than elsewhere in the utterance;
- **noise failure** — a case succeeds in its paired clean form but fails under
  the documented noise transform;
- **VAD failure** — speech is absent from the selected segments or endpointing
  truncates speech;
- **language-identification failure** — the detected language is not English or
  Filipino as appropriate for the manifest language mode.

Deletion, insertion, and substitution counts should come from a deterministic
minimum-edit alignment. The remaining tags require explicit annotations or
paired conditions; they must be reported as “not estimable” when those inputs
are absent. Never infer intent, emotion, identity, honesty, or trustworthiness
from an acoustic or ASR error.
