import pytest

pytest.importorskip("sklearn")

from finvoice_ai.research.contracts import INTENT_LABELS, ResearchCase
from finvoice_ai.research.text_baseline import MajorityIntentBaseline, TfidfIntentBaseline


def case(case_id: str, text: str, intent: str) -> ResearchCase:
    return ResearchCase(
        case_id=case_id,
        audio_path=f"{case_id}.wav",
        audio_sha256="0" * 64,
        reference_transcript=text,
        asr_transcript=text,
        speaker_id="speaker",
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        intent=INTENT_LABELS[intent],
        role="train",
    )


def test_majority_baseline_is_deterministic_and_requires_fit() -> None:
    baseline = MajorityIntentBaseline()
    with pytest.raises(RuntimeError, match="fitted"):
        baseline.predict([])

    predictions = baseline.fit(
        [
            case("one", "reset pin", "pin_reset"),
            case("two", "change pin", "pin_reset"),
            case("three", "show statement", "statement_access"),
        ]
    ).predict([case("target", "anything", "statement_access")])

    assert predictions.predictions == ("pin_reset",)
    assert sum(predictions.probabilities[0]) == 1.0


def test_tfidf_baseline_learns_training_vocabulary_only() -> None:
    training = [
        case("pin-one", "reset my secret pin", "pin_reset"),
        case("pin-two", "change the secret pin", "pin_reset"),
        case("statement-one", "download monthly statement", "statement_access"),
        case("statement-two", "show account statement", "statement_access"),
    ]
    target = [case("target", "download statement", "statement_access")]

    prediction = TfidfIntentBaseline(seed=7).fit(training).predict(target)

    assert prediction.predictions == ("statement_access",)
    assert prediction.case_ids == ("target",)
    assert sum(prediction.probabilities[0]) == pytest.approx(1.0)
