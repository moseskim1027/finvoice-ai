import pytest

pytest.importorskip("sklearn")

from finvoice_ai.research.contracts import INTENT_LABELS, ResearchCase
from finvoice_ai.research.evaluation import (
    evaluate_predictions,
    risk_coverage_curve,
    select_temperature,
    slice_metrics,
    temperature_scale,
)
from finvoice_ai.research.fusion import late_fuse
from finvoice_ai.research.text_baseline import PredictionSet


def prediction(probabilities: tuple[tuple[float, ...], ...]) -> PredictionSet:
    classes = ("pin_reset", "statement_access")
    return PredictionSet(
        case_ids=("one", "two"),
        labels=classes,
        predictions=tuple(classes[max(range(2), key=row.__getitem__)] for row in probabilities),
        probabilities=probabilities,
        classes=classes,
    )


def case(case_id: str, intent: str) -> ResearchCase:
    return ResearchCase(
        case_id=case_id,
        audio_path=f"{case_id}.wav",
        audio_sha256="0" * 64,
        reference_transcript="text",
        asr_transcript="text",
        speaker_id="speaker",
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        intent=INTENT_LABELS[intent],
        role="validation",
    )


def test_late_fusion_aligns_probabilities_and_cases() -> None:
    text = prediction(((0.8, 0.2), (0.4, 0.6)))
    acoustic = prediction(((0.6, 0.4), (0.2, 0.8)))

    fused = late_fuse(text, acoustic, 0.75)

    assert fused.probabilities[0] == pytest.approx((0.75, 0.25))
    assert fused.predictions == fused.labels
    with pytest.raises(ValueError, match="between zero and one"):
        late_fuse(text, acoustic, 2.0)


def test_metrics_calibration_and_selective_prediction() -> None:
    cases = [case("one", "pin_reset"), case("two", "statement_access")]
    predictions = prediction(((0.9, 0.1), (0.6, 0.4)))

    metrics = evaluate_predictions(predictions, cases)
    temperature = select_temperature(predictions)
    calibrated = temperature_scale(predictions, temperature)
    curve = risk_coverage_curve(calibrated, cases)

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix"] == [[1, 0], [1, 0]]
    assert temperature in {0.5, 0.75, 1.0, 1.5, 2.0, 3.0}
    assert curve[0]["coverage"] == 1.0
    assert slice_metrics(predictions, cases)[0]["case_count"] == 2
