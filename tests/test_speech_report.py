import pytest

from finvoice_ai.evaluation.speech_report import (
    SpeechCaseResult,
    build_slice_summaries,
    summarize_results,
)


def make_result(case_id: str, language: str, latency_ms: float) -> SpeechCaseResult:
    return SpeechCaseResult(
        case_id=case_id,
        language=language,
        language_mode="monolingual",
        noise_condition="clean",
        device="phone",
        speaker_id=f"speaker-{case_id}",
        intent_id="pin_reset",
        utterance_type="short-command",
        status="success",
        hypothesis="hello",
        detected_language=language,
        model="fake",
        confidence=0.75,
        word_errors=1,
        reference_words=4,
        character_errors=2,
        reference_characters=10,
        latency_ms=latency_ms,
        audio_duration_seconds=2.0,
    )


def test_summary_uses_micro_error_rates_and_nearest_rank_latency() -> None:
    first = make_result("one", "en", 100.0)
    second = make_result("two", "fil", 300.0).model_copy(
        update={"status": "failed", "word_errors": 4, "reference_words": 4}
    )

    summary = summarize_results([first, second])

    assert summary.case_count == 2
    assert summary.failure_count == 1
    assert summary.failure_rate == 0.5
    assert summary.word_error_rate == pytest.approx(5 / 8)
    assert summary.character_error_rate == 0.2
    assert summary.macro_word_error_rate == pytest.approx(0.625)
    assert summary.macro_character_error_rate == 0.2
    assert summary.language_identification_accuracy == 1.0
    assert summary.expected_calibration_error == 0.75
    assert summary.latency_p50_ms == 100.0
    assert summary.latency_p95_ms == 300.0
    assert summary.mean_real_time_factor == 0.1


def test_slice_summaries_cover_each_declared_dimension() -> None:
    summaries = build_slice_summaries(
        [make_result("one", "en", 100.0), make_result("two", "fil", 200.0)]
    )

    assert {(summary.dimension, summary.value) for summary in summaries} == {
        ("language", "en"),
        ("language", "fil"),
        ("language_mode", "monolingual"),
        ("noise_condition", "clean"),
        ("device", "phone"),
        ("speaker_id", "speaker-one"),
        ("speaker_id", "speaker-two"),
        ("intent_id", "pin_reset"),
        ("utterance_type", "short-command"),
    }


def test_empty_summary_is_explicitly_zeroed() -> None:
    summary = summarize_results([])

    assert summary.case_count == 0
    assert summary.failure_rate == 0.0
    assert summary.latency_p95_ms == 0.0
    assert summary.expected_calibration_error == 0.0
