import json
from pathlib import Path

import pytest

from finvoice_ai.evaluation.speech_comparison import compare_reports, main
from finvoice_ai.evaluation.speech_report import (
    SpeechCaseResult,
    SpeechEvaluationReport,
    build_slice_summaries,
    summarize_results,
)


def result(case_id: str, word_errors: int, character_errors: int, model: str) -> SpeechCaseResult:
    return SpeechCaseResult(
        case_id=case_id,
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        speaker_id="speaker-a",
        intent_id="pin_reset",
        utterance_type="short-command",
        status="success",
        hypothesis="text",
        detected_language="en",
        model=model,
        confidence=0.5,
        word_errors=word_errors,
        reference_words=4,
        character_errors=character_errors,
        reference_characters=10,
        latency_ms=10.0,
        audio_duration_seconds=1.0,
    )


def report(
    model: str, errors: tuple[int, int], split: str = "development"
) -> SpeechEvaluationReport:
    cases = [
        result("one", errors[0], errors[0], model),
        result("two", errors[1], errors[1], model),
    ]
    return SpeechEvaluationReport(
        dataset_name="fixture",
        dataset_version="1",
        evaluation_split=split,
        overall=summarize_results(cases),
        slices=build_slice_summaries(cases),
        cases=cases,
    )


def test_comparison_reports_paired_delta_and_reproducible_interval() -> None:
    comparison = compare_reports(
        report("baseline", (2, 2)),
        report("candidate", (1, 0)),
        seed=7,
        bootstrap_samples=100,
    )

    assert comparison["paired_mean_word_error_rate_delta"] == pytest.approx(-0.375)
    assert comparison["word_error_rate_delta_bootstrap_95_ci"] == [-0.5, -0.25]
    assert comparison["bootstrap_seed"] == 7


def test_comparison_rejects_test_selection_and_mismatched_cases() -> None:
    with pytest.raises(ValueError, match="development reports only"):
        compare_reports(report("one", (0, 0), split="test"), report("two", (0, 0)))
    mismatched = report("two", (0, 0))
    mismatched.cases[0].case_id = "different"
    with pytest.raises(ValueError, match="same case IDs"):
        compare_reports(report("one", (0, 0)), mismatched)


def test_cli_writes_aggregate_without_case_hypotheses(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "comparison.json"
    baseline_path.write_text(report("baseline", (1, 1)).model_dump_json(), encoding="utf-8")
    candidate_path.write_text(report("candidate", (0, 0)).model_dump_json(), encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")

    exit_code = main(
        [
            str(baseline_path),
            str(candidate_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--bootstrap-samples",
            "10",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["baseline_model"] == "baseline"
    assert "cases" not in payload
    assert "hypothesis" not in output.read_text(encoding="utf-8")
