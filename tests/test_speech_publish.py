import json
from pathlib import Path

from finvoice_ai.evaluation.speech_publish import main, safe_aggregate
from finvoice_ai.evaluation.speech_report import (
    SpeechCaseResult,
    SpeechEvaluationReport,
    build_slice_summaries,
    summarize_results,
)


def report() -> SpeechEvaluationReport:
    case = SpeechCaseResult(
        case_id="one",
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        speaker_id="speaker-a",
        intent_id="pin_reset",
        utterance_type="short-command",
        status="success",
        hypothesis="restricted utterance",
        detected_language="en",
        model="faster-whisper/base",
        confidence=0.7,
        word_errors=1,
        reference_words=2,
        character_errors=2,
        reference_characters=10,
        latency_ms=5.0,
        audio_duration_seconds=1.0,
    )
    return SpeechEvaluationReport(
        dataset_name="fixture",
        dataset_version="1",
        evaluation_split="test",
        overall=summarize_results([case]),
        slices=build_slice_summaries([case]),
        cases=[case],
    )


def test_safe_aggregate_excludes_per_utterance_content(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    payload = safe_aggregate(
        report(),
        manifest_path=manifest,
        configuration={"beam_size": 1},
    )

    serialized = json.dumps(payload)
    assert payload["overall"]["case_count"] == 1
    assert "restricted utterance" not in serialized
    assert "cases" not in payload


def test_publish_cli_records_frozen_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "aggregate.json"
    source.write_text(report().model_dump_json(), encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")

    exit_code = main(
        [
            str(source),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--model-size",
            "base",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--beam-size",
            "1",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["configuration"]["model_size"] == "base"
    assert payload["evaluation_split"] == "test"
