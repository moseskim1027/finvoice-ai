import json
from pathlib import Path

import pytest

from finvoice_ai.evaluation.speech_report import (
    SpeechCaseResult,
    SpeechEvaluationReport,
    summarize_results,
)
from finvoice_ai.research.dataset import load_research_dataset, manifest_sha256


def _result(case_id: str, hypothesis: str) -> SpeechCaseResult:
    return SpeechCaseResult(
        case_id=case_id,
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        speaker_id="speaker",
        intent_id="pin_reset",
        utterance_type="short-command",
        status="success",
        hypothesis=hypothesis,
        detected_language="en",
        model="fake",
        confidence=0.5,
        word_errors=0,
        reference_words=1,
        character_errors=0,
        reference_characters=1,
        latency_ms=1.0,
        audio_duration_seconds=1.0,
    )


def _report(path: Path, cases: list[SpeechCaseResult], split: str) -> None:
    path.write_text(
        SpeechEvaluationReport(
            dataset_name="fixture",
            dataset_version="1",
            evaluation_split=split,
            overall=summarize_results(cases),
            slices=[],
            cases=cases,
        ).model_dump_json(),
        encoding="utf-8",
    )


def test_research_loader_joins_manifest_and_asr_by_case_id(tmp_path: Path) -> None:
    manifest = json.loads(
        Path("src/finvoice_ai/evaluation/data/speech_manifest.example.json").read_text()
    )
    manifest["cases"][0]["speaker_id"] = "voice-c"
    train_case = dict(manifest["cases"][0])
    train_case.update(
        {
            "case_id": "train-case",
            "audio_path": "audio/train.wav",
            "audio_sha256": "2" * 64,
            "speaker_id": "voice-a",
        }
    )
    manifest["cases"].append(train_case)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    development = tmp_path / "development.json"
    test = tmp_path / "test.json"
    _report(
        development,
        [
            _result(manifest["cases"][0]["case_id"], "development text"),
            _result(train_case["case_id"], "training text"),
        ],
        "development",
    )
    _report(test, [_result(manifest["cases"][1]["case_id"], "test text")], "test")

    dataset = load_research_dataset(manifest_path, development, test)

    assert [case.role for case in dataset.cases] == ["validation", "test", "train"]
    assert dataset.cases[0].asr_transcript == "development text"
    assert len(manifest_sha256(manifest_path)) == 64


def test_research_loader_rejects_missing_asr_cases(tmp_path: Path) -> None:
    manifest_path = Path("src/finvoice_ai/evaluation/data/speech_manifest.example.json")
    empty = tmp_path / "empty.json"
    _report(empty, [], "development")

    with pytest.raises(ValueError, match="case mismatch"):
        load_research_dataset(manifest_path, empty, empty)
