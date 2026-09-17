import json
from pathlib import Path

import pytest

from finvoice_ai.evaluation.speech_manifest import (
    SpeechEvaluationCase,
    SpeechEvaluationManifest,
)
from finvoice_ai.evaluation.speech_runner import SpeechEvaluationRunner, main
from finvoice_ai.speech.models import SpeechAnalysis, TranscriptionResult
from tests.audio_helpers import create_wav


class FakeSpeechService:
    def __init__(self, outcomes: list[SpeechAnalysis | Exception]) -> None:
        self._outcomes = iter(outcomes)

    def analyze(self, wav_data: bytes) -> SpeechAnalysis:
        assert wav_data == b"wav"
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_case(case_id: str, audio_path: str, split: str = "development") -> SpeechEvaluationCase:
    return SpeechEvaluationCase(
        case_id=case_id,
        audio_path=audio_path,
        reference_transcript="hello world",
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="phone",
        utterance_type="short-command",
        intent_id="pin_reset",
        speaker_id=f"speaker-{case_id}",
        consent_basis="synthetic-generated",
        license="project-generated",
        provenance="test fixture",
        collection_method="test fixture",
        audio_sha256="0" * 64,
        split=split,
    )


def make_manifest(cases: list[SpeechEvaluationCase]) -> SpeechEvaluationManifest:
    return SpeechEvaluationManifest(
        dataset_name="fixture",
        version="1.0",
        taxonomy_version="1.0.0",
        transform_version="1.0.0",
        data_statement="synthetic fixture",
        cases=cases,
    )


def test_runner_reports_accuracy_latency_and_missing_files(tmp_path: Path) -> None:
    (tmp_path / "ok.wav").write_bytes(b"wav")
    analysis = SpeechAnalysis(
        duration_seconds=2.0,
        segments=(),
        transcription=TranscriptionResult(
            text="hello word",
            confidence=0.8,
            language="en",
            model="fake-v1",
        ),
    )
    clock = iter([10.0, 10.05])
    runner = SpeechEvaluationRunner(
        service=FakeSpeechService([analysis]),
        provider_name="fake-v1",
        clock=lambda: next(clock),
    )

    report = runner.run(
        make_manifest(
            [
                make_case("success", "ok.wav"),
                make_case("missing", "missing.wav"),
            ]
        ),
        tmp_path,
    )

    assert report.overall.case_count == 2
    assert report.overall.failure_count == 1
    assert report.overall.word_error_rate == pytest.approx(3 / 4)
    assert report.cases[0].latency_ms == pytest.approx(50.0)
    assert report.cases[0].real_time_factor == pytest.approx(0.025)
    assert report.cases[1].error_type == "FileNotFoundError"
    assert report.cases[1].word_error_rate == 1.0
    assert report.cases[1].real_time_factor == 0.0
    assert report.cases[0].model_dump()["word_error_rate"] == 0.5
    assert report.cases[0].confidence == 0.8
    assert report.cases[0].intent_id == "pin_reset"


def test_runner_records_provider_error_and_continues(tmp_path: Path) -> None:
    (tmp_path / "error.wav").write_bytes(b"wav")
    (tmp_path / "ok.wav").write_bytes(b"wav")
    analysis = SpeechAnalysis(
        duration_seconds=1.0,
        segments=(),
        transcription=TranscriptionResult("hello world", 1.0, "en", "fake-v1"),
    )
    clock = iter([1.0, 1.01, 2.0, 2.02])
    runner = SpeechEvaluationRunner(
        service=FakeSpeechService([RuntimeError("provider unavailable"), analysis]),
        provider_name="fake-v1",
        clock=lambda: next(clock),
    )

    report = runner.run(
        make_manifest([make_case("error", "error.wav"), make_case("success", "ok.wav")]),
        tmp_path,
    )

    assert [case.status for case in report.cases] == ["failed", "success"]
    assert report.cases[0].error_type == "RuntimeError"
    assert report.cases[0].latency_ms == pytest.approx(10.0)


def test_runner_filters_split_before_reading_audio(tmp_path: Path) -> None:
    runner = SpeechEvaluationRunner(
        service=FakeSpeechService([]),
        provider_name="fake-v1",
    )

    report = runner.run(
        make_manifest([make_case("held-out", "missing.wav", split="test")]),
        tmp_path,
        split="development",
    )

    assert report.evaluation_split == "development"
    assert report.cases == []
    assert report.overall.case_count == 0


def test_cli_writes_machine_readable_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FINVOICE_TRANSCRIPTION_PROVIDER", "deterministic")
    (tmp_path / "speech.wav").write_bytes(create_wav(duration_seconds=0.2))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        make_manifest([make_case("speech", "speech.wav")]).model_dump_json(),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    exit_code = main(
        [
            str(manifest_path),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dataset_name"] == "fixture"
    assert report["overall"]["case_count"] == 1
    assert report["cases"][0]["model"] == "deterministic-asr-stub-v1"


def test_cli_returns_nonzero_when_a_case_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FINVOICE_TRANSCRIPTION_PROVIDER", "deterministic")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        make_manifest([make_case("missing", "missing.wav")]).model_dump_json(),
        encoding="utf-8",
    )

    exit_code = main([str(manifest_path), "--dataset-root", str(tmp_path)])

    assert exit_code == 1
