import hashlib
import json
from pathlib import Path

from finvoice_ai.evaluation.speech_integrity import audit_speech_dataset, main
from finvoice_ai.evaluation.speech_manifest import (
    SpeechEvaluationCase,
    SpeechEvaluationManifest,
)
from tests.audio_helpers import create_wav


def make_case(
    case_id: str,
    audio_path: str,
    audio_sha256: str,
    *,
    split: str = "development",
    speaker_id: str | None = None,
    transcript: str = "Help me reset my PIN",
) -> SpeechEvaluationCase:
    return SpeechEvaluationCase(
        case_id=case_id,
        audio_path=audio_path,
        reference_transcript=transcript,
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        utterance_type="short-command",
        intent_id="pin_reset",
        speaker_id=speaker_id or f"speaker-{case_id}",
        consent_basis="synthetic-generated",
        license="project-generated",
        provenance="unit test fixture",
        collection_method="deterministic tone fixture",
        audio_sha256=audio_sha256,
        split=split,
    )


def make_manifest(cases: list[SpeechEvaluationCase]) -> SpeechEvaluationManifest:
    return SpeechEvaluationManifest(
        dataset_name="integrity-fixture",
        version="1.0.0",
        taxonomy_version="1.0.0",
        transform_version="1.0.0",
        data_statement="synthetic unit test fixtures",
        cases=cases,
    )


def test_integrity_audit_verifies_hash_and_wav_contract(tmp_path: Path) -> None:
    content = create_wav()
    (tmp_path / "valid.wav").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    report = audit_speech_dataset(
        make_manifest([make_case("valid", "valid.wav", digest)]), tmp_path
    )

    assert report.valid is True
    assert report.verified_file_count == 1
    assert report.issues == []


def test_integrity_audit_reports_missing_invalid_and_changed_audio(tmp_path: Path) -> None:
    invalid = b"not-a-wav"
    changed = create_wav(amplitude=100)
    (tmp_path / "invalid.wav").write_bytes(invalid)
    (tmp_path / "changed.wav").write_bytes(changed)
    cases = [
        make_case("missing", "missing.wav", "0" * 64),
        make_case("invalid", "invalid.wav", hashlib.sha256(invalid).hexdigest()),
        make_case("changed", "changed.wav", "1" * 64),
    ]

    report = audit_speech_dataset(make_manifest(cases), tmp_path)

    assert report.valid is False
    assert {issue.code for issue in report.issues} == {
        "missing_audio",
        "invalid_wav",
        "hash_mismatch",
        "duplicate_transcript",
    }


def test_integrity_audit_rejects_cross_split_transcript_and_audio_leakage(
    tmp_path: Path,
) -> None:
    content = create_wav()
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / "one.wav").write_bytes(content)
    (tmp_path / "two.wav").write_bytes(content)
    cases = [
        make_case("development", "one.wav", digest),
        make_case("test", "two.wav", digest, split="test"),
    ]

    report = audit_speech_dataset(make_manifest(cases), tmp_path)

    assert {issue.code for issue in report.issues} == {
        "transcript_leakage",
        "audio_leakage",
    }


def test_integrity_cli_writes_report_and_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        make_manifest([make_case("missing", "missing.wav", "0" * 64)]).model_dump_json(),
        encoding="utf-8",
    )
    output = tmp_path / "integrity.json"

    exit_code = main([str(manifest_path), "--dataset-root", str(tmp_path), "--output", str(output)])

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["issues"][0]["code"] == "missing_audio"
