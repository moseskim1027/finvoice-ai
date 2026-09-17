from pathlib import Path

import pytest
from pydantic import ValidationError

from finvoice_ai.evaluation.speech_manifest import (
    SpeechEvaluationManifest,
    load_speech_manifest,
    write_manifest_schema,
)

EXAMPLE_MANIFEST = (
    Path(__file__).parents[1]
    / "src"
    / "finvoice_ai"
    / "evaluation"
    / "data"
    / "speech_manifest.example.json"
)


def test_example_manifest_records_provenance_and_evaluation_slices() -> None:
    manifest = load_speech_manifest(EXAMPLE_MANIFEST)

    assert manifest.dataset_name == "finvoice-synthetic-speech-eval"
    assert {case.language_mode for case in manifest.cases} == {"monolingual", "code-switched"}
    assert all(case.consent_basis and case.license for case in manifest.cases)


def test_manifest_rejects_duplicate_ids_and_unsafe_paths() -> None:
    case = {
        "case_id": "duplicate",
        "audio_path": "../private.wav",
        "reference_transcript": "hello",
        "language": "en",
        "language_mode": "monolingual",
        "noise_condition": "clean",
        "device": "phone",
        "speaker_id": "speaker-a",
        "consent_basis": "explicit-consent",
        "license": "CC-BY-4.0",
        "split": "test",
    }

    with pytest.raises(ValidationError, match="audio_path"):
        SpeechEvaluationManifest(
            dataset_name="test",
            version="1",
            data_statement="test data",
            cases=[case, case],
        )


def test_manifest_schema_can_be_exported(tmp_path: Path) -> None:
    output = tmp_path / "speech-manifest.schema.json"

    write_manifest_schema(output)

    assert '"SpeechEvaluationManifest"' in output.read_text(encoding="utf-8")
