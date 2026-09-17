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
        "utterance_type": "short-command",
        "intent_id": "pin_reset",
        "speaker_id": "speaker-a",
        "consent_basis": "explicit-consent",
        "license": "CC-BY-4.0",
        "provenance": "consented local collection",
        "collection_method": "recorded in a quiet room",
        "audio_sha256": "0" * 64,
        "split": "test",
    }

    with pytest.raises(ValidationError, match="audio_path"):
        SpeechEvaluationManifest(
            dataset_name="test",
            version="1",
            taxonomy_version="1.0.0",
            transform_version="1.0.0",
            data_statement="test data",
            cases=[case, case],
        )


def test_manifest_schema_can_be_exported(tmp_path: Path) -> None:
    output = tmp_path / "speech-manifest.schema.json"

    write_manifest_schema(output)

    assert '"SpeechEvaluationManifest"' in output.read_text(encoding="utf-8")


def test_manifest_rejects_speaker_leakage_between_splits() -> None:
    manifest = load_speech_manifest(EXAMPLE_MANIFEST)
    leaking = manifest.cases[1].model_copy(update={"speaker_id": manifest.cases[0].speaker_id})

    with pytest.raises(ValidationError, match="speaker IDs must not cross"):
        SpeechEvaluationManifest(
            dataset_name=manifest.dataset_name,
            version=manifest.version,
            taxonomy_version=manifest.taxonomy_version,
            transform_version=manifest.transform_version,
            data_statement=manifest.data_statement,
            cases=[manifest.cases[0], leaking],
        )
