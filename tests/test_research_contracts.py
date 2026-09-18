import pytest
from pydantic import ValidationError

from finvoice_ai.research.contracts import (
    INTENT_LABELS,
    ExperimentConfig,
    ResearchCase,
    ResearchDataset,
    role_for_case,
)


def make_case(case_id: str, speaker: str, role: str) -> ResearchCase:
    return ResearchCase(
        case_id=case_id,
        audio_path=f"audio/{case_id}.wav",
        audio_sha256="0" * 64,
        reference_transcript="Help me reset my PIN",
        asr_transcript="help me reset my pin",
        speaker_id=speaker,
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        intent=INTENT_LABELS["pin_reset"],
        role=role,
    )


def test_policy_labels_do_not_derive_escalation_from_acoustics() -> None:
    assert INTENT_LABELS["pin_reset"].escalation_required is False
    assert INTENT_LABELS["card_security"].escalation_required is True
    assert INTENT_LABELS["money_transfer"].scope == "out-of-scope"
    assert all(label.policy_basis for label in INTENT_LABELS.values())


def test_role_assignment_keeps_manifest_test_held_out() -> None:
    assert role_for_case("development", "voice-a") == "train"
    assert role_for_case("development", "voice-c") == "validation"
    assert role_for_case("test", "voice-a") == "test"


def test_dataset_rejects_speaker_leakage_and_missing_roles() -> None:
    cases = [
        make_case("train", "speaker-a", "train"),
        make_case("validation", "speaker-a", "validation"),
        make_case("test", "speaker-b", "test"),
    ]

    with pytest.raises(ValidationError, match="exactly one role"):
        ResearchDataset(name="fixture", version="1", cases=cases)

    with pytest.raises(ValidationError, match="missing roles"):
        ResearchDataset(name="fixture", version="1", cases=[cases[0]])


def test_config_fingerprint_changes_with_selection_setting() -> None:
    config = ExperimentConfig(
        experiment_id="baseline-v1",
        dataset_version="1",
        manifest_sha256="0" * 64,
        seed=17,
    )

    assert config.fingerprint == config.model_copy().fingerprint
    assert config.fingerprint != config.model_copy(update={"regularization_c": 0.5}).fingerprint
