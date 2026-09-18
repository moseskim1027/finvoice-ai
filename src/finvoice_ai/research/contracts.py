import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class IntentLabel(BaseModel):
    intent_id: Literal[
        "pin_reset",
        "statement_access",
        "card_security",
        "money_transfer",
        "unrelated_request",
    ]
    scope: Literal["in-scope", "out-of-scope"]
    escalation_required: bool
    policy_basis: str = Field(min_length=1)


class ResearchCase(BaseModel):
    case_id: str = Field(min_length=1)
    audio_path: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_transcript: str = Field(min_length=1)
    asr_transcript: str
    speaker_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    language_mode: str = Field(min_length=1)
    noise_condition: str = Field(min_length=1)
    device: str = Field(min_length=1)
    intent: IntentLabel
    role: Literal["train", "validation", "test"]


class ExperimentConfig(BaseModel):
    experiment_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    transcript_source: Literal["reference", "asr"] = "asr"
    text_min_document_frequency: int = Field(default=1, ge=1)
    regularization_c: float = Field(default=1.0, gt=0.0)
    maximum_iterations: int = Field(default=1_000, ge=1)
    calibration_method: Literal["none", "temperature", "sigmoid", "isotonic"] = "temperature"
    abstention_minimum_accuracy: float = Field(default=0.8, ge=0.0, le=1.0)
    frozen_embedding_model: str = "microsoft/wavlm-base-plus@8d0b7c7"
    acoustic_feature_version: str = "summary-v1"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class ResearchDataset(BaseModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    cases: list[ResearchCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_speaker_groups_and_case_ids(self) -> "ResearchDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("research case IDs must be unique")
        speaker_roles: dict[str, set[str]] = {}
        for case in self.cases:
            speaker_roles.setdefault(case.speaker_id, set()).add(case.role)
        leaking = sorted(speaker for speaker, roles in speaker_roles.items() if len(roles) > 1)
        if leaking:
            raise ValueError("speaker IDs must belong to exactly one role: " + ", ".join(leaking))
        role_counts = Counter(case.role for case in self.cases)
        missing = {"train", "validation", "test"} - role_counts.keys()
        if missing:
            raise ValueError("research dataset is missing roles: " + ", ".join(sorted(missing)))
        return self


INTENT_LABELS = {
    "pin_reset": IntentLabel(
        intent_id="pin_reset",
        scope="in-scope",
        escalation_required=False,
        policy_basis="Approved self-service support topic",
    ),
    "statement_access": IntentLabel(
        intent_id="statement_access",
        scope="in-scope",
        escalation_required=False,
        policy_basis="Approved informational support topic",
    ),
    "card_security": IntentLabel(
        intent_id="card_security",
        scope="in-scope",
        escalation_required=True,
        policy_basis="Potential compromise requires human review",
    ),
    "money_transfer": IntentLabel(
        intent_id="money_transfer",
        scope="out-of-scope",
        escalation_required=True,
        policy_basis="Agent must not execute sensitive financial transactions",
    ),
    "unrelated_request": IntentLabel(
        intent_id="unrelated_request",
        scope="out-of-scope",
        escalation_required=True,
        policy_basis="No approved support context",
    ),
}


def role_for_case(manifest_split: str, speaker_id: str) -> Literal["train", "validation", "test"]:
    if manifest_split == "test":
        return "test"
    if speaker_id == "voice-c":
        return "validation"
    return "train"
