import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SpeechEvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    audio_path: str = Field(min_length=1)
    reference_transcript: str = Field(min_length=1)
    language: str = Field(min_length=2)
    language_mode: Literal["monolingual", "code-switched"]
    noise_condition: str = Field(min_length=1)
    device: str = Field(min_length=1)
    utterance_type: Literal["short-command", "conversational-request"]
    intent_id: Literal[
        "pin_reset",
        "statement_access",
        "card_security",
        "money_transfer",
        "unrelated_request",
    ]
    speaker_id: str = Field(min_length=1)
    consent_basis: str = Field(min_length=1)
    license: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    collection_method: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["development", "test"]

    @model_validator(mode="after")
    def require_relative_audio_path(self) -> "SpeechEvaluationCase":
        if Path(self.audio_path).is_absolute() or ".." in Path(self.audio_path).parts:
            raise ValueError("audio_path must be relative and remain inside the dataset root")
        return self


class SpeechEvaluationManifest(BaseModel):
    dataset_name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    transform_version: str = Field(min_length=1)
    data_statement: str = Field(min_length=1)
    cases: list[SpeechEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> "SpeechEvaluationManifest":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique")
        speaker_splits: dict[str, set[str]] = {}
        for case in self.cases:
            speaker_splits.setdefault(case.speaker_id, set()).add(case.split)
        leaking_speakers = sorted(
            speaker for speaker, splits in speaker_splits.items() if len(splits) > 1
        )
        if leaking_speakers:
            raise ValueError(
                "speaker IDs must not cross development and test splits: "
                + ", ".join(leaking_speakers)
            )
        return self


def load_speech_manifest(path: Path) -> SpeechEvaluationManifest:
    return SpeechEvaluationManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest_schema(path: Path) -> None:
    path.write_text(
        json.dumps(SpeechEvaluationManifest.model_json_schema(), indent=2) + "\n",
        encoding="utf-8",
    )
