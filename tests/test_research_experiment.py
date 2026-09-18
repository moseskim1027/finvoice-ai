from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sklearn")

from finvoice_ai.research.contracts import INTENT_LABELS, ExperimentConfig, ResearchCase
from finvoice_ai.research.experiment import run_experiment


def _case(index: int, role: str, intent: str) -> ResearchCase:
    words = {
        "pin_reset": "reset forgotten pin",
        "statement_access": "download account statement",
        "card_security": "stolen compromised card",
        "money_transfer": "send money transfer",
        "unrelated_request": "weather music unrelated",
    }
    return ResearchCase(
        case_id=f"{role}-{index}",
        audio_path=f"{role}-{index}.wav",
        audio_sha256=f"{index + 1:064x}",
        reference_transcript=words[intent],
        asr_transcript=words[intent],
        speaker_id={"train": "voice-a", "validation": "voice-c", "test": "voice-d"}[role],
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        intent=INTENT_LABELS[intent],
        role=role,
    )


def test_run_experiment_reports_validation_selection_and_test_results(monkeypatch) -> None:
    intents = list(INTENT_LABELS)
    cases = [
        _case(index, role, intent)
        for role in ("train", "validation", "test")
        for index, intent in enumerate(intents)
    ]
    dataset = SimpleNamespace(name="fixture", version="1", cases=cases)

    monkeypatch.setattr(
        "finvoice_ai.research.experiment.load_research_dataset",
        lambda *_args: dataset,
    )
    monkeypatch.setattr(
        "finvoice_ai.research.experiment.load_summary_feature_matrix",
        lambda selected, _root: [
            tuple(float(position == index) for position in range(8))
            for index, _case_value in enumerate(selected)
        ],
    )
    config = ExperimentConfig(
        experiment_id="fixture-study",
        dataset_version="1",
        manifest_sha256="0" * 64,
        seed=7,
    )

    report = run_experiment(
        Path("manifest.json"),
        Path("data"),
        Path("development.json"),
        Path("test.json"),
        config,
    )

    assert report["dataset"]["test_cases"] == 5
    assert report["selection"]["selected_on"] == "validation"
    assert set(report["selection"]["abstention_thresholds"]) == {
        "text",
        "acoustic",
        "late_fusion",
        "concatenated_fusion",
    }
    assert report["test_results"]["text"]["metrics"]["accuracy"] == 1.0
    assert (
        report["ablations"]["reference_vs_asr_text_validation"]["test_not_used_for_selection"] == 5
    )
    assert report["paired_text_vs_late_fusion"]["samples"] == 2_000
