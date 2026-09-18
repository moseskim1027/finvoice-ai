import hashlib
import json
from pathlib import Path

from finvoice_ai.evaluation.speech_manifest import load_speech_manifest
from finvoice_ai.evaluation.speech_report import SpeechEvaluationReport
from finvoice_ai.research.contracts import (
    INTENT_LABELS,
    ResearchCase,
    ResearchDataset,
    role_for_case,
)


def load_research_dataset(
    manifest_path: Path,
    development_asr_report: Path,
    test_asr_report: Path,
) -> ResearchDataset:
    manifest = load_speech_manifest(manifest_path)
    hypotheses = _load_hypotheses(development_asr_report, test_asr_report)
    manifest_case_ids = {case.case_id for case in manifest.cases}
    if hypotheses.keys() != manifest_case_ids:
        missing = sorted(manifest_case_ids - hypotheses.keys())
        extra = sorted(hypotheses.keys() - manifest_case_ids)
        raise ValueError(f"ASR report case mismatch; missing={missing}, extra={extra}")
    cases = [
        ResearchCase(
            case_id=case.case_id,
            audio_path=case.audio_path,
            audio_sha256=case.audio_sha256,
            reference_transcript=case.reference_transcript,
            asr_transcript=hypotheses[case.case_id],
            speaker_id=case.speaker_id,
            language=case.language,
            language_mode=case.language_mode,
            noise_condition=case.noise_condition,
            device=case.device,
            intent=INTENT_LABELS[case.intent_id],
            role=role_for_case(case.split, case.speaker_id),
        )
        for case in manifest.cases
    ]
    return ResearchDataset(name=manifest.dataset_name, version=manifest.version, cases=cases)


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_research_snapshot(dataset: ResearchDataset, path: Path) -> None:
    """Write a local restricted snapshot; callers must keep it outside Git."""
    path.write_text(
        json.dumps(dataset.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_hypotheses(*paths: Path) -> dict[str, str]:
    hypotheses: dict[str, str] = {}
    for path in paths:
        report = SpeechEvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))
        for case in report.cases:
            if case.case_id in hypotheses:
                raise ValueError(f"duplicate ASR case ID: {case.case_id}")
            hypotheses[case.case_id] = case.hypothesis
    return hypotheses
