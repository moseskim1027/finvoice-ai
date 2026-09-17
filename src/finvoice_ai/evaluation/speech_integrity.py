import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from finvoice_ai.evaluation.speech_manifest import (
    SpeechEvaluationManifest,
    load_speech_manifest,
)
from finvoice_ai.speech.wav import InvalidAudioError, PcmWavLoader


class IntegrityIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    case_ids: tuple[str, ...] = ()


class IntegrityReport(BaseModel):
    dataset_name: str
    dataset_version: str
    case_count: int
    verified_file_count: int
    issues: list[IntegrityIssue]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def audit_speech_dataset(
    manifest: SpeechEvaluationManifest,
    dataset_root: Path,
) -> IntegrityReport:
    root = dataset_root.resolve()
    loader = PcmWavLoader()
    issues: list[IntegrityIssue] = []
    verified = 0
    transcripts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for case in manifest.cases:
        transcripts[_normalize_transcript(case.reference_transcript)].append(
            (case.case_id, case.split)
        )
        hashes[case.audio_sha256].append((case.case_id, case.split))
        path = (root / case.audio_path).resolve()
        if not path.is_relative_to(root):
            issues.append(_issue("unsafe_path", "audio path escapes dataset root", case.case_id))
            continue
        try:
            content = path.read_bytes()
        except OSError as error:
            issues.append(_issue("missing_audio", type(error).__name__, case.case_id))
            continue
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != case.audio_sha256:
            issues.append(
                _issue(
                    "hash_mismatch",
                    f"expected {case.audio_sha256}, calculated {actual_hash}",
                    case.case_id,
                )
            )
            continue
        try:
            loader.load(content)
        except InvalidAudioError as error:
            issues.append(_issue("invalid_wav", str(error), case.case_id))
            continue
        verified += 1

    issues.extend(_cross_split_issues("transcript_leakage", transcripts))
    issues.extend(_cross_split_issues("audio_leakage", hashes))
    issues.extend(_within_split_duplicate_warnings(transcripts))
    return IntegrityReport(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.version,
        case_count=len(manifest.cases),
        verified_file_count=verified,
        issues=issues,
    )


def _normalize_transcript(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _issue(code: str, message: str, *case_ids: str) -> IntegrityIssue:
    return IntegrityIssue(severity="error", code=code, message=message, case_ids=case_ids)


def _cross_split_issues(
    code: str,
    values: dict[str, list[tuple[str, str]]],
) -> list[IntegrityIssue]:
    issues = []
    for occurrences in values.values():
        if len({split for _, split in occurrences}) > 1:
            case_ids = tuple(case_id for case_id, _ in occurrences)
            issues.append(_issue(code, "value occurs in development and test", *case_ids))
    return issues


def _within_split_duplicate_warnings(
    transcripts: dict[str, list[tuple[str, str]]],
) -> list[IntegrityIssue]:
    issues = []
    for occurrences in transcripts.values():
        if len(occurrences) > 1 and len({split for _, split in occurrences}) == 1:
            issues.append(
                IntegrityIssue(
                    severity="warning",
                    code="duplicate_transcript",
                    message="normalized transcript occurs more than once within one split",
                    case_ids=tuple(case_id for case_id, _ in occurrences),
                )
            )
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a governed speech dataset")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_speech_dataset(load_speech_manifest(args.manifest), args.dataset_root)
    output = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return int(not report.valid)


if __name__ == "__main__":
    raise SystemExit(main())
