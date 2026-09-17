import argparse
import hashlib
import json
import platform
import random
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from finvoice_ai.evaluation.speech_report import SpeechEvaluationReport


def compare_reports(
    baseline: SpeechEvaluationReport,
    candidate: SpeechEvaluationReport,
    *,
    seed: int = 20260918,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    if baseline.evaluation_split != "development" or candidate.evaluation_split != "development":
        raise ValueError("model selection comparisons must use development reports only")
    baseline_cases = {case.case_id: case for case in baseline.cases}
    candidate_cases = {case.case_id: case for case in candidate.cases}
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("reports must contain the same case IDs")
    case_ids = sorted(baseline_cases)
    wer_differences = [
        candidate_cases[case_id].word_error_rate - baseline_cases[case_id].word_error_rate
        for case_id in case_ids
    ]
    cer_differences = [
        candidate_cases[case_id].character_error_rate - baseline_cases[case_id].character_error_rate
        for case_id in case_ids
    ]
    return {
        "dataset_name": baseline.dataset_name,
        "dataset_version": baseline.dataset_version,
        "evaluation_split": "development",
        "baseline_model": _single_model(baseline),
        "candidate_model": _single_model(candidate),
        "baseline": baseline.overall.model_dump(mode="json"),
        "candidate": candidate.overall.model_dump(mode="json"),
        "paired_mean_word_error_rate_delta": _mean(wer_differences),
        "paired_mean_character_error_rate_delta": _mean(cer_differences),
        "word_error_rate_delta_bootstrap_95_ci": _bootstrap_interval(
            wer_differences, seed, bootstrap_samples
        ),
        "character_error_rate_delta_bootstrap_95_ci": _bootstrap_interval(
            cer_differences, seed + 1, bootstrap_samples
        ),
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "case_count": len(case_ids),
    }


def environment_record(manifest_path: Path) -> dict[str, Any]:
    packages = {}
    for package in ("faster-whisper", "ctranslate2", "numpy", "finvoice-ai"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _single_model(report: SpeechEvaluationReport) -> str:
    models = {case.model for case in report.cases}
    if len(models) != 1:
        raise ValueError("each report must contain exactly one model identifier")
    return models.pop()


def _bootstrap_interval(values: list[float], seed: int, samples: int) -> list[float]:
    if not values or samples <= 0:
        return [0.0, 0.0]
    generator = random.Random(seed)
    estimates = sorted(_mean([generator.choice(values) for _ in values]) for _ in range(samples))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _percentile(values: list[float], quantile: float) -> float:
    index = round((len(values) - 1) * quantile)
    return values[index]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare development ASR reports")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = SpeechEvaluationReport.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    candidate = SpeechEvaluationReport.model_validate_json(
        args.candidate.read_text(encoding="utf-8")
    )
    comparison = compare_reports(
        baseline,
        candidate,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    comparison["environment"] = environment_record(args.manifest)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
