import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from finvoice_ai.evaluation.speech_comparison import environment_record
from finvoice_ai.evaluation.speech_report import SpeechEvaluationReport


def safe_aggregate(
    report: SpeechEvaluationReport,
    *,
    manifest_path: Path,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    models = sorted({case.model for case in report.cases})
    if len(models) != 1:
        raise ValueError("report must contain exactly one model identifier")
    return {
        "dataset_name": report.dataset_name,
        "dataset_version": report.dataset_version,
        "evaluation_split": report.evaluation_split,
        "model": models[0],
        "configuration": configuration,
        "overall": report.overall.model_dump(mode="json"),
        "slices": [slice_summary.model_dump(mode="json") for slice_summary in report.slices],
        "environment": environment_record(manifest_path),
        "publication_note": (
            "Aggregate synthetic-speech results only; raw audio, references, hypotheses, "
            "and per-utterance results are intentionally excluded."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a safe aggregate speech report")
    parser.add_argument("report", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--compute-type", required=True)
    parser.add_argument("--beam-size", type=int, required=True)
    parser.add_argument("--language", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = SpeechEvaluationReport.model_validate_json(args.report.read_text(encoding="utf-8"))
    payload = safe_aggregate(
        report,
        manifest_path=args.manifest,
        configuration={
            "model_size": args.model_size,
            "device": args.device,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "language": args.language,
        },
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
