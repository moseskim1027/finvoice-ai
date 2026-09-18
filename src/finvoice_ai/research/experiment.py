import argparse
import json
import pickle
import platform
import random
import resource
import sys
import time
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from finvoice_ai.research.acoustic import AcousticIntentBaseline, load_summary_feature_matrix
from finvoice_ai.research.contracts import ExperimentConfig, ResearchCase
from finvoice_ai.research.dataset import load_research_dataset, manifest_sha256
from finvoice_ai.research.evaluation import (
    evaluate_predictions,
    risk_coverage_curve,
    select_abstention_threshold,
    select_temperature,
    slice_metrics,
    temperature_scale,
)
from finvoice_ai.research.fusion import ConcatenatedIntentBaseline, late_fuse
from finvoice_ai.research.text_baseline import (
    MajorityIntentBaseline,
    PredictionSet,
    TfidfIntentBaseline,
)


def run_experiment(
    manifest_path: Path,
    dataset_root: Path,
    development_asr_report: Path,
    test_asr_report: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = load_research_dataset(manifest_path, development_asr_report, test_asr_report)
    train = [case for case in dataset.cases if case.role == "train"]
    validation = [case for case in dataset.cases if case.role == "validation"]
    test = [case for case in dataset.cases if case.role == "test"]
    train_features = load_summary_feature_matrix(train, dataset_root)
    validation_features = load_summary_feature_matrix(validation, dataset_root)
    test_features = load_summary_feature_matrix(test, dataset_root)

    model_timings: dict[str, dict[str, float]] = {}
    timer = time.perf_counter()
    majority = MajorityIntentBaseline().fit(train)
    model_timings["majority"] = {"training_selection_seconds": time.perf_counter() - timer}
    timer = time.perf_counter()
    majority_test = majority.predict(test)
    model_timings["majority"]["test_inference_seconds"] = time.perf_counter() - timer

    timer = time.perf_counter()
    text_model, text_c, text_validation = _select_text_model(train, validation, config.seed)
    model_timings["text"] = {"training_selection_seconds": time.perf_counter() - timer}
    timer = time.perf_counter()
    text_test = text_model.predict(test)
    model_timings["text"]["test_inference_seconds"] = time.perf_counter() - timer
    timer = time.perf_counter()
    acoustic_model, acoustic_c, acoustic_validation = _select_acoustic_model(
        train,
        validation,
        train_features,
        validation_features,
        config.seed,
    )
    model_timings["acoustic"] = {"training_selection_seconds": time.perf_counter() - timer}
    timer = time.perf_counter()
    acoustic_test = acoustic_model.predict(test_features, test)
    model_timings["acoustic"]["test_inference_seconds"] = time.perf_counter() - timer
    timer = time.perf_counter()
    concat_model, concat_c, concat_validation = _select_concat_model(
        train,
        validation,
        train_features,
        validation_features,
        config.seed,
    )
    model_timings["concatenated_fusion"] = {
        "training_selection_seconds": time.perf_counter() - timer
    }
    timer = time.perf_counter()
    concat_test = concat_model.predict(test, test_features)
    model_timings["concatenated_fusion"]["test_inference_seconds"] = time.perf_counter() - timer

    timer = time.perf_counter()
    fusion_weight = _select_fusion_weight(text_validation, acoustic_validation, validation)
    late_validation = late_fuse(text_validation, acoustic_validation, fusion_weight)
    model_timings["late_fusion"] = {"training_selection_seconds": time.perf_counter() - timer}
    timer = time.perf_counter()
    late_test = late_fuse(text_test, acoustic_test, fusion_weight)
    model_timings["late_fusion"]["test_inference_seconds"] = time.perf_counter() - timer

    validation_sets = {
        "text": text_validation,
        "acoustic": acoustic_validation,
        "late_fusion": late_validation,
        "concatenated_fusion": concat_validation,
    }
    test_sets = {
        "majority": majority_test,
        "text": text_test,
        "acoustic": acoustic_test,
        "late_fusion": late_test,
        "concatenated_fusion": concat_test,
    }
    temperatures = {
        name: select_temperature(predictions) for name, predictions in validation_sets.items()
    }
    calibrated_validation = {
        name: temperature_scale(predictions, temperatures[name])
        for name, predictions in validation_sets.items()
    }
    abstention_thresholds = {
        name: select_abstention_threshold(
            predictions,
            validation,
            minimum_accuracy=config.abstention_minimum_accuracy,
        )
        for name, predictions in calibrated_validation.items()
    }
    calibrated_test = {
        "majority": majority_test,
        **{
            name: temperature_scale(predictions, temperatures[name])
            for name, predictions in test_sets.items()
            if name != "majority"
        },
    }
    reference_ablation = _reference_text_ablation(train, validation, test, text_c, config.seed)
    acoustic_ablations = _acoustic_ablations(
        train,
        validation,
        train_features,
        validation_features,
        acoustic_c,
        config.seed,
    )
    models = {
        "majority": majority,
        "text": text_model,
        "acoustic": acoustic_model,
        "concatenated_fusion": concat_model,
    }
    results = {
        name: {
            "metrics": evaluate_predictions(predictions, test),
            "selective_at_threshold": evaluate_predictions(
                predictions,
                test,
                abstention_threshold=abstention_thresholds.get(name, 0.0),
            ),
            "risk_coverage": risk_coverage_curve(predictions, test),
            "slices": slice_metrics(predictions, test),
            "artifact_bytes": len(pickle.dumps(models.get(name, {}))),
            "temperature": temperatures.get(name, 1.0),
            "abstention_threshold": abstention_thresholds.get(name, 0.0),
            "timing": model_timings[name],
        }
        for name, predictions in calibrated_test.items()
    }
    results["late_fusion"]["artifact_bytes"] = (
        results["text"]["artifact_bytes"] + results["acoustic"]["artifact_bytes"]
    )
    comparison = _paired_accuracy_bootstrap(
        calibrated_test["text"],
        calibrated_test["late_fusion"],
        seed=config.seed,
    )
    concatenated_comparison = _paired_accuracy_bootstrap(
        calibrated_test["text"],
        calibrated_test["concatenated_fusion"],
        seed=config.seed + 1,
    )
    elapsed = time.perf_counter() - started
    return {
        "experiment_id": config.experiment_id,
        "config_fingerprint": config.fingerprint,
        "config": config.model_dump(mode="json"),
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "train_cases": len(train),
            "validation_cases": len(validation),
            "test_cases": len(test),
            "train_speakers": sorted({case.speaker_id for case in train}),
            "validation_speakers": sorted({case.speaker_id for case in validation}),
            "test_speakers": sorted({case.speaker_id for case in test}),
        },
        "selection": {
            "text_c": text_c,
            "acoustic_c": acoustic_c,
            "concatenated_c": concat_c,
            "late_fusion_text_weight": fusion_weight,
            "temperatures": temperatures,
            "abstention_thresholds": abstention_thresholds,
            "selected_on": "validation",
        },
        "test_results": results,
        "ablations": {
            "reference_vs_asr_text_validation": reference_ablation,
            "acoustic_feature_groups_validation": acoustic_ablations,
        },
        "paired_text_vs_late_fusion": comparison,
        "paired_text_vs_concatenated_fusion": concatenated_comparison,
        "runtime": {
            "total_seconds": elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "environment": _environment(),
        "limitations": [
            "Synthetic English eSpeak voices do not represent Filipino speakers.",
            "The sample has only four voice variants and one held-out test voice.",
            "Acoustic summary features are engineering baselines, not emotion measurements.",
            "The WavLM cache adapter is implemented but learned embeddings are not reported here.",
            "Confidence intervals describe this synthetic sample only.",
        ],
    }


def _select_text_model(train, validation, seed):
    def build(c):
        return TfidfIntentBaseline(c=c, seed=seed).fit(train)

    return _select_model(build, lambda model: model.predict(validation), validation)


def _select_acoustic_model(train, validation, train_features, validation_features, seed):
    def build(c):
        return AcousticIntentBaseline(c=c, seed=seed).fit(train_features, train)

    return _select_model(
        build,
        lambda model: model.predict(validation_features, validation),
        validation,
    )


def _select_concat_model(train, validation, train_features, validation_features, seed):
    def build(c):
        return ConcatenatedIntentBaseline(c=c, seed=seed).fit(train, train_features)

    return _select_model(
        build,
        lambda model: model.predict(validation, validation_features),
        validation,
    )


def _select_model(
    builder: Callable[[float], Any],
    predictor: Callable[[Any], PredictionSet],
    validation: list[ResearchCase],
):
    candidates = []
    for c in (0.1, 1.0, 10.0):
        model = builder(c)
        predictions = predictor(model)
        score = evaluate_predictions(predictions, validation)["macro_f1"]
        candidates.append((score, -c, model, c, predictions))
    _, _, model, c, predictions = max(candidates, key=lambda value: value[:2])
    return model, c, predictions


def _select_fusion_weight(
    text: PredictionSet,
    acoustic: PredictionSet,
    validation: list[ResearchCase],
) -> float:
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    return max(
        candidates,
        key=lambda weight: (
            evaluate_predictions(late_fuse(text, acoustic, weight), validation)["macro_f1"],
            -abs(weight - 0.5),
        ),
    )


def _reference_text_ablation(train, validation, test, c, seed):
    asr_model = TfidfIntentBaseline(c=c, seed=seed).fit(train, transcript_source="asr")
    reference_model = TfidfIntentBaseline(c=c, seed=seed).fit(train, transcript_source="reference")
    return {
        "asr_validation_macro_f1": evaluate_predictions(
            asr_model.predict(validation, transcript_source="asr"), validation
        )["macro_f1"],
        "reference_validation_macro_f1": evaluate_predictions(
            reference_model.predict(validation, transcript_source="reference"), validation
        )["macro_f1"],
        "test_not_used_for_selection": len(test),
    }


def _acoustic_ablations(train, validation, train_features, validation_features, c, seed):
    groups = {
        "all": tuple(range(8)),
        "without_timing": tuple(range(1, 8)),
        "without_energy": (0, 4),
        "without_frame_statistics": (0, 1, 2, 3, 4),
    }
    output = {}
    for name, indexes in groups.items():
        selected_train = [tuple(row[index] for index in indexes) for row in train_features]
        selected_validation = [
            tuple(row[index] for index in indexes) for row in validation_features
        ]
        model = AcousticIntentBaseline(c=c, seed=seed).fit(selected_train, train)
        output[name] = evaluate_predictions(
            model.predict(selected_validation, validation), validation
        )["macro_f1"]
    return output


def _paired_accuracy_bootstrap(
    baseline: PredictionSet,
    candidate: PredictionSet,
    *,
    seed: int,
    samples: int = 2_000,
) -> dict[str, Any]:
    differences = [
        int(candidate_value == label) - int(baseline_value == label)
        for label, baseline_value, candidate_value in zip(
            baseline.labels,
            baseline.predictions,
            candidate.predictions,
            strict=True,
        )
    ]
    generator = random.Random(seed)
    estimates = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return {
        "accuracy_delta": sum(differences) / len(differences),
        "bootstrap_95_ci": [
            estimates[round(0.025 * (samples - 1))],
            estimates[round(0.975 * (samples - 1))],
        ],
        "seed": seed,
        "samples": samples,
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for package in ("finvoice-ai", "numpy", "scikit-learn", "scipy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multimodal intent baselines")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--development-asr-report", type=Path, required=True)
    parser.add_argument("--test-asr-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--abstention-minimum-accuracy", type=float, default=0.8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        experiment_id="multimodal-intent-baseline-v1",
        dataset_version="1.0.0",
        manifest_sha256=manifest_sha256(args.manifest),
        seed=args.seed,
        abstention_minimum_accuracy=args.abstention_minimum_accuracy,
    )
    report = run_experiment(
        args.manifest,
        args.dataset_root,
        args.development_asr_report,
        args.test_asr_report,
        config,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
