import math
from typing import Any

from finvoice_ai.research.contracts import INTENT_LABELS, ResearchCase
from finvoice_ai.research.text_baseline import PredictionSet


def evaluate_predictions(
    predictions: PredictionSet,
    cases: list[ResearchCase],
    *,
    abstention_threshold: float = 0.0,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    confidences = [max(row) for row in predictions.probabilities]
    accepted = [confidence >= abstention_threshold for confidence in confidences]
    accepted_indexes = [index for index, value in enumerate(accepted) if value]
    labels = list(predictions.labels)
    predicted = list(predictions.predictions)
    accepted_labels = [labels[index] for index in accepted_indexes]
    accepted_predictions = [predicted[index] for index in accepted_indexes]
    if accepted_indexes:
        macro_f1 = f1_score(accepted_labels, accepted_predictions, average="macro", zero_division=0)
        balanced_accuracy = _balanced_accuracy(accepted_labels, accepted_predictions)
        accuracy = accuracy_score(accepted_labels, accepted_predictions)
    else:
        macro_f1 = balanced_accuracy = accuracy = 0.0
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        labels=list(predictions.classes),
        zero_division=0,
    )
    escalation_labels = [case.intent.escalation_required for case in cases]
    escalation_predictions = [INTENT_LABELS[value].escalation_required for value in predicted]
    escalation_precision, escalation_recall, _, _ = precision_recall_fscore_support(
        escalation_labels,
        escalation_predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "case_count": len(cases),
        "coverage": len(accepted_indexes) / max(1, len(cases)),
        "accuracy": float(accuracy),
        "selective_risk": float(1.0 - accuracy) if accepted_indexes else 0.0,
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "brier_score": _multiclass_brier(predictions),
        "expected_calibration_error": _ece(predictions),
        "escalation_precision": float(escalation_precision),
        "escalation_recall": float(escalation_recall),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=list(predictions.classes)
        ).tolist(),
        "classes": list(predictions.classes),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(predictions.classes)
        },
    }


def risk_coverage_curve(
    predictions: PredictionSet, cases: list[ResearchCase]
) -> list[dict[str, float]]:
    return [
        {
            "threshold": threshold,
            **{
                key: value
                for key, value in evaluate_predictions(
                    predictions, cases, abstention_threshold=threshold
                ).items()
                if key in {"coverage", "accuracy", "selective_risk"}
            },
        }
        for threshold in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ]


def slice_metrics(
    predictions: PredictionSet,
    cases: list[ResearchCase],
) -> list[dict[str, Any]]:
    output = []
    for dimension in ("language_mode", "noise_condition", "device", "speaker_id"):
        for value in sorted({getattr(case, dimension) for case in cases}):
            indexes = [
                index for index, case in enumerate(cases) if getattr(case, dimension) == value
            ]
            selected_cases = [cases[index] for index in indexes]
            selected_predictions = _select(predictions, indexes)
            metrics = evaluate_predictions(selected_predictions, selected_cases)
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "case_count": len(indexes),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
    return output


def temperature_scale(predictions: PredictionSet, temperature: float) -> PredictionSet:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    rows = []
    for probabilities in predictions.probabilities:
        logits = [math.log(max(value, 1e-12)) / temperature for value in probabilities]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        denominator = sum(exponentials)
        rows.append(tuple(value / denominator for value in exponentials))
    predicted = tuple(
        predictions.classes[max(range(len(row)), key=row.__getitem__)] for row in rows
    )
    return PredictionSet(
        case_ids=predictions.case_ids,
        labels=predictions.labels,
        predictions=predicted,
        probabilities=tuple(rows),
        classes=predictions.classes,
    )


def select_temperature(predictions: PredictionSet) -> float:
    candidates = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    return min(
        candidates,
        key=lambda value: _negative_log_likelihood(temperature_scale(predictions, value)),
    )


def select_abstention_threshold(
    predictions: PredictionSet,
    cases: list[ResearchCase],
    *,
    minimum_accuracy: float = 0.8,
) -> float:
    candidates = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    eligible = []
    for threshold in candidates:
        metrics = evaluate_predictions(
            predictions,
            cases,
            abstention_threshold=threshold,
        )
        if metrics["coverage"] > 0.0 and metrics["accuracy"] >= minimum_accuracy:
            eligible.append((metrics["coverage"], -threshold, threshold))
    return max(eligible)[2] if eligible else 0.0


def _negative_log_likelihood(predictions: PredictionSet) -> float:
    indexes = {label: index for index, label in enumerate(predictions.classes)}
    return -sum(
        math.log(max(row[indexes[label]], 1e-12))
        for label, row in zip(predictions.labels, predictions.probabilities, strict=True)
    ) / max(1, len(predictions.labels))


def _multiclass_brier(predictions: PredictionSet) -> float:
    indexes = {label: index for index, label in enumerate(predictions.classes)}
    return sum(
        sum(
            (probability - float(index == indexes[label])) ** 2
            for index, probability in enumerate(row)
        )
        for label, row in zip(predictions.labels, predictions.probabilities, strict=True)
    ) / max(1, len(predictions.labels))


def _ece(predictions: PredictionSet, bins: int = 10) -> float:
    confidences = [max(row) for row in predictions.probabilities]
    correct = [
        expected == predicted
        for expected, predicted in zip(predictions.labels, predictions.predictions, strict=True)
    ]
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            item
            for item, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if selected:
            accuracy = sum(correct[item] for item in selected) / len(selected)
            confidence = sum(confidences[item] for item in selected) / len(selected)
            error += len(selected) / len(confidences) * abs(accuracy - confidence)
    return error


def _select(predictions: PredictionSet, indexes: list[int]) -> PredictionSet:
    return PredictionSet(
        case_ids=tuple(predictions.case_ids[index] for index in indexes),
        labels=tuple(predictions.labels[index] for index in indexes),
        predictions=tuple(predictions.predictions[index] for index in indexes),
        probabilities=tuple(predictions.probabilities[index] for index in indexes),
        classes=predictions.classes,
    )


def _balanced_accuracy(labels: list[str], predictions: list[str]) -> float:
    observed = sorted(set(labels))
    recalls = []
    for label in observed:
        indexes = [index for index, value in enumerate(labels) if value == label]
        recalls.append(sum(predictions[index] == label for index in indexes) / len(indexes))
    return sum(recalls) / len(recalls) if recalls else 0.0
