import math
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class SpeechCaseResult(BaseModel):
    case_id: str
    language: str
    language_mode: str
    noise_condition: str
    device: str
    speaker_id: str
    intent_id: str
    utterance_type: str
    status: Literal["success", "failed"]
    hypothesis: str
    detected_language: str
    model: str
    confidence: float = Field(ge=0.0, le=1.0)
    word_errors: int = Field(ge=0)
    reference_words: int = Field(ge=0)
    character_errors: int = Field(ge=0)
    reference_characters: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    audio_duration_seconds: float = Field(ge=0.0)
    error_type: str | None = None

    @computed_field
    @property
    def word_error_rate(self) -> float:
        return self.word_errors / max(1, self.reference_words)

    @computed_field
    @property
    def character_error_rate(self) -> float:
        return self.character_errors / max(1, self.reference_characters)

    @computed_field
    @property
    def real_time_factor(self) -> float:
        if self.audio_duration_seconds <= 0.0:
            return 0.0
        return self.latency_ms / 1000.0 / self.audio_duration_seconds


class SpeechMetricSummary(BaseModel):
    case_count: int
    failure_count: int
    failure_rate: float
    word_error_rate: float
    character_error_rate: float
    macro_word_error_rate: float
    macro_character_error_rate: float
    language_identification_accuracy: float
    expected_calibration_error: float
    latency_p50_ms: float
    latency_p95_ms: float
    mean_real_time_factor: float


class SpeechSliceSummary(BaseModel):
    dimension: str
    value: str
    metrics: SpeechMetricSummary


class SpeechEvaluationReport(BaseModel):
    dataset_name: str
    dataset_version: str
    evaluation_split: str
    overall: SpeechMetricSummary
    slices: list[SpeechSliceSummary]
    cases: list[SpeechCaseResult]


def summarize_results(results: Iterable[SpeechCaseResult]) -> SpeechMetricSummary:
    cases = list(results)
    if not cases:
        return SpeechMetricSummary(
            case_count=0,
            failure_count=0,
            failure_rate=0.0,
            word_error_rate=0.0,
            character_error_rate=0.0,
            macro_word_error_rate=0.0,
            macro_character_error_rate=0.0,
            language_identification_accuracy=0.0,
            expected_calibration_error=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            mean_real_time_factor=0.0,
        )

    failure_count = sum(case.status == "failed" for case in cases)
    return SpeechMetricSummary(
        case_count=len(cases),
        failure_count=failure_count,
        failure_rate=failure_count / len(cases),
        word_error_rate=sum(case.word_errors for case in cases)
        / max(1, sum(case.reference_words for case in cases)),
        character_error_rate=sum(case.character_errors for case in cases)
        / max(1, sum(case.reference_characters for case in cases)),
        macro_word_error_rate=sum(case.word_error_rate for case in cases) / len(cases),
        macro_character_error_rate=sum(case.character_error_rate for case in cases) / len(cases),
        language_identification_accuracy=sum(_language_matches(case) for case in cases)
        / len(cases),
        expected_calibration_error=_expected_calibration_error(cases),
        latency_p50_ms=_percentile([case.latency_ms for case in cases], 0.50),
        latency_p95_ms=_percentile([case.latency_ms for case in cases], 0.95),
        mean_real_time_factor=sum(case.real_time_factor for case in cases) / len(cases),
    )


def build_slice_summaries(results: Iterable[SpeechCaseResult]) -> list[SpeechSliceSummary]:
    cases = list(results)
    dimensions = (
        "language",
        "language_mode",
        "noise_condition",
        "device",
        "speaker_id",
        "intent_id",
        "utterance_type",
    )
    summaries: list[SpeechSliceSummary] = []
    for dimension in dimensions:
        values = sorted({str(getattr(case, dimension)) for case in cases})
        for value in values:
            selected = [case for case in cases if str(getattr(case, dimension)) == value]
            summaries.append(
                SpeechSliceSummary(
                    dimension=dimension,
                    value=value,
                    metrics=summarize_results(selected),
                )
            )
    return summaries


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _language_matches(case: SpeechCaseResult) -> bool:
    expected = set(case.language.split("-"))
    return case.detected_language.casefold() in expected


def _expected_calibration_error(cases: list[SpeechCaseResult], bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            case
            for case in cases
            if lower <= case.confidence < upper or (index == bins - 1 and case.confidence == 1.0)
        ]
        if not selected:
            continue
        accuracy = sum(case.word_errors == 0 for case in selected) / len(selected)
        mean_confidence = sum(case.confidence for case in selected) / len(selected)
        error += len(selected) / len(cases) * abs(accuracy - mean_confidence)
    return error
