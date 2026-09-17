import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from finvoice_ai.config import Settings
from finvoice_ai.speech.models import AudioBuffer, SpeechSegment, TranscriptionResult
from finvoice_ai.speech.preprocessing import ASR_SAMPLE_RATE_HZ, prepare_speech_samples
from finvoice_ai.speech.providers import DeterministicTranscriptionProvider


class AsrDependencyError(RuntimeError):
    """Raised when an explicitly selected ASR provider is unavailable."""


@dataclass(frozen=True)
class WhisperConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5


@dataclass(frozen=True)
class BackendSegment:
    text: str
    start_seconds: float
    end_seconds: float
    average_log_probability: float


@dataclass(frozen=True)
class BackendTranscription:
    segments: tuple[BackendSegment, ...]
    language: str


class WhisperBackend(Protocol):
    def transcribe(
        self,
        samples: Sequence[float],
        *,
        language: str | None,
        beam_size: int,
    ) -> BackendTranscription: ...


class FasterWhisperBackend:
    """Thin lazy boundary around the optional faster-whisper dependency."""

    def __init__(self, config: WhisperConfig) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise AsrDependencyError(
                "faster-whisper is not installed; install the project with the 'asr' extra"
            ) from error

        self._model: Any = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe(
        self,
        samples: Sequence[float],
        *,
        language: str | None,
        beam_size: int,
    ) -> BackendTranscription:
        try:
            import numpy as np
        except ImportError as error:
            raise AsrDependencyError("faster-whisper requires NumPy") from error

        raw_segments, info = self._model.transcribe(
            np.asarray(samples, dtype="float32"),
            language=language,
            beam_size=beam_size,
        )
        segments = tuple(self._convert_segments(raw_segments))
        return BackendTranscription(segments=segments, language=info.language or language or "und")

    @staticmethod
    def _convert_segments(raw_segments: Iterable[Any]) -> Iterable[BackendSegment]:
        for segment in raw_segments:
            yield BackendSegment(
                text=segment.text.strip(),
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                average_log_probability=float(segment.avg_logprob),
            )


class FasterWhisperTranscriptionProvider:
    def __init__(self, config: WhisperConfig, backend: WhisperBackend | None = None) -> None:
        self._config = config
        self._backend = backend or FasterWhisperBackend(config)
        self.model_name = f"faster-whisper/{config.model_size}"

    def transcribe(
        self,
        audio: AudioBuffer,
        segments: list[SpeechSegment],
    ) -> TranscriptionResult:
        samples = prepare_speech_samples(audio, segments, ASR_SAMPLE_RATE_HZ)
        if not samples:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=self._config.language or "und",
                model=self.model_name,
            )

        result = self._backend.transcribe(
            samples,
            language=self._config.language,
            beam_size=self._config.beam_size,
        )
        return TranscriptionResult(
            text=" ".join(segment.text for segment in result.segments if segment.text).strip(),
            confidence=_duration_weighted_confidence(result.segments),
            language=result.language,
            model=self.model_name,
        )


def _duration_weighted_confidence(segments: Sequence[BackendSegment]) -> float:
    weights = [max(segment.end_seconds - segment.start_seconds, 0.0) for segment in segments]
    denominator = sum(weights)
    if denominator <= 0.0:
        return 0.0
    weighted = sum(
        math.exp(min(segment.average_log_probability, 0.0)) * weight
        for segment, weight in zip(segments, weights, strict=True)
    )
    return min(max(weighted / denominator, 0.0), 1.0)


def build_transcription_provider(settings: Settings):
    if settings.transcription_provider == "deterministic":
        return DeterministicTranscriptionProvider()
    config = WhisperConfig(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        beam_size=settings.whisper_beam_size,
    )
    return FasterWhisperTranscriptionProvider(config)
