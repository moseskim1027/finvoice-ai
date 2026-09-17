import math
from dataclasses import dataclass

import pytest

from finvoice_ai.config import Settings
from finvoice_ai.speech.asr import (
    BackendSegment,
    BackendTranscription,
    FasterWhisperTranscriptionProvider,
    WhisperConfig,
    build_transcription_provider,
)
from finvoice_ai.speech.models import AudioBuffer, SpeechSegment
from finvoice_ai.speech.providers import DeterministicTranscriptionProvider


@dataclass
class RecordingBackend:
    result: BackendTranscription
    samples: list[float] | None = None
    language: str | None = None
    beam_size: int | None = None

    def transcribe(self, samples, *, language, beam_size) -> BackendTranscription:
        self.samples = list(samples)
        self.language = language
        self.beam_size = beam_size
        return self.result


def test_provider_transcribes_only_voiced_audio_and_aggregates_confidence() -> None:
    backend = RecordingBackend(
        BackendTranscription(
            segments=(
                BackendSegment("reset my", 0.0, 0.25, math.log(0.8)),
                BackendSegment("pin", 0.25, 0.75, math.log(0.5)),
            ),
            language="en",
        )
    )
    provider = FasterWhisperTranscriptionProvider(
        WhisperConfig(model_size="tiny", language="en", beam_size=3),
        backend=backend,
    )
    audio = AudioBuffer(sample_rate_hz=8_000, samples=(0,) * 800 + (16_384,) * 800)
    segments = [SpeechSegment(0.1, 0.2, 0.5)]

    result = provider.transcribe(audio, segments)

    assert result.text == "reset my pin"
    assert result.confidence == pytest.approx(0.6)
    assert result.language == "en"
    assert result.model == "faster-whisper/tiny"
    assert backend.language == "en"
    assert backend.beam_size == 3
    assert backend.samples is not None
    assert len(backend.samples) == 1_600
    assert set(backend.samples) == {0.5}


def test_provider_abstains_without_speech() -> None:
    backend = RecordingBackend(BackendTranscription(segments=(), language="en"))
    provider = FasterWhisperTranscriptionProvider(WhisperConfig(language="fil"), backend)

    result = provider.transcribe(AudioBuffer(16_000, (1, 2)), [])

    assert result.text == ""
    assert result.confidence == 0.0
    assert result.language == "fil"
    assert backend.samples is None


def test_provider_returns_zero_confidence_for_zero_duration_output() -> None:
    backend = RecordingBackend(
        BackendTranscription(
            segments=(BackendSegment("hello", 0.0, 0.0, -0.1),),
            language="en",
        )
    )
    provider = FasterWhisperTranscriptionProvider(WhisperConfig(), backend)

    result = provider.transcribe(
        AudioBuffer(16_000, (100,) * 10),
        [SpeechSegment(0.0, 0.001, 0.1)],
    )

    assert result.confidence == 0.0


def test_factory_preserves_lightweight_default() -> None:
    provider = build_transcription_provider(Settings(_env_file=None))

    assert isinstance(provider, DeterministicTranscriptionProvider)
