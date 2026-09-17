import pytest

from finvoice_ai.speech.models import AudioBuffer
from finvoice_ai.speech.vad import EnergyVadConfig, EnergyVoiceActivityDetector

SAMPLE_RATE = 1_000


def audio(*regions: tuple[int, int]) -> AudioBuffer:
    samples = tuple(
        sample
        for duration_ms, amplitude in regions
        for sample in [amplitude] * round(SAMPLE_RATE * duration_ms / 1_000)
    )
    return AudioBuffer(sample_rate_hz=SAMPLE_RATE, samples=samples)


def detector(**overrides: int | float) -> EnergyVoiceActivityDetector:
    return EnergyVoiceActivityDetector(EnergyVadConfig(**overrides))


def test_vad_detects_single_speech_region() -> None:
    segments = detector().detect(audio((100, 0), (200, 2_000), (100, 0)))

    assert len(segments) == 1
    assert segments[0].start_seconds == pytest.approx(0.1)
    assert segments[0].end_seconds == pytest.approx(0.3)
    assert segments[0].mean_rms == pytest.approx(2_000)


def test_vad_bridges_short_silence() -> None:
    segments = detector(maximum_silence_ms=100).detect(audio((100, 2_000), (60, 0), (100, 2_000)))

    assert len(segments) == 1
    assert segments[0].duration_seconds == pytest.approx(0.26)


def test_vad_splits_long_silence() -> None:
    segments = detector(maximum_silence_ms=40).detect(audio((100, 2_000), (100, 0), (100, 2_000)))

    assert len(segments) == 2


def test_vad_drops_short_burst() -> None:
    segments = detector(minimum_speech_ms=80).detect(audio((40, 2_000), (100, 0)))

    assert segments == []


def test_vad_returns_no_segments_for_silence() -> None:
    assert detector().detect(audio((200, 0))) == []
