import pytest

from finvoice_ai.speech.models import AudioBuffer, SpeechSegment
from finvoice_ai.speech.preprocessing import prepare_speech_samples, resample_linear


def test_preparation_selects_segments_and_normalizes_pcm16() -> None:
    audio = AudioBuffer(sample_rate_hz=16_000, samples=(0, 16_384, -32_768, 32_767))

    samples = prepare_speech_samples(
        audio,
        [SpeechSegment(1 / 16_000, 3 / 16_000, 0.5)],
    )

    assert samples == [0.5, -1.0]


def test_linear_resampling_interpolates_and_validates_rates() -> None:
    assert resample_linear([0.0, 1.0], 2, 4) == [0.0, 0.5, 1.0, 1.0]
    assert resample_linear([], 8_000, 16_000) == []
    assert resample_linear([0.25], 8_000, 16_000) == [0.25, 0.25]

    with pytest.raises(ValueError, match="sample rates must be positive"):
        resample_linear([0.0], 0, 16_000)
