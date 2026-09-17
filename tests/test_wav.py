import pytest

from finvoice_ai.speech.wav import InvalidAudioError, PcmWavLoader
from tests.audio_helpers import create_wav


def test_loader_reads_supported_pcm_wav() -> None:
    audio = PcmWavLoader().load(create_wav(duration_seconds=0.25))

    assert audio.sample_rate_hz == 16_000
    assert audio.duration_seconds == pytest.approx(0.25)
    assert len(audio.samples) == 4_000


@pytest.mark.parametrize(
    ("wav_data", "message"),
    [
        (b"not-a-wave", "invalid WAV"),
        (create_wav(channels=2), "mono"),
        (create_wav(sample_width=1), "16-bit"),
        (create_wav(sample_rate_hz=44_100), "sample rate"),
    ],
)
def test_loader_rejects_unsupported_audio(wav_data: bytes, message: str) -> None:
    with pytest.raises(InvalidAudioError, match=message):
        PcmWavLoader().load(wav_data)


def test_loader_enforces_duration_limit() -> None:
    with pytest.raises(InvalidAudioError, match="maximum duration"):
        PcmWavLoader(maximum_duration_seconds=0.05).load(create_wav(duration_seconds=0.1))


def test_loader_accepts_empty_wav() -> None:
    audio = PcmWavLoader().load(create_wav(duration_seconds=0.0))

    assert audio.samples == ()
    assert audio.duration_seconds == 0.0
