import struct
import wave
from io import BytesIO

from finvoice_ai.speech.models import AudioBuffer


class InvalidAudioError(ValueError):
    """Raised when audio violates the supported deterministic WAV contract."""


class PcmWavLoader:
    SUPPORTED_SAMPLE_RATES = frozenset({8_000, 16_000, 24_000, 48_000})

    def __init__(self, maximum_duration_seconds: float = 60.0) -> None:
        self.maximum_duration_seconds = maximum_duration_seconds

    def load(self, data: bytes) -> AudioBuffer:
        try:
            with wave.open(BytesIO(data), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                compression = wav_file.getcomptype()
                frames = wav_file.readframes(frame_count)
        except (EOFError, wave.Error) as error:
            raise InvalidAudioError("invalid WAV container") from error

        if channels != 1:
            raise InvalidAudioError("audio must be mono")
        if sample_width != 2:
            raise InvalidAudioError("audio must use signed 16-bit PCM")
        if compression != "NONE":
            raise InvalidAudioError("compressed WAV audio is not supported")
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise InvalidAudioError("unsupported sample rate")

        duration_seconds = frame_count / sample_rate
        if duration_seconds > self.maximum_duration_seconds:
            raise InvalidAudioError("audio exceeds maximum duration")

        samples = struct.unpack(f"<{frame_count}h", frames) if frame_count else ()
        return AudioBuffer(sample_rate_hz=sample_rate, samples=samples)
