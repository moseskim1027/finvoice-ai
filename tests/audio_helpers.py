import math
import struct
import wave
from io import BytesIO


def create_wav(
    *,
    sample_rate_hz: int = 16_000,
    duration_seconds: float = 0.1,
    amplitude: int = 4_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    frame_count = round(sample_rate_hz * duration_seconds)
    mono_samples = [
        round(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate_hz))
        for index in range(frame_count)
    ]
    samples = [sample for sample in mono_samples for _ in range(channels)]

    if sample_width == 2:
        frames = struct.pack(f"<{len(samples)}h", *samples)
    else:
        frames = bytes([128] * len(samples))

    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(frames)
    return output.getvalue()
