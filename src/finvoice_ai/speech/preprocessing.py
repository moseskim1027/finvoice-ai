from collections.abc import Sequence

from finvoice_ai.speech.models import AudioBuffer, SpeechSegment

ASR_SAMPLE_RATE_HZ = 16_000
INT16_SCALE = 32_768.0


def prepare_speech_samples(
    audio: AudioBuffer,
    segments: Sequence[SpeechSegment],
    target_rate_hz: int = ASR_SAMPLE_RATE_HZ,
) -> list[float]:
    """Select voiced samples, normalize PCM16, and resample for ASR."""
    selected: list[int] = []
    for segment in segments:
        start = max(0, round(segment.start_seconds * audio.sample_rate_hz))
        end = min(len(audio.samples), round(segment.end_seconds * audio.sample_rate_hz))
        selected.extend(audio.samples[start:end])

    normalized = [sample / INT16_SCALE for sample in selected]
    return resample_linear(normalized, audio.sample_rate_hz, target_rate_hz)


def resample_linear(
    samples: Sequence[float],
    source_rate_hz: int,
    target_rate_hz: int,
) -> list[float]:
    """Deterministic linear interpolation suitable for the bounded demo input."""
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    if not samples or source_rate_hz == target_rate_hz:
        return list(samples)

    output_length = round(len(samples) * target_rate_hz / source_rate_hz)
    if output_length <= 1 or len(samples) == 1:
        return [float(samples[0])] * output_length

    scale = source_rate_hz / target_rate_hz
    last_index = len(samples) - 1
    output: list[float] = []
    for output_index in range(output_length):
        source_position = min(output_index * scale, last_index)
        left = int(source_position)
        right = min(left + 1, last_index)
        fraction = source_position - left
        output.append(float(samples[left] * (1.0 - fraction) + samples[right] * fraction))
    return output
