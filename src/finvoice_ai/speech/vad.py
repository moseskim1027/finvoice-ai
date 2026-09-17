import math
from dataclasses import dataclass

from finvoice_ai.speech.models import AudioBuffer, SpeechSegment


@dataclass(frozen=True)
class EnergyVadConfig:
    frame_duration_ms: int = 20
    rms_threshold: float = 500.0
    minimum_speech_ms: int = 60
    maximum_silence_ms: int = 100


class EnergyVoiceActivityDetector:
    """Deterministic energy baseline for segmenting likely speech regions."""

    def __init__(self, config: EnergyVadConfig | None = None) -> None:
        self.config = config or EnergyVadConfig()

    def detect(self, audio: AudioBuffer) -> list[SpeechSegment]:
        frame_size = max(1, round(audio.sample_rate_hz * self.config.frame_duration_ms / 1_000))
        frames = [
            audio.samples[start : start + frame_size]
            for start in range(0, len(audio.samples), frame_size)
        ]
        frame_rms = [self._rms(frame) for frame in frames]

        segments: list[SpeechSegment] = []
        start_frame: int | None = None
        last_voiced_frame: int | None = None
        voiced_rms: list[float] = []
        allowed_silent_frames = self.config.maximum_silence_ms // self.config.frame_duration_ms

        for index, rms in enumerate(frame_rms):
            if rms >= self.config.rms_threshold:
                if start_frame is None:
                    start_frame = index
                last_voiced_frame = index
                voiced_rms.append(rms)
            elif (
                start_frame is not None
                and last_voiced_frame is not None
                and index - last_voiced_frame > allowed_silent_frames
            ):
                self._append_segment(
                    segments,
                    start_frame,
                    last_voiced_frame,
                    frame_size,
                    audio.sample_rate_hz,
                    voiced_rms,
                )
                start_frame = None
                last_voiced_frame = None
                voiced_rms = []

        if start_frame is not None and last_voiced_frame is not None:
            self._append_segment(
                segments,
                start_frame,
                last_voiced_frame,
                frame_size,
                audio.sample_rate_hz,
                voiced_rms,
            )
        return segments

    def _append_segment(
        self,
        segments: list[SpeechSegment],
        start_frame: int,
        end_frame: int,
        frame_size: int,
        sample_rate_hz: int,
        voiced_rms: list[float],
    ) -> None:
        start_seconds = start_frame * frame_size / sample_rate_hz
        end_seconds = (end_frame + 1) * frame_size / sample_rate_hz
        if (end_seconds - start_seconds) * 1_000 < self.config.minimum_speech_ms:
            return
        segments.append(
            SpeechSegment(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                mean_rms=sum(voiced_rms) / len(voiced_rms),
            )
        )

    @staticmethod
    def _rms(samples: tuple[int, ...]) -> float:
        if not samples:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples))
