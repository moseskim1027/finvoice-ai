from dataclasses import dataclass


@dataclass(frozen=True)
class AudioBuffer:
    sample_rate_hz: int
    samples: tuple[int, ...]

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate_hz


@dataclass(frozen=True)
class SpeechSegment:
    start_seconds: float
    end_seconds: float
    mean_rms: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds
