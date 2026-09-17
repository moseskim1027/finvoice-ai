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


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    confidence: float
    language: str
    model: str


@dataclass(frozen=True)
class SpeechAnalysis:
    duration_seconds: float
    segments: tuple[SpeechSegment, ...]
    transcription: TranscriptionResult
