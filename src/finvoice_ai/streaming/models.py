from dataclasses import dataclass
from typing import Literal, TypeAlias

from finvoice_ai.speech.models import AudioBuffer, TranscriptionResult


@dataclass(frozen=True)
class PcmChunk:
    session_id: str
    utterance_id: str
    sequence: int
    sample_rate_hz: int
    samples: tuple[int, ...]

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate_hz

    def as_audio(self) -> AudioBuffer:
        return AudioBuffer(sample_rate_hz=self.sample_rate_hz, samples=self.samples)


@dataclass(frozen=True)
class AcceptedEvent:
    session_id: str
    utterance_id: str
    sequence: int
    duplicate: bool = False
    type: Literal["accepted"] = "accepted"


@dataclass(frozen=True)
class SpeechStartedEvent:
    session_id: str
    utterance_id: str
    detected_at: float
    type: Literal["speech_started"] = "speech_started"


@dataclass(frozen=True)
class PartialEvent:
    session_id: str
    utterance_id: str
    transcription: TranscriptionResult
    emitted_at: float
    type: Literal["partial"] = "partial"


@dataclass(frozen=True)
class FinalizedEvent:
    session_id: str
    utterance_id: str
    transcription: TranscriptionResult
    reason: Literal["trailing_silence", "maximum_duration"]
    audio_duration_seconds: float
    finalized_at: float
    type: Literal["finalized"] = "finalized"


@dataclass(frozen=True)
class InterruptedEvent:
    session_id: str
    utterance_id: str
    response_id: str
    interrupted_at: float
    type: Literal["interrupted"] = "interrupted"


@dataclass(frozen=True)
class ErrorEvent:
    session_id: str
    utterance_id: str
    code: str
    message: str
    sequence: int | None = None
    type: Literal["error"] = "error"


StreamEvent: TypeAlias = (
    AcceptedEvent
    | SpeechStartedEvent
    | PartialEvent
    | FinalizedEvent
    | InterruptedEvent
    | ErrorEvent
)
