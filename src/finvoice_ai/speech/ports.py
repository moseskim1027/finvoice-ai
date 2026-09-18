from typing import Protocol

from finvoice_ai.speech.models import AudioBuffer, SpeechSegment, TranscriptionResult


class TranscriptionUnavailableError(RuntimeError):
    """Raised when an ASR provider times out or is unavailable."""


class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        audio: AudioBuffer,
        segments: list[SpeechSegment],
    ) -> TranscriptionResult: ...
