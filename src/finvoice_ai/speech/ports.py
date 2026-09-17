from typing import Protocol

from finvoice_ai.speech.models import AudioBuffer, SpeechSegment, TranscriptionResult


class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        audio: AudioBuffer,
        segments: list[SpeechSegment],
    ) -> TranscriptionResult: ...
