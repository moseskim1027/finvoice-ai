from typing import Protocol

from finvoice_ai.speech.models import AudioBuffer, TranscriptionResult


class StreamingTranscriptionProvider(Protocol):
    """A replaceable partial/final transcription boundary.

    Implementations may re-run offline ASR over the accumulated audio. The
    protocol does not imply token-level model streaming.
    """

    def transcribe(self, audio: AudioBuffer, *, is_final: bool) -> TranscriptionResult: ...


class StreamingMetrics(Protocol):
    def observe(self, name: str, value: float, attributes: dict[str, str]) -> None: ...
