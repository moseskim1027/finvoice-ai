from finvoice_ai.speech.models import AudioBuffer, SpeechSegment, TranscriptionResult


class DeterministicTranscriptionProvider:
    """Explicit offline stub used to exercise the speech contract without an ASR model."""

    model_name = "deterministic-asr-stub-v1"

    def __init__(self, transcript: str = "demo speech detected", language: str = "und") -> None:
        self.transcript = transcript
        self.language = language

    def transcribe(
        self,
        audio: AudioBuffer,
        segments: list[SpeechSegment],
    ) -> TranscriptionResult:
        del audio
        if not segments:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=self.language,
                model=self.model_name,
            )
        return TranscriptionResult(
            text=self.transcript,
            confidence=1.0,
            language=self.language,
            model=self.model_name,
        )
