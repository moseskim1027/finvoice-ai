from finvoice_ai.speech.models import AudioBuffer, TranscriptionResult
from finvoice_ai.speech.ports import TranscriptionProvider
from finvoice_ai.speech.vad import EnergyVoiceActivityDetector


class OfflineStreamingTranscriptionAdapter:
    """Simulate partials by re-running an offline provider over buffered audio.

    This intentionally offers deterministic streaming mechanics rather than
    claiming genuine incremental decoding or token streaming.
    """

    def __init__(
        self,
        provider: TranscriptionProvider,
        vad: EnergyVoiceActivityDetector | None = None,
    ) -> None:
        self._provider = provider
        self._vad = vad or EnergyVoiceActivityDetector()

    def transcribe(self, audio: AudioBuffer, *, is_final: bool) -> TranscriptionResult:
        del is_final
        return self._provider.transcribe(audio, self._vad.detect(audio))
