from finvoice_ai.speech.models import SpeechAnalysis
from finvoice_ai.speech.ports import TranscriptionProvider
from finvoice_ai.speech.vad import EnergyVoiceActivityDetector
from finvoice_ai.speech.wav import PcmWavLoader


class SpeechService:
    """Compose audio validation, segmentation, and a replaceable ASR provider."""

    def __init__(
        self,
        loader: PcmWavLoader,
        vad: EnergyVoiceActivityDetector,
        transcriber: TranscriptionProvider,
    ) -> None:
        self._loader = loader
        self._vad = vad
        self._transcriber = transcriber

    def analyze(self, wav_data: bytes) -> SpeechAnalysis:
        audio = self._loader.load(wav_data)
        segments = self._vad.detect(audio)
        transcription = self._transcriber.transcribe(audio, segments)
        return SpeechAnalysis(
            duration_seconds=audio.duration_seconds,
            segments=tuple(segments),
            transcription=transcription,
        )
