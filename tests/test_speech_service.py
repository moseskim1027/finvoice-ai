from finvoice_ai.speech.providers import DeterministicTranscriptionProvider
from finvoice_ai.speech.service import SpeechService
from finvoice_ai.speech.vad import EnergyVadConfig, EnergyVoiceActivityDetector
from finvoice_ai.speech.wav import PcmWavLoader
from tests.audio_helpers import create_wav


def build_service() -> SpeechService:
    return SpeechService(
        loader=PcmWavLoader(),
        vad=EnergyVoiceActivityDetector(EnergyVadConfig(rms_threshold=500, minimum_speech_ms=40)),
        transcriber=DeterministicTranscriptionProvider(
            transcript="reset my pin",
            language="en",
        ),
    )


def test_speech_service_composes_pipeline() -> None:
    analysis = build_service().analyze(create_wav(duration_seconds=0.2, amplitude=4_000))

    assert analysis.duration_seconds == 0.2
    assert len(analysis.segments) == 1
    assert analysis.transcription.text == "reset my pin"
    assert analysis.transcription.confidence == 1.0
    assert analysis.transcription.model == "deterministic-asr-stub-v1"


def test_speech_service_abstains_on_silence() -> None:
    analysis = build_service().analyze(create_wav(duration_seconds=0.2, amplitude=0))

    assert analysis.segments == ()
    assert analysis.transcription.text == ""
    assert analysis.transcription.confidence == 0.0
