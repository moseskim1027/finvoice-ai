from dataclasses import dataclass, field

from finvoice_ai.speech.models import AudioBuffer, SpeechSegment, TranscriptionResult
from finvoice_ai.streaming.transcription import OfflineStreamingTranscriptionAdapter


@dataclass
class RecordingProvider:
    calls: list[tuple[AudioBuffer, list[SpeechSegment]]] = field(default_factory=list)

    def transcribe(self, audio: AudioBuffer, segments: list[SpeechSegment]) -> TranscriptionResult:
        self.calls.append((audio, segments))
        return TranscriptionResult("simulated partial", 0.6, "en", "offline")


def test_offline_adapter_runs_vad_for_partial_and_final_calls() -> None:
    provider = RecordingProvider()
    adapter = OfflineStreamingTranscriptionAdapter(provider)
    audio = AudioBuffer(1_000, (1_000,) * 100)

    partial = adapter.transcribe(audio, is_final=False)
    final = adapter.transcribe(audio, is_final=True)

    assert partial == final
    assert partial.text == "simulated partial"
    assert len(provider.calls) == 2
    assert provider.calls[0][0] is audio
    assert provider.calls[0][1] == [SpeechSegment(0.0, 0.1, 1_000.0)]
