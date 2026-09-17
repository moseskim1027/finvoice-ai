from finvoice_ai.speech.models import TranscriptionResult
from finvoice_ai.streaming.models import (
    AcceptedEvent,
    ErrorEvent,
    FinalizedEvent,
    InterruptedEvent,
    PartialEvent,
    PcmChunk,
    SpeechStartedEvent,
)


def test_pcm_chunk_exposes_duration_and_audio_buffer() -> None:
    chunk = PcmChunk("session", "utterance", 3, 16_000, (1,) * 320)

    assert chunk.duration_seconds == 0.02
    assert chunk.as_audio().sample_rate_hz == 16_000
    assert chunk.as_audio().samples == chunk.samples


def test_stream_events_have_stable_discriminators() -> None:
    transcription = TranscriptionResult("hello", 0.8, "en", "stub")
    events = [
        AcceptedEvent("s", "u", 0),
        SpeechStartedEvent("s", "u", 1.0),
        PartialEvent("s", "u", transcription, 1.1),
        FinalizedEvent("s", "u", transcription, "trailing_silence", 0.5, 1.5),
        InterruptedEvent("s", "u", "response", 1.0),
        ErrorEvent("s", "u", "bad_chunk", "invalid", 4),
    ]

    assert [event.type for event in events] == [
        "accepted",
        "speech_started",
        "partial",
        "finalized",
        "interrupted",
        "error",
    ]
