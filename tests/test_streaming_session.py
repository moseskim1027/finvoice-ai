from dataclasses import dataclass, field

import pytest

from finvoice_ai.speech.models import AudioBuffer, TranscriptionResult
from finvoice_ai.streaming.models import PcmChunk
from finvoice_ai.streaming.session import (
    SessionPhase,
    StreamingConfig,
    StreamingSession,
    StreamingSessionManager,
)


@dataclass
class FakeClock:
    value: float = 10.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class StubTranscriber:
    calls: list[tuple[AudioBuffer, bool]] = field(default_factory=list)

    def transcribe(self, audio: AudioBuffer, *, is_final: bool) -> TranscriptionResult:
        self.calls.append((audio, is_final))
        return TranscriptionResult(f"samples:{len(audio.samples)}", 0.75, "en", "stub")


@dataclass
class MetricRecorder:
    observations: list[tuple[str, float, dict[str, str]]] = field(default_factory=list)

    def observe(self, name: str, value: float, attributes: dict[str, str]) -> None:
        self.observations.append((name, value, attributes))


def chunk(
    sequence: int,
    value: int,
    *,
    utterance_id: str = "utterance-1",
    session_id: str = "session-1",
    samples: int = 20,
    sample_rate_hz: int = 1_000,
) -> PcmChunk:
    return PcmChunk(session_id, utterance_id, sequence, sample_rate_hz, (value,) * samples)


def event_types(events: list[object]) -> list[str]:
    return [event.type for event in events]  # type: ignore[attr-defined]


def config(**overrides: int | float) -> StreamingConfig:
    values: dict[str, int | float] = {
        "rms_threshold": 500.0,
        "minimum_speech_ms": 60,
        "trailing_silence_ms": 60,
        "maximum_utterance_ms": 500,
        "maximum_buffer_ms": 600,
        "partial_interval_ms": 60,
        "inactive_session_ms": 1_000,
    }
    values.update(overrides)
    return StreamingConfig(**values)  # type: ignore[arg-type]


def test_ordered_chunks_produce_partial_and_one_final_utterance() -> None:
    transcriber = StubTranscriber()
    session = StreamingSession("session-1", transcriber, config())

    events = []
    for sequence in range(3):
        events.extend(session.accept(chunk(sequence, 1_000)))
    for sequence in range(3, 6):
        events.extend(session.accept(chunk(sequence, 0)))

    assert event_types(events).count("speech_started") == 1
    assert event_types(events).count("partial") == 1
    assert event_types(events).count("finalized") == 1
    assert [is_final for _, is_final in transcriber.calls] == [False, True]
    assert session.phase == SessionPhase.FINALIZING


def test_duplicates_are_idempotent_and_out_of_order_chunks_are_rejected() -> None:
    session = StreamingSession("session-1", StubTranscriber(), config())

    first = session.accept(chunk(0, 0))
    duplicate = session.accept(chunk(0, 0))
    rejected = session.accept(chunk(2, 0))
    second = session.accept(chunk(1, 0))

    assert first[0].type == "accepted"
    assert duplicate[0].duplicate is True  # type: ignore[union-attr]
    assert rejected[0].code == "out_of_order"  # type: ignore[union-attr]
    assert second[0].type == "accepted"


def test_initial_silence_and_short_noise_do_not_create_utterance() -> None:
    session = StreamingSession("session-1", StubTranscriber(), config())

    events = session.accept(chunk(0, 0)) + session.accept(chunk(1, 1_000))
    events += session.accept(chunk(2, 0))

    assert "speech_started" not in event_types(events)
    assert "finalized" not in event_types(events)


def test_short_pause_does_not_finalize_but_trailing_silence_does() -> None:
    session = StreamingSession("session-1", StubTranscriber(), config())
    events = []
    for sequence in range(3):
        events.extend(session.accept(chunk(sequence, 1_000)))
    events.extend(session.accept(chunk(3, 0)))
    events.extend(session.accept(chunk(4, 1_000)))
    events.extend(session.accept(chunk(5, 0)))
    events.extend(session.accept(chunk(6, 0)))
    events.extend(session.accept(chunk(7, 0)))

    assert event_types(events).count("finalized") == 1
    final = next(event for event in events if event.type == "finalized")
    assert final.reason == "trailing_silence"  # type: ignore[union-attr]


def test_maximum_duration_forces_finalization_exactly_once() -> None:
    session = StreamingSession(
        "session-1",
        StubTranscriber(),
        config(maximum_utterance_ms=100, partial_interval_ms=1_000),
    )
    events = []
    for sequence in range(5):
        events.extend(session.accept(chunk(sequence, 1_000)))

    assert event_types(events).count("finalized") == 1
    final = next(event for event in events if event.type == "finalized")
    assert final.reason == "maximum_duration"  # type: ignore[union-attr]


def test_buffer_limit_is_enforced() -> None:
    session = StreamingSession(
        "session-1",
        StubTranscriber(),
        config(maximum_buffer_ms=30, minimum_speech_ms=100),
    )

    session.accept(chunk(0, 0, samples=20))
    events = session.accept(chunk(1, 0, samples=20))

    assert event_types(events) == ["error"]
    assert events[-1].code == "buffer_limit"  # type: ignore[union-attr]


def test_barge_in_cancels_active_response_once_and_late_completion_is_ignored() -> None:
    session = StreamingSession("session-1", StubTranscriber(), config())
    cancellations: list[str] = []
    for sequence in range(3):
        session.accept(chunk(sequence, 1_000))
    for sequence in range(3, 6):
        session.accept(chunk(sequence, 0))
    generation = session.start_response("response-1", lambda: cancellations.append("cancelled"))

    events = []
    for sequence in range(3):
        events.extend(
            session.accept(chunk(sequence, 1_000, utterance_id="utterance-2"))
        )
    events.extend(session.accept(chunk(3, 1_000, utterance_id="utterance-2")))

    assert event_types(events).count("interrupted") == 1
    assert cancellations == ["cancelled"]
    assert session.complete_response("response-1", generation) is False


def test_response_completion_requires_current_identity_and_generation() -> None:
    session = StreamingSession("session-1", StubTranscriber(), config())
    session.phase = SessionPhase.FINALIZING
    generation = session.start_response("response")

    assert session.complete_response("other", generation) is False
    assert session.complete_response("response", generation + 1) is False
    assert session.complete_response("response", generation) is True
    assert session.phase == SessionPhase.IDLE


def test_metrics_capture_streaming_timing() -> None:
    clock = FakeClock()
    metrics = MetricRecorder()
    session = StreamingSession("session-1", StubTranscriber(), config(), clock, metrics)
    for sequence in range(3):
        clock.advance(0.02)
        session.accept(chunk(sequence, 1_000))
    for sequence in range(3, 6):
        clock.advance(0.02)
        session.accept(chunk(sequence, 0))

    names = {name for name, _, _ in metrics.observations}
    assert names == {
        "time_to_speech_start_ms",
        "time_to_first_partial_ms",
        "endpoint_delay_ms",
        "final_transcript_latency_ms",
    }


def test_manager_bounds_sessions_and_cleans_up_inactive_state() -> None:
    clock = FakeClock()
    manager = StreamingSessionManager(
        StubTranscriber,
        config=config(inactive_session_ms=100),
        clock=clock,
        maximum_sessions=1,
    )
    first = manager.get_or_create("one")
    assert manager.get_or_create("one") is first
    with pytest.raises(RuntimeError, match="session limit"):
        manager.get_or_create("two")

    clock.advance(0.101)
    second = manager.get_or_create("two")

    assert second.session_id == "two"
    assert manager.active_count == 1
    assert manager.remove("two") is True
    assert manager.remove("missing") is False


@pytest.mark.parametrize(
    ("bad_chunk", "code"),
    [
        (chunk(0, 0, session_id="wrong"), "session_mismatch"),
        (chunk(-1, 0), "invalid_chunk"),
        (chunk(0, 0, samples=0), "invalid_chunk"),
    ],
)
def test_invalid_chunks_return_error_events(bad_chunk: PcmChunk, code: str) -> None:
    session = StreamingSession("session-1", StubTranscriber(), config())

    events = session.accept(bad_chunk)

    assert events[0].code == code  # type: ignore[union-attr]


def test_concurrent_sessions_do_not_share_audio_or_state() -> None:
    transcribers: list[StubTranscriber] = []

    def factory() -> StubTranscriber:
        transcriber = StubTranscriber()
        transcribers.append(transcriber)
        return transcriber

    manager = StreamingSessionManager(factory, config=config())
    first = manager.get_or_create("first")
    second = manager.get_or_create("second")

    first.accept(chunk(0, 1_000, session_id="first"))
    second.accept(chunk(0, 0, session_id="second"))

    assert first.phase == SessionPhase.LISTENING
    assert second.phase == SessionPhase.LISTENING
    assert first is not second
