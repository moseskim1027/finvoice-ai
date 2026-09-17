import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from finvoice_ai.speech.models import AudioBuffer
from finvoice_ai.streaming.models import (
    AcceptedEvent,
    ErrorEvent,
    FinalizedEvent,
    InterruptedEvent,
    PartialEvent,
    PcmChunk,
    SpeechStartedEvent,
    StreamEvent,
)
from finvoice_ai.streaming.ports import StreamingMetrics, StreamingTranscriptionProvider


class SessionPhase(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEECH = "speech"
    FINALIZING = "finalizing"
    RESPONDING = "responding"


@dataclass(frozen=True)
class StreamingConfig:
    rms_threshold: float = 500.0
    minimum_speech_ms: int = 60
    trailing_silence_ms: int = 300
    maximum_utterance_ms: int = 30_000
    maximum_buffer_ms: int = 35_000
    partial_interval_ms: int = 500
    inactive_session_ms: int = 60_000


@dataclass
class StreamingSession:
    session_id: str
    transcriber: StreamingTranscriptionProvider
    config: StreamingConfig = field(default_factory=StreamingConfig)
    clock: Callable[[], float] = time.monotonic
    metrics: StreamingMetrics | None = None
    phase: SessionPhase = SessionPhase.IDLE
    last_activity_at: float = field(init=False)
    utterance_id: str | None = None
    active_response_id: str | None = None
    _response_generation: int = 0
    _seen_sequences: set[int] = field(default_factory=set)
    _next_sequence: int = 0
    _sample_rate_hz: int | None = None
    _samples: list[int] = field(default_factory=list)
    _speech_samples: int = 0
    _silence_samples: int = 0
    _last_partial_samples: int = 0
    _utterance_started_at: float | None = None
    _speech_started_at: float | None = None
    _first_partial_emitted: bool = False

    def __post_init__(self) -> None:
        self.last_activity_at = self.clock()

    def accept(self, chunk: PcmChunk) -> list[StreamEvent]:
        now = self.clock()
        validation_error = self._validate(chunk)
        if validation_error is not None:
            return [validation_error]
        if self._starts_new_utterance(chunk):
            self._reset_utterance(chunk, now)
        if chunk.sequence in self._seen_sequences:
            self.last_activity_at = now
            return [
                AcceptedEvent(chunk.session_id, chunk.utterance_id, chunk.sequence, duplicate=True)
            ]

        if chunk.sequence != self._next_sequence:
            return [
                ErrorEvent(
                    chunk.session_id,
                    chunk.utterance_id,
                    "out_of_order",
                    f"expected sequence {self._next_sequence}, received {chunk.sequence}",
                    chunk.sequence,
                )
            ]

        if self._duration_ms(len(self._samples) + len(chunk.samples)) > (
            self.config.maximum_buffer_ms
        ):
            return [
                ErrorEvent(
                    chunk.session_id,
                    chunk.utterance_id,
                    "buffer_limit",
                    "streaming audio buffer limit exceeded",
                    chunk.sequence,
                )
            ]

        self.last_activity_at = now
        self._seen_sequences.add(chunk.sequence)
        self._next_sequence += 1
        self._samples.extend(chunk.samples)
        events: list[StreamEvent] = [
            AcceptedEvent(chunk.session_id, chunk.utterance_id, chunk.sequence)
        ]

        voiced = self._rms(chunk.samples) >= self.config.rms_threshold
        if voiced:
            self._speech_samples += len(chunk.samples)
            self._silence_samples = 0
        elif self.phase == SessionPhase.SPEECH:
            self._silence_samples += len(chunk.samples)
        elif self.phase == SessionPhase.LISTENING:
            self._speech_samples = 0

        if self.phase == SessionPhase.LISTENING and self._speech_duration_ms >= (
            self.config.minimum_speech_ms
        ):
            self.phase = SessionPhase.SPEECH
            self._speech_started_at = now
            events.extend(self._interrupt_response(chunk.utterance_id, now))
            events.append(SpeechStartedEvent(self.session_id, chunk.utterance_id, now))
            self._observe(
                "time_to_speech_start_ms",
                self._elapsed_ms(self._utterance_started_at, now),
            )

        if self.phase == SessionPhase.SPEECH:
            if self._utterance_duration_ms >= self.config.maximum_utterance_ms:
                events.extend(self._finalize("maximum_duration", now))
            elif self._silence_duration_ms >= self.config.trailing_silence_ms:
                events.extend(self._finalize("trailing_silence", now))
            elif (
                self._speech_samples - self._last_partial_samples
                >= self._samples_for_ms(self.config.partial_interval_ms)
            ):
                events.append(self._partial(now))

        return events

    def start_response(self, response_id: str) -> int:
        if self.phase not in {SessionPhase.FINALIZING, SessionPhase.RESPONDING}:
            raise ValueError(f"cannot start a response while session is {self.phase}")
        if not response_id:
            raise ValueError("response_id must not be empty")
        self.phase = SessionPhase.RESPONDING
        self.active_response_id = response_id
        self._response_generation += 1
        return self._response_generation

    def complete_response(self, response_id: str, generation: int) -> bool:
        if response_id != self.active_response_id or generation != self._response_generation:
            return False
        self.active_response_id = None
        self.phase = SessionPhase.IDLE
        return True

    def _validate(self, chunk: PcmChunk) -> ErrorEvent | None:
        if chunk.session_id != self.session_id:
            return ErrorEvent(
                chunk.session_id,
                chunk.utterance_id,
                "session_mismatch",
                "chunk session does not match streaming session",
                chunk.sequence,
            )
        if not chunk.utterance_id or chunk.sequence < 0 or chunk.sample_rate_hz <= 0:
            return ErrorEvent(
                chunk.session_id,
                chunk.utterance_id,
                "invalid_chunk",
                "utterance ID, sequence, and sample rate must be valid",
                chunk.sequence,
            )
        if not chunk.samples:
            return ErrorEvent(
                chunk.session_id,
                chunk.utterance_id,
                "invalid_chunk",
                "chunk samples must not be empty",
                chunk.sequence,
            )
        if self.utterance_id == chunk.utterance_id and self._sample_rate_hz not in {
            None,
            chunk.sample_rate_hz,
        }:
            return ErrorEvent(
                chunk.session_id,
                chunk.utterance_id,
                "sample_rate_changed",
                "sample rate cannot change within an utterance",
                chunk.sequence,
            )
        if self.utterance_id not in {None, chunk.utterance_id} and chunk.sequence != 0:
            return ErrorEvent(
                chunk.session_id,
                chunk.utterance_id,
                "utterance_mismatch",
                "a new utterance must begin at sequence zero",
                chunk.sequence,
            )
        if (
            self.utterance_id not in {None, chunk.utterance_id}
            and self.phase not in {SessionPhase.IDLE, SessionPhase.RESPONDING}
        ):
            return ErrorEvent(
                chunk.session_id,
                chunk.utterance_id,
                "invalid_transition",
                f"cannot start a new utterance while session is {self.phase}",
                chunk.sequence,
            )
        return None

    def _starts_new_utterance(self, chunk: PcmChunk) -> bool:
        return self.utterance_id != chunk.utterance_id

    def _reset_utterance(self, chunk: PcmChunk, now: float) -> None:
        if self.phase not in {SessionPhase.IDLE, SessionPhase.RESPONDING}:
            raise ValueError(f"cannot replace active utterance while session is {self.phase}")
        self.utterance_id = chunk.utterance_id
        self._seen_sequences = set()
        self._next_sequence = 0
        self._sample_rate_hz = chunk.sample_rate_hz
        self._samples = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_partial_samples = 0
        self._utterance_started_at = now
        self._speech_started_at = None
        self._first_partial_emitted = False
        self.phase = SessionPhase.LISTENING

    def _interrupt_response(self, utterance_id: str, now: float) -> list[InterruptedEvent]:
        if self.active_response_id is None:
            return []
        response_id = self.active_response_id
        self.active_response_id = None
        self._response_generation += 1
        self._observe("interruption_latency_ms", 0.0)
        return [InterruptedEvent(self.session_id, utterance_id, response_id, now)]

    def _partial(self, now: float) -> PartialEvent:
        self._last_partial_samples = self._speech_samples
        result = self.transcriber.transcribe(self._audio(), is_final=False)
        if not self._first_partial_emitted:
            self._first_partial_emitted = True
            self._observe(
                "time_to_first_partial_ms",
                self._elapsed_ms(self._speech_started_at, now),
            )
        return PartialEvent(self.session_id, self.utterance_id or "", result, now)

    def _finalize(
        self, reason: str, now: float
    ) -> list[FinalizedEvent]:
        self.phase = SessionPhase.FINALIZING
        result = self.transcriber.transcribe(self._audio(), is_final=True)
        duration_seconds = len(self._samples) / (self._sample_rate_hz or 1)
        self._observe("endpoint_delay_ms", self._silence_duration_ms)
        self._observe("final_transcript_latency_ms", self._elapsed_ms(self._speech_started_at, now))
        return [
            FinalizedEvent(
                self.session_id,
                self.utterance_id or "",
                result,
                reason,  # type: ignore[arg-type]
                duration_seconds,
                now,
            )
        ]

    def _audio(self) -> AudioBuffer:
        return AudioBuffer(self._sample_rate_hz or 1, tuple(self._samples))

    @property
    def _speech_duration_ms(self) -> float:
        return self._duration_ms(self._speech_samples)

    @property
    def _silence_duration_ms(self) -> float:
        return self._duration_ms(self._silence_samples)

    @property
    def _utterance_duration_ms(self) -> float:
        return self._duration_ms(len(self._samples))

    def _duration_ms(self, sample_count: int) -> float:
        return sample_count * 1_000 / (self._sample_rate_hz or 1)

    def _samples_for_ms(self, duration_ms: int) -> int:
        return round((self._sample_rate_hz or 1) * duration_ms / 1_000)

    def _observe(self, name: str, value: float) -> None:
        if self.metrics is not None:
            self.metrics.observe(name, value, {"session_id": self.session_id})

    @staticmethod
    def _elapsed_ms(start: float | None, end: float) -> float:
        return 0.0 if start is None else (end - start) * 1_000

    @staticmethod
    def _rms(samples: tuple[int, ...]) -> float:
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class StreamingSessionManager:
    def __init__(
        self,
        transcriber_factory: Callable[[], StreamingTranscriptionProvider],
        *,
        config: StreamingConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        metrics: StreamingMetrics | None = None,
        maximum_sessions: int = 100,
    ) -> None:
        self._transcriber_factory = transcriber_factory
        self._config = config or StreamingConfig()
        self._clock = clock
        self._metrics = metrics
        self._maximum_sessions = maximum_sessions
        self._sessions: dict[str, StreamingSession] = {}

    def get_or_create(self, session_id: str) -> StreamingSession:
        if not session_id:
            raise ValueError("session_id must not be empty")
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        self.cleanup_inactive()
        if len(self._sessions) >= self._maximum_sessions:
            raise RuntimeError("streaming session limit reached")
        session = StreamingSession(
            session_id,
            self._transcriber_factory(),
            self._config,
            self._clock,
            self._metrics,
        )
        self._sessions[session_id] = session
        return session

    def remove(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def cleanup_inactive(self) -> tuple[str, ...]:
        now = self._clock()
        expired = tuple(
            session_id
            for session_id, session in self._sessions.items()
            if (now - session.last_activity_at) * 1_000 >= self._config.inactive_session_ms
        )
        for session_id in expired:
            self.remove(session_id)
        return expired

    @property
    def active_count(self) -> int:
        return len(self._sessions)
