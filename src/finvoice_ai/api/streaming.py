import base64
import binascii
import struct
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from finvoice_ai.observability import METRICS, operation
from finvoice_ai.streaming.models import ErrorEvent, PcmChunk, StreamEvent
from finvoice_ai.streaming.session import StreamingSessionManager

router = APIRouter()


@router.websocket("/v1/audio/stream/{session_id}")
async def stream_audio(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    manager: StreamingSessionManager = websocket.app.state.streaming_manager
    try:
        with operation("streaming.session.open"):
            session = manager.get_or_create(session_id)
        METRICS.set_gauge("finvoice_streaming_active_sessions", manager.active_count)
    except (RuntimeError, ValueError) as error:
        METRICS.increment("finvoice_streaming_rejections", {"reason": "session_limit"})
        await websocket.send_json(_error_payload(session_id, "", "session_rejected", str(error)))
        await websocket.close(code=1013)
        return

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                chunk = _decode_chunk(payload, session_id)
            except (KeyError, TypeError, ValueError, binascii.Error) as error:
                await websocket.send_json(
                    _error_payload(
                        session_id,
                        str(payload.get("utterance_id", "")),
                        "bad_message",
                        str(error),
                    )
                )
                continue
            with operation("streaming.chunk.accept"):
                events = session.accept(chunk)
            for event in events:
                await websocket.send_json(_event_payload(event))
    except WebSocketDisconnect:
        pass
    finally:
        manager.remove(session_id)
        METRICS.set_gauge("finvoice_streaming_active_sessions", manager.active_count)


def _decode_chunk(payload: dict[str, Any], session_id: str) -> PcmChunk:
    if payload.get("type") != "audio_chunk":
        raise ValueError("message type must be 'audio_chunk'")
    encoded = payload["pcm_s16le_base64"]
    if not isinstance(encoded, str):
        raise TypeError("pcm_s16le_base64 must be a string")
    raw = base64.b64decode(encoded, validate=True)
    if not raw or len(raw) % 2:
        raise ValueError("PCM payload must contain complete signed 16-bit samples")
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    return PcmChunk(
        session_id=session_id,
        utterance_id=str(payload["utterance_id"]),
        sequence=int(payload["sequence"]),
        sample_rate_hz=int(payload["sample_rate_hz"]),
        samples=samples,
    )


def _event_payload(event: StreamEvent) -> dict[str, Any]:
    return asdict(event)


def _error_payload(
    session_id: str,
    utterance_id: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    return asdict(ErrorEvent(session_id, utterance_id, code, message))
