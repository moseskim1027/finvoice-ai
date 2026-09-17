import base64
import struct

from fastapi.testclient import TestClient

from finvoice_ai.main import create_app


def encoded_samples(value: int, count: int = 320) -> str:
    raw = struct.pack(f"<{count}h", *((value,) * count))
    return base64.b64encode(raw).decode("ascii")


def message(sequence: int, value: int) -> dict[str, object]:
    return {
        "type": "audio_chunk",
        "utterance_id": "utterance-1",
        "sequence": sequence,
        "sample_rate_hz": 16_000,
        "pcm_s16le_base64": encoded_samples(value),
    }


def test_websocket_streams_ordered_pcm_to_a_final_transcript() -> None:
    app = create_app()
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/audio/stream/session-1") as websocket,
    ):
        event_types = []
        final = None
        for sequence in range(3):
            websocket.send_json(message(sequence, 1_000))
            event_types.extend(websocket.receive_json()["type"] for _ in range(1))
            if sequence == 2:
                event_types.append(websocket.receive_json()["type"])
        for sequence in range(3, 18):
            websocket.send_json(message(sequence, 0))
            event_types.append(websocket.receive_json()["type"])
            if sequence == 17:
                final = websocket.receive_json()
                event_types.append(final["type"])

    assert event_types.count("speech_started") == 1
    assert event_types.count("finalized") == 1
    assert final is not None
    assert final["transcription"]["text"] == "demo speech detected"
    assert app.state.streaming_manager.active_count == 0


def test_websocket_reports_bad_payload_and_out_of_order_chunk() -> None:
    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/v1/audio/stream/session-1") as websocket,
    ):
        websocket.send_json({"type": "unknown"})
        bad_message = websocket.receive_json()
        websocket.send_json(message(1, 0))
        out_of_order = websocket.receive_json()

    assert bad_message["type"] == "error"
    assert bad_message["code"] == "bad_message"
    assert out_of_order["type"] == "error"
    assert out_of_order["code"] == "out_of_order"


def test_websocket_rejects_malformed_pcm() -> None:
    payload = message(0, 0)
    payload["pcm_s16le_base64"] = "not base64!"

    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/v1/audio/stream/session-1") as websocket,
    ):
        websocket.send_json(payload)
        event = websocket.receive_json()

    assert event["code"] == "bad_message"
