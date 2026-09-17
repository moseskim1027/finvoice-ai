from fastapi.testclient import TestClient

from finvoice_ai.api.routes import MAXIMUM_WAV_BYTES
from finvoice_ai.main import app
from tests.audio_helpers import create_wav

client = TestClient(app)


def test_audio_analysis_endpoint_returns_segments_and_model_metadata() -> None:
    response = client.post(
        "/v1/audio/analyze",
        files={"file": ("speech.wav", create_wav(duration_seconds=0.2), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["duration_seconds"] == 0.2
    assert len(payload["segments"]) == 1
    assert payload["transcription"]["text"] == "demo speech detected"
    assert payload["transcription"]["model"] == "deterministic-asr-stub-v1"


def test_audio_analysis_endpoint_rejects_invalid_wav() -> None:
    response = client.post(
        "/v1/audio/analyze",
        files={"file": ("invalid.wav", b"not-a-wav", "audio/wav")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid WAV container"


def test_audio_analysis_endpoint_rejects_oversized_upload() -> None:
    response = client.post(
        "/v1/audio/analyze",
        files={"file": ("large.wav", b"0" * (MAXIMUM_WAV_BYTES + 1), "audio/wav")},
    )

    assert response.status_code == 413
