from uuid import UUID

from fastapi.testclient import TestClient

from finvoice_ai.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_console_is_available() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "FinVoice AI" in response.text
    assert "/static/app.js" in response.text


def test_model_lab_is_available() -> None:
    response = client.get("/lab")

    assert response.status_code == 200
    assert "Model Lab" in response.text


def test_local_runtime_discloses_deterministic_mode() -> None:
    response = client.get("/v1/demo/runtime")

    assert response.status_code == 200
    assert response.json()["output_mode"] == "repeatable contract fixture"


def test_demo_tool_inspector_enforces_confirmation() -> None:
    denied = client.post(
        "/v1/demo/tools/create_demo_support_ticket",
        json={"arguments": {"session_id": "demo", "category": "technical"}},
    )
    allowed = client.post(
        "/v1/demo/tools/create_demo_support_ticket",
        json={"arguments": {"session_id": "demo", "category": "technical"}, "confirmed": True},
    )

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["content"]["synthetic"] is True


def test_readiness_metrics_and_correlation_header() -> None:
    ready = client.get("/ready", headers={"x-request-id": "correlation-123"})
    metrics = client.get("/metrics")

    assert ready.json() == {"status": "ready"}
    assert ready.headers["x-request-id"] == "correlation-123"
    assert "finvoice_http_requests_total" in metrics.text


def test_conversation_can_respond() -> None:
    response = client.post(
        "/v1/conversations/respond",
        json={
            "session_id": "test-session",
            "message": "How do I reset my PIN?",
            "confidence": 0.92,
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "respond"
    assert response.json()["reason"] is None
    assert response.json()["citations"][0]["document_id"] == "pin-reset"
    assert response.json()["citations"][0]["source"] == "knowledge/pin-reset.md"
    assert response.json()["citations"][0]["score"] > 0
    assert response.json()["citations"][0]["excerpt"]
    assert response.json()["provider"] == {
        "model": "local-template-v1",
        "retrieved_documents": 1,
    }
    UUID(response.json()["request_id"])


def test_low_confidence_escalates() -> None:
    response = client.post(
        "/v1/conversations/respond",
        json={
            "session_id": "test-session",
            "message": "I need help with something",
            "confidence": 0.42,
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "escalate"
    assert response.json()["reason"] == "low_confidence"


def test_sensitive_request_escalates() -> None:
    response = client.post(
        "/v1/conversations/respond",
        json={
            "session_id": "test-session",
            "message": "Please transfer money to another account",
            "confidence": 0.99,
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "escalate"
    assert response.json()["reason"] == "sensitive_financial_request"


def test_missing_context_escalates() -> None:
    response = client.post(
        "/v1/conversations/respond",
        json={
            "session_id": "test-session",
            "message": "Can you explain orbital mechanics?",
            "confidence": 0.99,
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "escalate"
    assert response.json()["reason"] == "missing_approved_context"
    assert response.json()["citations"] == []
