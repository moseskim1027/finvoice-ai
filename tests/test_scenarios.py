import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from finvoice_ai.main import app

SCENARIO_PATH = Path(__file__).parent / "scenarios" / "conversation.json"
SCENARIOS: list[dict[str, Any]] = json.loads(SCENARIO_PATH.read_text())
client = TestClient(app)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[item["name"] for item in SCENARIOS])
def test_conversation_scenario(scenario: dict[str, Any]) -> None:
    response = client.post("/v1/conversations/respond", json=scenario["request"])

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == scenario["expected_decision"]
    assert payload["reason"] == scenario["expected_reason"]

    citation_ids = [citation["document_id"] for citation in payload["citations"]]
    expected_citation = scenario["expected_citation"]
    if expected_citation is None:
        assert citation_ids == []
    else:
        assert expected_citation in citation_ids


@pytest.mark.parametrize(
    "request_payload",
    [
        {"session_id": "", "message": "hello", "confidence": 0.9},
        {"session_id": "validation", "message": "", "confidence": 0.9},
        {"session_id": "validation", "message": "hello", "confidence": 1.1},
    ],
)
def test_malformed_request_is_rejected(request_payload: dict[str, Any]) -> None:
    response = client.post("/v1/conversations/respond", json=request_payload)

    assert response.status_code == 422
