import json
import logging

import pytest

from finvoice_ai.observability import (
    JsonFormatter,
    MetricRegistry,
    new_request_id,
    operation,
    percentile,
    request_id_context,
)


def test_json_logs_allow_only_safe_context() -> None:
    token = request_id_context.set("request-1")
    try:
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "complete", (), None)
        record.safe_fields = {
            "event": "complete",
            "model": "stub-v1",
            "transcript": "do not emit me",
            "account_id": "DEMO-123",
        }
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_context.reset(token)

    assert payload["request_id"] == "request-1"
    assert payload["model"] == "stub-v1"
    assert "transcript" not in payload
    assert "account_id" not in payload


def test_metrics_render_and_operation_errors_are_observable(monkeypatch) -> None:
    registry = MetricRegistry()
    registry.increment("requests", {"route": "/safe"})
    registry.observe("duration_seconds", 0.25)
    registry.set_gauge("active_sessions", 2)

    assert 'requests_total{route="/safe"} 1' in registry.prometheus()
    assert "duration_seconds_count 1" in registry.prometheus()
    assert "active_sessions 2" in registry.prometheus()
    registry.reset()
    assert registry.prometheus() == ""

    with pytest.raises(RuntimeError), operation("fault.test"):
        raise RuntimeError("injected")


def test_request_ids_and_percentiles_are_bounded() -> None:
    assert new_request_id("safe-id") == "safe-id"
    assert new_request_id("unsafe id") != "unsafe id"
    assert percentile([], 0.5) == 0.0
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
