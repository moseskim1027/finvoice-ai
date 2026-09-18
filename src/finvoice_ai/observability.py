import json
import logging
import math
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

request_id_context: ContextVar[str] = ContextVar("request_id", default="")
session_id_context: ContextVar[str] = ContextVar("session_id", default="")

SAFE_LOG_FIELDS = {
    "request_id",
    "session_id",
    "utterance_id",
    "audit_id",
    "model",
    "policy_decision",
    "citation_count",
    "escalation_reason",
    "method",
    "route",
    "status_code",
    "duration_ms",
    "event",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                "request_id": request_id_context.get(),
                "session_id": session_id_context.get(),
            }
        )
        fields = getattr(record, "safe_fields", {})
        payload.update({key: value for key, value in fields.items() if key in SAFE_LOG_FIELDS})
        return json.dumps({key: value for key, value in payload.items() if value != ""})


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


@dataclass
class MetricRegistry:
    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    observations: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        with self._lock:
            self.counters[(name, _labels(labels))] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self.observations[(name, _labels(labels))].append(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self.gauges[(name, _labels(labels))] = value

    def prometheus(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), value in sorted(self.counters.items()):
                lines.append(f"{name}_total{_format_labels(labels)} {value:g}")
            for (name, labels), values in sorted(self.observations.items()):
                lines.append(f"{name}_count{_format_labels(labels)} {len(values)}")
                lines.append(f"{name}_sum{_format_labels(labels)} {sum(values):g}")
            for (name, labels), value in sorted(self.gauges.items()):
                lines.append(f"{name}{_format_labels(labels)} {value:g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.observations.clear()
            self.gauges.clear()


class SanitizedSpanExporter(SpanExporter):
    """Retain bounded span metadata for local evidence without request content."""

    def __init__(self, maximum_spans: int = 1_000) -> None:
        self.maximum_spans = maximum_spans
        self.spans: list[dict[str, Any]] = []

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            attributes = {
                key: value
                for key, value in (span.attributes or {}).items()
                if key.startswith(("http.", "finvoice."))
            }
            self.spans.append(
                {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "attributes": attributes,
                }
            )
        self.spans[:] = self.spans[-self.maximum_spans :]
        return SpanExportResult.SUCCESS


METRICS = MetricRegistry()
SPAN_EXPORTER = SanitizedSpanExporter()


class BoundedStreamingMetrics:
    def observe(self, name: str, value: float, tags: dict[str, str]) -> None:
        METRICS.observe(f"finvoice_streaming_{name}", value)


def configure_tracing() -> None:
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "finvoice-ai"}))
    provider.add_span_processor(SimpleSpanProcessor(SPAN_EXPORTER))
    trace.set_tracer_provider(provider)


@contextmanager
def operation(name: str, **attributes: str | int | float | bool) -> Iterator[None]:
    tracer = trace.get_tracer("finvoice_ai")
    started = time.perf_counter()
    with tracer.start_as_current_span(
        name, attributes={f"finvoice.{k}": v for k, v in attributes.items()}
    ):
        try:
            yield
        except Exception:
            METRICS.increment("finvoice_operation_errors", {"operation": name})
            raise
        finally:
            METRICS.observe(
                "finvoice_operation_duration_seconds",
                time.perf_counter() - started,
                {"operation": name},
            )


def new_request_id(value: str | None = None) -> str:
    candidate = (value or "").strip()
    if (
        candidate
        and len(candidate) <= 128
        and all(char.isalnum() or char in "-_." for char in candidate)
    ):
        return candidate
    return str(uuid4())


def _labels(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in (labels or {}).items()))


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    values = ",".join(f'{key}="{value}"' for key, value in labels)
    return "{" + values + "}"


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]
