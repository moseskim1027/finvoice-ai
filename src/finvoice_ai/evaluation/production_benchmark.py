import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finvoice_ai.application.conversation import ConversationService
from finvoice_ai.domain.models import ConversationRequest
from finvoice_ai.domain.policy import SupportPolicy
from finvoice_ai.infrastructure.document_loader import load_default_documents
from finvoice_ai.infrastructure.local_providers import (
    InMemoryConversationStore,
    TemplateResponseGenerator,
)
from finvoice_ai.infrastructure.retrieval import BM25Retriever
from finvoice_ai.observability import percentile

CPU_DOLLARS_PER_HOUR = 0.05


@dataclass(frozen=True)
class Sample:
    latency_ms: float
    failed: bool
    escalated: bool


def run_benchmark(iterations: int, concurrency: int) -> dict[str, Any]:
    shared = _service()
    requests = [
        ConversationRequest(
            session_id=f"load-{index}",
            message=(
                "How do I reset my PIN?"
                if index % 4
                else "Please transfer money to another account"
            ),
            confidence=0.95,
        )
        for index in range(iterations)
    ]
    warm = _measure(requests, concurrency, lambda request: shared.respond(request))
    cold = _measure(requests, concurrency, lambda request: _service().respond(request))
    return {
        "schema_version": "1.0",
        "workload": {
            "iterations": iterations,
            "concurrency": concurrency,
            "mix": "75% grounded PIN response, 25% policy escalation",
            "external_services": False,
        },
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "logical_cpus": os.cpu_count(),
        },
        "warm_shared_retriever": warm,
        "cold_rebuild_retriever": cold,
        "cache_effect": {
            "p95_latency_reduction_ms": cold["latency_p95_ms"] - warm["latency_p95_ms"],
            "throughput_ratio": warm["throughput_requests_per_second"]
            / max(cold["throughput_requests_per_second"], 1e-12),
        },
        "cost_model": {
            "cpu_dollars_per_hour_assumption": CPU_DOLLARS_PER_HOUR,
            "warm_estimated_cpu_cost_per_conversation_usd": warm[
                "estimated_cpu_cost_per_conversation_usd"
            ],
            "excludes": "network, storage, observability backend, GPU, and external model fees",
        },
        "limitations": [
            "In-process deterministic providers are not a production traffic model.",
            "Process peak RSS includes interpreter and test harness memory.",
            "Cost uses an explicit illustrative CPU-hour rate, not a cloud quote.",
        ],
    }


def _measure(
    requests: list[ConversationRequest],
    concurrency: int,
    handler: Callable[[ConversationRequest], Any],
) -> dict[str, float | int]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()

    def invoke(request: ConversationRequest) -> Sample:
        started = time.perf_counter()
        try:
            response = handler(request)
        except Exception:
            return Sample((time.perf_counter() - started) * 1000, True, False)
        return Sample(
            (time.perf_counter() - started) * 1000,
            False,
            response.decision.value == "escalate",
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(invoke, requests))
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    latencies = [sample.latency_ms for sample in samples]
    return {
        "requests": len(samples),
        "failures": sum(sample.failed for sample in samples),
        "escalations": sum(sample.escalated for sample in samples),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "latency_mean_ms": statistics.fmean(latencies),
        "throughput_requests_per_second": len(samples) / wall_seconds,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "normalized_cpu_utilization": cpu_seconds / wall_seconds / max(os.cpu_count() or 1, 1),
        "peak_rss_bytes": _peak_rss_bytes(),
        "estimated_cpu_cost_per_conversation_usd": (
            cpu_seconds / 3600 * CPU_DOLLARS_PER_HOUR / len(samples)
        ),
    }


def _service() -> ConversationService:
    return ConversationService(
        SupportPolicy(0.7),
        BM25Retriever(load_default_documents()),
        TemplateResponseGenerator(),
        InMemoryConversationStore(),
    )


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic production evidence workload")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations < 1 or args.concurrency < 1:
        raise SystemExit("iterations and concurrency must be positive")
    report = run_benchmark(args.iterations, args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
