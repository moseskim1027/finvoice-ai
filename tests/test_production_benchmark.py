import json

import pytest

from finvoice_ai.evaluation.production_benchmark import main, run_benchmark


def test_benchmark_reports_latency_resources_cache_and_cost() -> None:
    report = run_benchmark(iterations=20, concurrency=2)

    assert report["warm_shared_retriever"]["requests"] == 20
    assert report["warm_shared_retriever"]["failures"] == 0
    assert report["warm_shared_retriever"]["escalations"] == 5
    assert report["warm_shared_retriever"]["latency_p99_ms"] >= 0
    assert report["warm_shared_retriever"]["peak_rss_bytes"] > 0
    assert report["cost_model"]["cpu_dollars_per_hour_assumption"] == 0.05
    assert "throughput_ratio" in report["cache_effect"]


def test_benchmark_cli_writes_aggregate_report(tmp_path) -> None:
    output = tmp_path / "report.json"

    assert main(["--iterations", "10", "--concurrency", "2", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["workload"]["iterations"] == 10

    with pytest.raises(SystemExit, match="must be positive"):
        main(["--iterations", "0", "--output", str(output)])
