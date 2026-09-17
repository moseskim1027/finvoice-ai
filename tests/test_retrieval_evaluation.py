import json
from pathlib import Path

import pytest

from finvoice_ai.evaluation.retrieval import (
    RetrievalCase,
    default_case_path,
    evaluate_retriever,
    load_cases,
)
from finvoice_ai.infrastructure.document_loader import load_default_documents
from finvoice_ai.infrastructure.retrieval import BM25Retriever


def test_default_bm25_baseline_scores_all_cases() -> None:
    metrics = evaluate_retriever(
        BM25Retriever(load_default_documents()),
        load_cases(default_case_path()),
    )

    assert metrics.case_count == 6
    assert metrics.hit_rate == 1.0
    assert metrics.mean_reciprocal_rank == 1.0


def test_case_loader_reads_versioned_dataset(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"query": "example", "relevant_document_ids": ["document"]}]),
        encoding="utf-8",
    )

    assert load_cases(path) == [
        RetrievalCase(query="example", relevant_document_ids=frozenset({"document"}))
    ]


def test_evaluation_requires_cases() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        evaluate_retriever(BM25Retriever(load_default_documents()), [])
