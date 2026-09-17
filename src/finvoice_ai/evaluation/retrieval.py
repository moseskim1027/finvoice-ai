import json
from dataclasses import dataclass
from pathlib import Path

from finvoice_ai.application.ports import Retriever
from finvoice_ai.infrastructure.document_loader import load_default_documents
from finvoice_ai.infrastructure.retrieval import BM25Retriever


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_document_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    case_count: int
    hit_rate: float
    mean_reciprocal_rank: float


def load_cases(path: Path) -> list[RetrievalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RetrievalCase(
            query=item["query"],
            relevant_document_ids=frozenset(item["relevant_document_ids"]),
        )
        for item in payload
    ]


def evaluate_retriever(retriever: Retriever, cases: list[RetrievalCase]) -> RetrievalMetrics:
    if not cases:
        raise ValueError("retrieval evaluation requires at least one case")

    hits = 0
    reciprocal_rank_total = 0.0
    for case in cases:
        results = retriever.retrieve(case.query)
        relevant_ranks = [
            rank
            for rank, document in enumerate(results, start=1)
            if document.document_id in case.relevant_document_ids
        ]
        if relevant_ranks:
            hits += 1
            reciprocal_rank_total += 1 / min(relevant_ranks)

    return RetrievalMetrics(
        case_count=len(cases),
        hit_rate=hits / len(cases),
        mean_reciprocal_rank=reciprocal_rank_total / len(cases),
    )


def default_case_path() -> Path:
    return Path(__file__).parent / "data" / "retrieval_cases.json"


def main() -> None:
    metrics = evaluate_retriever(
        BM25Retriever(load_default_documents()),
        load_cases(default_case_path()),
    )
    print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":
    main()
