import pytest

from finvoice_ai.application.ports import RetrievedDocument
from finvoice_ai.infrastructure.document_loader import load_default_documents
from finvoice_ai.infrastructure.retrieval import BM25Retriever


def test_bm25_ranks_pin_document_first() -> None:
    documents = BM25Retriever(load_default_documents()).retrieve("How do I reset my PIN?")

    assert documents[0].document_id == "pin-reset"
    assert documents[0].score > 0


def test_bm25_returns_no_irrelevant_documents() -> None:
    documents = BM25Retriever(load_default_documents()).retrieve("orbital mechanics")

    assert documents == []


def test_bm25_honors_top_k() -> None:
    documents = BM25Retriever(load_default_documents(), top_k=1).retrieve("account support")

    assert len(documents) <= 1


def test_bm25_requires_documents() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        BM25Retriever([])


def test_bm25_requires_positive_top_k() -> None:
    document = RetrievedDocument(document_id="one", title="One", content="Content")

    with pytest.raises(ValueError, match="top_k must be positive"):
        BM25Retriever([document], top_k=0)


def test_bm25_ignores_empty_query() -> None:
    assert BM25Retriever(load_default_documents()).retrieve("---") == []
