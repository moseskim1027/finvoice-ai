from pathlib import Path

import pytest

from finvoice_ai.infrastructure.document_loader import (
    InvalidDocumentError,
    MarkdownDocumentLoader,
    load_default_documents,
)


def test_default_knowledge_base_loads() -> None:
    documents = load_default_documents()

    assert {document.document_id for document in documents} == {
        "card-security",
        "pin-reset",
        "statements",
    }
    assert all(document.content for document in documents)
    assert all(document.source.startswith("knowledge/") for document in documents)


def test_document_requires_heading(tmp_path: Path) -> None:
    path = tmp_path / "invalid.md"
    path.write_text("No heading\n\nSome content", encoding="utf-8")

    with pytest.raises(InvalidDocumentError, match="level-one heading"):
        MarkdownDocumentLoader().load(path)


def test_document_requires_body(tmp_path: Path) -> None:
    path = tmp_path / "invalid.md"
    path.write_text("# Empty document", encoding="utf-8")

    with pytest.raises(InvalidDocumentError, match="title and body"):
        MarkdownDocumentLoader().load(path)


def test_directory_requires_documents(tmp_path: Path) -> None:
    with pytest.raises(InvalidDocumentError, match="no Markdown documents"):
        MarkdownDocumentLoader().load_directory(tmp_path)
