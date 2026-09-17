from finvoice_ai.application.grounding import validate_citations
from finvoice_ai.application.ports import RetrievedDocument

DOCUMENT = RetrievedDocument(
    document_id="approved",
    title="Approved document",
    content="Approved content",
)


def test_valid_citation_resolves_retrieved_document() -> None:
    validation = validate_citations(("approved",), [DOCUMENT])

    assert validation.is_valid is True
    assert validation.cited_documents == (DOCUMENT,)


def test_missing_citation_is_invalid() -> None:
    assert validate_citations((), [DOCUMENT]).is_valid is False


def test_unretrieved_citation_is_invalid() -> None:
    assert validate_citations(("invented",), [DOCUMENT]).is_valid is False
