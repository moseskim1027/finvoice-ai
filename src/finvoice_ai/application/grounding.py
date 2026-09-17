from dataclasses import dataclass

from finvoice_ai.application.ports import RetrievedDocument


@dataclass(frozen=True)
class CitationValidation:
    is_valid: bool
    cited_documents: tuple[RetrievedDocument, ...] = ()


def validate_citations(
    cited_document_ids: tuple[str, ...],
    retrieved_documents: list[RetrievedDocument],
) -> CitationValidation:
    """Require every generated citation to refer to retrieved approved context."""

    if not cited_document_ids:
        return CitationValidation(is_valid=False)

    documents_by_id = {document.document_id: document for document in retrieved_documents}
    if any(document_id not in documents_by_id for document_id in cited_document_ids):
        return CitationValidation(is_valid=False)

    return CitationValidation(
        is_valid=True,
        cited_documents=tuple(documents_by_id[document_id] for document_id in cited_document_ids),
    )
