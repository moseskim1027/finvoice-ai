from finvoice_ai.infrastructure.local_providers import KeywordRetriever, TemplateResponseGenerator


def test_keyword_retriever_returns_approved_pin_document() -> None:
    documents = KeywordRetriever().retrieve("How do I reset my PIN?")

    assert [document.document_id for document in documents] == ["help-pin-reset"]


def test_template_generator_abstains_without_context() -> None:
    result = TemplateResponseGenerator().generate("Unknown question", [])

    assert result.confidence == 0.0
    assert result.model == "local-template-v1"
