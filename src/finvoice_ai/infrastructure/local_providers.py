from finvoice_ai.application.ports import (
    ConversationRecord,
    GenerationResult,
    RetrievedDocument,
)


class KeywordRetriever:
    """Deterministic approved-document retrieval for local development."""

    _DOCUMENTS = (
        RetrievedDocument(
            document_id="help-pin-reset",
            title="Resetting your PIN",
            content="Open Settings, choose Security, and select Reset PIN.",
        ),
        RetrievedDocument(
            document_id="help-statements",
            title="Finding account statements",
            content="Open Accounts, select an account, and choose Statements.",
        ),
    )

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        terms = set(query.casefold().split())
        return [
            document
            for document in self._DOCUMENTS
            if terms.intersection(document.content.casefold().split())
            or terms.intersection(document.title.casefold().split())
        ]


class TemplateResponseGenerator:
    """Render approved context without calling an external model."""

    model_name = "local-template-v1"

    def generate(
        self,
        message: str,
        context: list[RetrievedDocument],
    ) -> GenerationResult:
        del message
        if not context:
            return GenerationResult(
                text="I could not find approved information for that question.",
                confidence=0.0,
                model=self.model_name,
            )

        return GenerationResult(
            text=context[0].content,
            confidence=1.0,
            model=self.model_name,
        )


class InMemoryConversationStore:
    def __init__(self) -> None:
        self.records: list[ConversationRecord] = []

    def append(self, record: ConversationRecord) -> None:
        self.records.append(record)
