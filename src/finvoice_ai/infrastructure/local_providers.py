import re

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

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.casefold()))

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        terms = self._terms(query)
        return [
            document
            for document in self._DOCUMENTS
            if terms.intersection(self._terms(document.content))
            or terms.intersection(self._terms(document.title))
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
