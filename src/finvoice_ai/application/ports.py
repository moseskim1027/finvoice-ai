from dataclasses import dataclass
from typing import Protocol

from finvoice_ai.domain.policy import PolicyDecision


class ProviderUnavailableError(RuntimeError):
    """Raised when an external or local provider cannot complete a request."""


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    title: str
    content: str
    source: str = "embedded"
    score: float = 0.0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    confidence: float
    model: str
    cited_document_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationRecord:
    request_id: str
    session_id: str
    user_message: str
    assistant_message: str
    decision: str


class SafetyPolicy(Protocol):
    def evaluate(self, message: str, confidence: float) -> PolicyDecision: ...


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedDocument]: ...


class ResponseGenerator(Protocol):
    def generate(
        self,
        message: str,
        context: list[RetrievedDocument],
    ) -> GenerationResult: ...


class ConversationStore(Protocol):
    def append(self, record: ConversationRecord) -> None: ...
