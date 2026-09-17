from finvoice_ai.application.conversation import ConversationService
from finvoice_ai.application.ports import (
    GenerationResult,
    ProviderUnavailableError,
    RetrievedDocument,
)
from finvoice_ai.domain.models import ConversationRequest, Decision
from finvoice_ai.domain.policy import SupportPolicy
from finvoice_ai.infrastructure.document_loader import load_default_documents
from finvoice_ai.infrastructure.local_providers import (
    InMemoryConversationStore,
    TemplateResponseGenerator,
)
from finvoice_ai.infrastructure.retrieval import BM25Retriever


def build_service(store: InMemoryConversationStore | None = None) -> ConversationService:
    return ConversationService(
        policy=SupportPolicy(minimum_confidence=0.70),
        retriever=BM25Retriever(load_default_documents()),
        generator=TemplateResponseGenerator(),
        store=store or InMemoryConversationStore(),
    )


def test_service_runs_without_http_transport() -> None:
    service = build_service()

    response = service.respond(
        ConversationRequest(
            session_id="service-test",
            message="Where can I find my statements?",
            confidence=0.95,
        )
    )

    assert response.decision is Decision.RESPOND
    assert response.session_id == "service-test"
    assert response.citations[0].document_id == "statements"
    assert response.provider is not None
    assert response.provider.model == "local-template-v1"


def test_service_escalates_policy_decision() -> None:
    service = build_service()

    response = service.respond(
        ConversationRequest(
            session_id="service-test",
            message="My card was stolen",
            confidence=0.99,
        )
    )

    assert response.decision is Decision.ESCALATE
    assert response.reason == "sensitive_financial_request"


def test_service_records_completed_turn() -> None:
    store = InMemoryConversationStore()
    service = build_service(store)

    service.respond(
        ConversationRequest(
            session_id="stored-session",
            message="How do I reset my PIN?",
            confidence=0.95,
        )
    )

    assert len(store.records) == 1
    assert store.records[0].request_id
    assert store.records[0].session_id == "stored-session"
    assert store.records[0].decision == "respond"


def test_service_safely_escalates_provider_failure() -> None:
    class UnavailableRetriever:
        def retrieve(self, query: str) -> list[RetrievedDocument]:
            del query
            raise ProviderUnavailableError

    service = ConversationService(
        policy=SupportPolicy(minimum_confidence=0.70),
        retriever=UnavailableRetriever(),
        generator=TemplateResponseGenerator(),
        store=InMemoryConversationStore(),
    )

    response = service.respond(
        ConversationRequest(
            session_id="failure-test",
            message="How do I reset my PIN?",
            confidence=0.95,
        )
    )

    assert response.decision is Decision.ESCALATE
    assert response.reason == "provider_unavailable"


def test_service_rejects_unretrieved_citation() -> None:
    class InvalidCitationGenerator:
        def generate(
            self,
            message: str,
            context: list[RetrievedDocument],
        ) -> GenerationResult:
            del message, context
            return GenerationResult(
                text="Unsupported answer",
                confidence=0.99,
                model="invalid-test-model",
                cited_document_ids=("invented-document",),
            )

    service = ConversationService(
        policy=SupportPolicy(minimum_confidence=0.70),
        retriever=BM25Retriever(load_default_documents()),
        generator=InvalidCitationGenerator(),
        store=InMemoryConversationStore(),
    )

    response = service.respond(
        ConversationRequest(
            session_id="citation-test",
            message="How do I reset my PIN?",
            confidence=0.95,
        )
    )

    assert response.decision is Decision.ESCALATE
    assert response.reason == "invalid_citation"
