from finvoice_ai.application.conversation import ConversationService
from finvoice_ai.domain.models import ConversationRequest, Decision
from finvoice_ai.domain.policy import SupportPolicy
from finvoice_ai.infrastructure.local_providers import (
    InMemoryConversationStore,
    KeywordRetriever,
    TemplateResponseGenerator,
)


def build_service(store: InMemoryConversationStore | None = None) -> ConversationService:
    return ConversationService(
        policy=SupportPolicy(minimum_confidence=0.70),
        retriever=KeywordRetriever(),
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
    assert store.records[0].session_id == "stored-session"
    assert store.records[0].decision == "respond"
