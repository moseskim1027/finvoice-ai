from finvoice_ai.application.conversation import ConversationService
from finvoice_ai.domain.models import ConversationRequest, Decision
from finvoice_ai.domain.policy import SupportPolicy


def test_service_runs_without_http_transport() -> None:
    service = ConversationService(policy=SupportPolicy(minimum_confidence=0.70))

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
    service = ConversationService(policy=SupportPolicy(minimum_confidence=0.70))

    response = service.respond(
        ConversationRequest(
            session_id="service-test",
            message="My card was stolen",
            confidence=0.99,
        )
    )

    assert response.decision is Decision.ESCALATE
    assert response.reason == "sensitive_financial_request"
