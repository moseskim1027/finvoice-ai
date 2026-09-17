from finvoice_ai.domain.models import ConversationRequest, ConversationResponse, Decision
from finvoice_ai.domain.policy import SupportPolicy


class ConversationService:
    """Coordinate policy decisions independently from the HTTP transport."""

    def __init__(self, policy: SupportPolicy) -> None:
        self._policy = policy

    def respond(self, request: ConversationRequest) -> ConversationResponse:
        decision = self._policy.evaluate(request.message, request.confidence)

        if decision.should_escalate:
            return ConversationResponse(
                session_id=request.session_id,
                decision=Decision.ESCALATE,
                message="I am transferring this conversation to a support specialist.",
                reason=decision.reason,
            )

        return ConversationResponse(
            session_id=request.session_id,
            decision=Decision.RESPOND,
            message=(
                "This scaffold has accepted the request. "
                "A grounded response provider will be added in a later milestone."
            ),
        )
