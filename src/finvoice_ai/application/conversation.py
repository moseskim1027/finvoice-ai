from finvoice_ai.application.ports import (
    ConversationRecord,
    ConversationStore,
    ResponseGenerator,
    Retriever,
    SafetyPolicy,
)
from finvoice_ai.domain.models import ConversationRequest, ConversationResponse, Decision


class ConversationService:
    """Coordinate policy decisions independently from the HTTP transport."""

    def __init__(
        self,
        policy: SafetyPolicy,
        retriever: Retriever,
        generator: ResponseGenerator,
        store: ConversationStore,
    ) -> None:
        self._policy = policy
        self._retriever = retriever
        self._generator = generator
        self._store = store

    def respond(self, request: ConversationRequest) -> ConversationResponse:
        decision = self._policy.evaluate(request.message, request.confidence)

        if decision.should_escalate:
            response = ConversationResponse(
                session_id=request.session_id,
                decision=Decision.ESCALATE,
                message="I am transferring this conversation to a support specialist.",
                reason=decision.reason,
            )
        else:
            context = self._retriever.retrieve(request.message)
            generation = self._generator.generate(request.message, context)
            response = ConversationResponse(
                session_id=request.session_id,
                decision=Decision.RESPOND,
                message=generation.text,
            )

        self._store.append(
            ConversationRecord(
                session_id=request.session_id,
                user_message=request.message,
                assistant_message=response.message,
                decision=response.decision.value,
            )
        )
        return response
