from uuid import uuid4

from finvoice_ai.application.ports import (
    ConversationRecord,
    ConversationStore,
    ProviderUnavailableError,
    ResponseGenerator,
    Retriever,
    SafetyPolicy,
)
from finvoice_ai.domain.models import (
    Citation,
    ConversationRequest,
    ConversationResponse,
    Decision,
    ProviderMetadata,
)


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
        request_id = str(uuid4())
        decision = self._policy.evaluate(request.message, request.confidence)

        if decision.should_escalate:
            response = ConversationResponse(
                request_id=request_id,
                session_id=request.session_id,
                decision=Decision.ESCALATE,
                message="I am transferring this conversation to a support specialist.",
                reason=decision.reason,
            )
        else:
            try:
                context = self._retriever.retrieve(request.message)
                generation = self._generator.generate(request.message, context)
            except ProviderUnavailableError:
                response = ConversationResponse(
                    request_id=request_id,
                    session_id=request.session_id,
                    decision=Decision.ESCALATE,
                    message="The automated service is unavailable. I am transferring your request.",
                    reason="provider_unavailable",
                )
            else:
                if not context:
                    response = ConversationResponse(
                        request_id=request_id,
                        session_id=request.session_id,
                        decision=Decision.ESCALATE,
                        message="I could not verify an answer from approved information.",
                        reason="missing_approved_context",
                        confidence=generation.confidence,
                        provider=ProviderMetadata(
                            model=generation.model,
                            retrieved_documents=0,
                        ),
                    )
                else:
                    response = ConversationResponse(
                        request_id=request_id,
                        session_id=request.session_id,
                        decision=Decision.RESPOND,
                        message=generation.text,
                        confidence=generation.confidence,
                        citations=[
                            Citation(document_id=document.document_id, title=document.title)
                            for document in context
                        ],
                        provider=ProviderMetadata(
                            model=generation.model,
                            retrieved_documents=len(context),
                        ),
                    )

        self._store.append(
            ConversationRecord(
                request_id=request_id,
                session_id=request.session_id,
                user_message=request.message,
                assistant_message=response.message,
                decision=response.decision.value,
            )
        )
        return response
