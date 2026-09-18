from uuid import uuid4

from finvoice_ai.application.grounding import validate_citations
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
from finvoice_ai.observability import METRICS, operation


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
        with operation("policy.evaluate"):
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
                with operation("retrieval.retrieve"):
                    context = self._retriever.retrieve(request.message)
                METRICS.increment(
                    "finvoice_retrieval_results", {"has_context": str(bool(context)).lower()}
                )
                with operation("generation.generate"):
                    generation = self._generator.generate(request.message, context)
                with operation("grounding.validate"):
                    validation = validate_citations(generation.cited_document_ids, context)
                METRICS.increment(
                    "finvoice_grounding_results", {"valid": str(validation.is_valid).lower()}
                )
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
                elif not validation.is_valid:
                    response = ConversationResponse(
                        request_id=request_id,
                        session_id=request.session_id,
                        decision=Decision.ESCALATE,
                        message="I could not validate the sources for an automated answer.",
                        reason="invalid_citation",
                        confidence=generation.confidence,
                        provider=ProviderMetadata(
                            model=generation.model,
                            retrieved_documents=len(context),
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
                            Citation(
                                document_id=document.document_id,
                                title=document.title,
                                source=document.source,
                                score=document.score,
                                excerpt=document.content[:240],
                            )
                            for document in validation.cited_documents
                        ],
                        provider=ProviderMetadata(
                            model=generation.model,
                            retrieved_documents=len(context),
                        ),
                    )

        with operation("persistence.append"):
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
