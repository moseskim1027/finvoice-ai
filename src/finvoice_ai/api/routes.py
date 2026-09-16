from typing import Annotated

from fastapi import APIRouter, Depends

from finvoice_ai.config import Settings, get_settings
from finvoice_ai.domain.models import ConversationRequest, ConversationResponse, Decision
from finvoice_ai.domain.policy import SupportPolicy

router = APIRouter()


@router.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/v1/conversations/respond",
    response_model=ConversationResponse,
    tags=["conversations"],
)
def respond(
    request: ConversationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationResponse:
    policy = SupportPolicy(settings.minimum_response_confidence)
    decision = policy.evaluate(request.message, request.confidence)

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
