from typing import Annotated

from fastapi import APIRouter, Depends

from finvoice_ai.application.conversation import ConversationService
from finvoice_ai.config import Settings, get_settings
from finvoice_ai.domain.models import ConversationRequest, ConversationResponse
from finvoice_ai.domain.policy import SupportPolicy
from finvoice_ai.infrastructure.local_providers import (
    InMemoryConversationStore,
    KeywordRetriever,
    TemplateResponseGenerator,
)

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
    service = ConversationService(
        policy=SupportPolicy(settings.minimum_response_confidence),
        retriever=KeywordRetriever(),
        generator=TemplateResponseGenerator(),
        store=InMemoryConversationStore(),
    )
    return service.respond(request)
