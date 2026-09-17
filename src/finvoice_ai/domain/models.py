from enum import StrEnum

from pydantic import BaseModel, Field


class Decision(StrEnum):
    RESPOND = "respond"
    ESCALATE = "escalate"


class ConversationRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Citation(BaseModel):
    document_id: str
    title: str
    source: str
    score: float = Field(ge=0.0)
    excerpt: str


class ProviderMetadata(BaseModel):
    model: str
    retrieved_documents: int = Field(ge=0)


class ConversationResponse(BaseModel):
    request_id: str
    session_id: str
    decision: Decision
    message: str
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    provider: ProviderMetadata | None = None
