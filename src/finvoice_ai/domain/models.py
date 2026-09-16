from enum import StrEnum

from pydantic import BaseModel, Field


class Decision(StrEnum):
    RESPOND = "respond"
    ESCALATE = "escalate"


class ConversationRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ConversationResponse(BaseModel):
    session_id: str
    decision: Decision
    message: str
    reason: str | None = None
