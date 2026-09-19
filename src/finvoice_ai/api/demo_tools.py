"""HTTP-only inspector for the synthetic MCP tool policy demonstration."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from finvoice_ai.config import Settings, get_settings
from finvoice_ai.tools.authorization import ToolAuthorizationError
from finvoice_ai.tools.demo_tools import DemoSupportRepository, build_demo_tool_definitions
from finvoice_ai.tools.gateway import ToolGateway, UnknownToolError
from finvoice_ai.tools.models import ToolContext

router = APIRouter(prefix="/v1/demo", tags=["local demo"])
_gateway = ToolGateway(build_demo_tool_definitions(DemoSupportRepository.create()))


class DemoToolRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False


class DemoToolResponse(BaseModel):
    tool_name: str
    audit_id: str
    outcome: str
    content: dict[str, Any]


class LocalRuntimeResponse(BaseModel):
    transcription_provider: str
    model: str
    output_mode: str


@router.get("/runtime", response_model=LocalRuntimeResponse)
def local_runtime(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LocalRuntimeResponse:
    """Expose the local ASR mode so the lab labels outputs accurately."""
    if settings.transcription_provider == "faster_whisper":
        return LocalRuntimeResponse(
            transcription_provider="faster_whisper",
            model=f"faster-whisper/{settings.whisper_model_size}",
            output_mode="local model inference",
        )
    return LocalRuntimeResponse(
        transcription_provider="deterministic",
        model="deterministic-transcription-v1",
        output_mode="repeatable contract fixture",
    )


@router.post("/tools/{tool_name}", response_model=DemoToolResponse)
def inspect_tool(tool_name: str, request: DemoToolRequest) -> DemoToolResponse:
    """Exercise synthetic tools with a fixed demo principal; never accept user scopes."""
    try:
        result = _gateway.execute(
            tool_name,
            request.arguments,
            ToolContext(
                principal_id="browser-demo",
                authenticated=True,
                scopes=frozenset({"accounts:read", "tickets:write"}),
                confirmed=request.confirmed,
            ),
        )
    except UnknownToolError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown synthetic tool") from error
    except ToolAuthorizationError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return DemoToolResponse(
        tool_name=result.tool_name,
        audit_id=result.audit_id,
        outcome="succeeded",
        content=result.content,
    )
