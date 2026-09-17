from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from finvoice_ai.tools.authorization import ToolAuthorizationPolicy
from finvoice_ai.tools.models import ToolContext, ToolResult, ToolSpec

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class UnknownToolError(LookupError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    spec: ToolSpec
    handler: ToolHandler


class ToolGateway:
    """Authorize, execute, and audit a fixed registry of typed tool handlers."""

    def __init__(
        self,
        definitions: list[ToolDefinition],
        policy: ToolAuthorizationPolicy | None = None,
    ) -> None:
        self._definitions = {definition.spec.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("tool names must be unique")
        self._policy = policy or ToolAuthorizationPolicy()

    def list_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(definition.spec for definition in self._definitions.values())

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        definition = self._definitions.get(tool_name)
        if definition is None:
            raise UnknownToolError(tool_name)

        self._policy.authorize(definition.spec, context)
        return ToolResult(
            tool_name=tool_name,
            audit_id=str(uuid4()),
            content=definition.handler(arguments),
        )
