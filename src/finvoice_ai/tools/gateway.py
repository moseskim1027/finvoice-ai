from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from finvoice_ai.tools.authorization import ToolAuthorizationPolicy
from finvoice_ai.tools.models import ToolAuditEvent, ToolContext, ToolResult, ToolSpec

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
        self.audit_events: list[ToolAuditEvent] = []

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

        audit_id = str(uuid4())
        argument_names = tuple(sorted(arguments))
        try:
            self._policy.authorize(definition.spec, context)
            content = definition.handler(arguments)
        except Exception:
            self.audit_events.append(
                ToolAuditEvent(
                    audit_id=audit_id,
                    tool_name=tool_name,
                    principal_id=context.principal_id,
                    outcome="denied_or_failed",
                    argument_names=argument_names,
                )
            )
            raise

        self.audit_events.append(
            ToolAuditEvent(
                audit_id=audit_id,
                tool_name=tool_name,
                principal_id=context.principal_id,
                outcome="succeeded",
                argument_names=argument_names,
            )
        )
        return ToolResult(
            tool_name=tool_name,
            audit_id=audit_id,
            content=content,
        )
