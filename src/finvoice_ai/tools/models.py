from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolRisk(StrEnum):
    READ_ONLY = "read_only"
    SIDE_EFFECT = "side_effect"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risk: ToolRisk
    required_scopes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolContext:
    principal_id: str
    authenticated: bool
    scopes: frozenset[str] = frozenset()
    confirmed: bool = False


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    audit_id: str
    content: dict[str, Any] = field(default_factory=dict)
