import re
from dataclasses import dataclass
from typing import Any

from finvoice_ai.tools.gateway import ToolDefinition
from finvoice_ai.tools.models import ToolRisk, ToolSpec


class InvalidToolArgumentsError(ValueError):
    pass


@dataclass
class DemoSupportRepository:
    """Synthetic local data only; never stores or returns real customer data."""

    account_statuses: dict[str, str]
    tickets: list[dict[str, str]]

    @classmethod
    def create(cls) -> "DemoSupportRepository":
        return cls(
            account_statuses={"DEMO-001": "active", "DEMO-002": "locked"},
            tickets=[],
        )

    def get_account_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        account_id = arguments.get("account_id")
        if not isinstance(account_id, str) or re.fullmatch(r"DEMO-[0-9]{3}", account_id) is None:
            raise InvalidToolArgumentsError("account_id must be a synthetic DEMO identifier")

        return {
            "account_id": account_id,
            "status": self.account_statuses.get(account_id, "not_found"),
            "synthetic": True,
        }

    def create_support_ticket(self, arguments: dict[str, Any]) -> dict[str, Any]:
        session_id = arguments.get("session_id")
        category = arguments.get("category")
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise InvalidToolArgumentsError("session_id is required")
        if category not in {"card", "account", "technical"}:
            raise InvalidToolArgumentsError("unsupported ticket category")

        ticket_id = f"DEMO-TICKET-{len(self.tickets) + 1:04d}"
        ticket = {"ticket_id": ticket_id, "session_id": session_id, "category": category}
        self.tickets.append(ticket)
        return {**ticket, "synthetic": True}


def build_demo_tool_definitions(repository: DemoSupportRepository) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            spec=ToolSpec(
                name="get_demo_account_status",
                description="Read the status of a synthetic demo account.",
                risk=ToolRisk.READ_ONLY,
                required_scopes=frozenset({"accounts:read"}),
            ),
            handler=repository.get_account_status,
        ),
        ToolDefinition(
            spec=ToolSpec(
                name="create_demo_support_ticket",
                description="Create a synthetic support ticket after explicit confirmation.",
                risk=ToolRisk.SIDE_EFFECT,
                required_scopes=frozenset({"tickets:write"}),
            ),
            handler=repository.create_support_ticket,
        ),
    ]
