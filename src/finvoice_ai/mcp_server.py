from typing import Any

from mcp.server import MCPServer

from finvoice_ai.tools.demo_tools import DemoSupportRepository, build_demo_tool_definitions
from finvoice_ai.tools.gateway import ToolGateway
from finvoice_ai.tools.models import ToolContext

mcp = MCPServer(
    "FinVoice Demo Tools",
    instructions=(
        "Synthetic portfolio tools only. Never pass real customer identifiers or financial data."
    ),
)

_repository = DemoSupportRepository.create()
_gateway = ToolGateway(build_demo_tool_definitions(_repository))


@mcp.tool()
def get_demo_account_status(account_id: str) -> dict[str, Any]:
    """Read the status of a synthetic account whose ID begins with DEMO-."""

    result = _gateway.execute(
        "get_demo_account_status",
        {"account_id": account_id},
        ToolContext(
            principal_id="mcp-demo-host",
            authenticated=True,
            scopes=frozenset({"accounts:read"}),
        ),
    )
    return {**result.content, "audit_id": result.audit_id}


@mcp.tool()
def create_demo_support_ticket(
    session_id: str,
    category: str,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create a synthetic ticket; confirmed must be true for the write to occur."""

    result = _gateway.execute(
        "create_demo_support_ticket",
        {"session_id": session_id, "category": category},
        ToolContext(
            principal_id="mcp-demo-host",
            authenticated=True,
            scopes=frozenset({"tickets:write"}),
            confirmed=confirmed,
        ),
    )
    return {**result.content, "audit_id": result.audit_id}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
