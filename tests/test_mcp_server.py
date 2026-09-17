import asyncio
from uuid import UUID

import pytest

from finvoice_ai.mcp_server import create_demo_support_ticket, get_demo_account_status, mcp
from finvoice_ai.tools.authorization import ConfirmationRequiredError


def test_mcp_server_identity() -> None:
    assert mcp.name == "FinVoice Demo Tools"


def test_mcp_server_registers_only_intended_tools() -> None:
    tools = asyncio.run(mcp.list_tools())

    assert {tool.name for tool in tools} == {
        "create_demo_support_ticket",
        "get_demo_account_status",
    }


def test_mcp_account_tool_uses_gateway() -> None:
    result = get_demo_account_status("DEMO-001")

    assert result["status"] == "active"
    assert result["synthetic"] is True
    UUID(result["audit_id"])


def test_mcp_ticket_tool_requires_confirmation() -> None:
    with pytest.raises(ConfirmationRequiredError):
        create_demo_support_ticket("mcp-session", "technical")


def test_mcp_ticket_tool_returns_audit_record() -> None:
    result = create_demo_support_ticket("mcp-session", "technical", confirmed=True)

    assert result["ticket_id"].startswith("DEMO-TICKET-")
    assert result["synthetic"] is True
    UUID(result["audit_id"])
