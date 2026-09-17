from uuid import UUID

import pytest

from finvoice_ai.tools.authorization import ConfirmationRequiredError
from finvoice_ai.tools.demo_tools import (
    DemoSupportRepository,
    InvalidToolArgumentsError,
    build_demo_tool_definitions,
)
from finvoice_ai.tools.gateway import ToolGateway, UnknownToolError
from finvoice_ai.tools.models import ToolContext


def build_gateway() -> tuple[ToolGateway, DemoSupportRepository]:
    repository = DemoSupportRepository.create()
    return ToolGateway(build_demo_tool_definitions(repository)), repository


def test_gateway_reads_synthetic_account_status() -> None:
    gateway, _ = build_gateway()

    result = gateway.execute(
        "get_demo_account_status",
        {"account_id": "DEMO-001"},
        ToolContext(
            principal_id="demo-user",
            authenticated=True,
            scopes=frozenset({"accounts:read"}),
        ),
    )

    assert result.content == {
        "account_id": "DEMO-001",
        "status": "active",
        "synthetic": True,
    }
    UUID(result.audit_id)
    assert gateway.audit_events[0].audit_id == result.audit_id
    assert gateway.audit_events[0].outcome == "succeeded"
    assert gateway.audit_events[0].argument_names == ("account_id",)


def test_gateway_requires_confirmation_before_ticket_write() -> None:
    gateway, repository = build_gateway()

    with pytest.raises(ConfirmationRequiredError):
        gateway.execute(
            "create_demo_support_ticket",
            {"session_id": "session", "category": "card"},
            ToolContext(
                principal_id="demo-user",
                authenticated=True,
                scopes=frozenset({"tickets:write"}),
            ),
        )

    assert repository.tickets == []
    assert gateway.audit_events[0].outcome == "denied_or_failed"


def test_gateway_executes_confirmed_ticket_write() -> None:
    gateway, repository = build_gateway()

    result = gateway.execute(
        "create_demo_support_ticket",
        {"session_id": "session", "category": "card"},
        ToolContext(
            principal_id="demo-user",
            authenticated=True,
            scopes=frozenset({"tickets:write"}),
            confirmed=True,
        ),
    )

    assert result.content["ticket_id"] == "DEMO-TICKET-0001"
    assert len(repository.tickets) == 1


def test_gateway_rejects_unknown_tool() -> None:
    gateway, _ = build_gateway()

    with pytest.raises(UnknownToolError):
        gateway.execute(
            "invented_tool",
            {},
            ToolContext(principal_id="demo-user", authenticated=True),
        )


def test_gateway_rejects_non_demo_account_identifier() -> None:
    gateway, _ = build_gateway()

    with pytest.raises(InvalidToolArgumentsError, match="synthetic DEMO"):
        gateway.execute(
            "get_demo_account_status",
            {"account_id": "REAL-123"},
            ToolContext(
                principal_id="demo-user",
                authenticated=True,
                scopes=frozenset({"accounts:read"}),
            ),
        )


def test_gateway_requires_unique_names() -> None:
    repository = DemoSupportRepository.create()
    definition = build_demo_tool_definitions(repository)[0]

    with pytest.raises(ValueError, match="unique"):
        ToolGateway([definition, definition])
