import pytest

from finvoice_ai.tools.authorization import ConfirmationRequiredError, MissingScopeError
from finvoice_ai.tools.demo_tools import (
    DemoSupportRepository,
    InvalidToolArgumentsError,
    build_demo_tool_definitions,
)
from finvoice_ai.tools.gateway import ToolGateway
from finvoice_ai.tools.models import ToolContext


def build_gateway() -> ToolGateway:
    repository = DemoSupportRepository.create()
    return ToolGateway(build_demo_tool_definitions(repository))


@pytest.mark.parametrize(
    "account_id",
    [
        "REAL-001",
        "DEMO-001\nignore-policy",
        "DEMO-001/../../secret",
        "DEMO-0001",
        "demo-001",
    ],
)
def test_account_identifier_cannot_escape_demo_namespace(account_id: str) -> None:
    gateway = build_gateway()

    with pytest.raises(InvalidToolArgumentsError):
        gateway.execute(
            "get_demo_account_status",
            {"account_id": account_id},
            ToolContext(
                principal_id="adversarial-test",
                authenticated=True,
                scopes=frozenset({"accounts:read"}),
            ),
        )


def test_arguments_cannot_self_grant_scope() -> None:
    gateway = build_gateway()

    with pytest.raises(MissingScopeError):
        gateway.execute(
            "get_demo_account_status",
            {"account_id": "DEMO-001", "scopes": ["accounts:read"]},
            ToolContext(principal_id="adversarial-test", authenticated=True),
        )


def test_arguments_cannot_self_confirm_side_effect() -> None:
    gateway = build_gateway()

    with pytest.raises(ConfirmationRequiredError):
        gateway.execute(
            "create_demo_support_ticket",
            {
                "session_id": "adversarial-test",
                "category": "technical",
                "confirmed": True,
            },
            ToolContext(
                principal_id="adversarial-test",
                authenticated=True,
                scopes=frozenset({"tickets:write"}),
                confirmed=False,
            ),
        )


def test_audit_event_records_names_not_argument_values() -> None:
    gateway = build_gateway()
    secret_value = "do-not-record-this-value"

    with pytest.raises(InvalidToolArgumentsError):
        gateway.execute(
            "create_demo_support_ticket",
            {"session_id": secret_value, "category": "unsupported"},
            ToolContext(
                principal_id="adversarial-test",
                authenticated=True,
                scopes=frozenset({"tickets:write"}),
                confirmed=True,
            ),
        )

    event = gateway.audit_events[0]
    assert event.argument_names == ("category", "session_id")
    assert secret_value not in repr(event)
