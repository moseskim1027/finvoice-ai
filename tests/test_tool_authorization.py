import pytest

from finvoice_ai.tools.authorization import (
    AuthenticationRequiredError,
    ConfirmationRequiredError,
    MissingScopeError,
    ToolAuthorizationPolicy,
)
from finvoice_ai.tools.models import ToolContext, ToolRisk, ToolSpec

ACCOUNT_LOOKUP = ToolSpec(
    name="get_demo_account_status",
    description="Read synthetic account status.",
    risk=ToolRisk.READ_ONLY,
    required_scopes=frozenset({"accounts:read"}),
)
CREATE_TICKET = ToolSpec(
    name="create_demo_support_ticket",
    description="Create a synthetic support ticket.",
    risk=ToolRisk.SIDE_EFFECT,
    required_scopes=frozenset({"tickets:write"}),
)


def test_protected_tool_requires_authentication() -> None:
    with pytest.raises(AuthenticationRequiredError):
        ToolAuthorizationPolicy().authorize(
            ACCOUNT_LOOKUP,
            ToolContext(principal_id="anonymous", authenticated=False),
        )


def test_protected_tool_requires_scope() -> None:
    with pytest.raises(MissingScopeError, match="accounts:read"):
        ToolAuthorizationPolicy().authorize(
            ACCOUNT_LOOKUP,
            ToolContext(principal_id="demo", authenticated=True),
        )


def test_side_effect_requires_confirmation() -> None:
    with pytest.raises(ConfirmationRequiredError):
        ToolAuthorizationPolicy().authorize(
            CREATE_TICKET,
            ToolContext(
                principal_id="demo",
                authenticated=True,
                scopes=frozenset({"tickets:write"}),
            ),
        )


def test_authorized_confirmed_action_passes() -> None:
    ToolAuthorizationPolicy().authorize(
        CREATE_TICKET,
        ToolContext(
            principal_id="demo",
            authenticated=True,
            scopes=frozenset({"tickets:write"}),
            confirmed=True,
        ),
    )
