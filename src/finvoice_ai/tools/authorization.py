from finvoice_ai.tools.models import ToolContext, ToolRisk, ToolSpec


class ToolAuthorizationError(PermissionError):
    """Base error for tool authorization failures."""


class AuthenticationRequiredError(ToolAuthorizationError):
    pass


class MissingScopeError(ToolAuthorizationError):
    pass


class ConfirmationRequiredError(ToolAuthorizationError):
    pass


class ToolAuthorizationPolicy:
    """Apply deterministic least-privilege checks before every tool call."""

    def authorize(self, spec: ToolSpec, context: ToolContext) -> None:
        if spec.required_scopes and not context.authenticated:
            raise AuthenticationRequiredError(spec.name)

        missing_scopes = spec.required_scopes - context.scopes
        if missing_scopes:
            raise MissingScopeError(",".join(sorted(missing_scopes)))

        if spec.risk is ToolRisk.SIDE_EFFECT and not context.confirmed:
            raise ConfirmationRequiredError(spec.name)
