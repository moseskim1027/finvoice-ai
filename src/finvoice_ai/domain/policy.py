from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    should_escalate: bool
    reason: str | None = None


class SupportPolicy:
    """Small deterministic boundary that stays independent from model behavior."""

    _SENSITIVE_PHRASES = (
        "close my account",
        "freeze my account",
        "send money",
        "transfer money",
        "unauthorized transaction",
        "card was stolen",
    )

    def __init__(self, minimum_confidence: float) -> None:
        self.minimum_confidence = minimum_confidence

    def evaluate(self, message: str, confidence: float) -> PolicyDecision:
        normalized_message = message.casefold()

        if any(phrase in normalized_message for phrase in self._SENSITIVE_PHRASES):
            return PolicyDecision(True, "sensitive_financial_request")

        if confidence < self.minimum_confidence:
            return PolicyDecision(True, "low_confidence")

        return PolicyDecision(False)
