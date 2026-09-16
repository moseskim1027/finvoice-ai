import pytest

from finvoice_ai.domain.policy import SupportPolicy


@pytest.mark.parametrize(
    ("message", "confidence", "expected_reason"),
    [
        ("My card was stolen", 0.99, "sensitive_financial_request"),
        ("What are your opening hours?", 0.20, "low_confidence"),
    ],
)
def test_policy_escalation(message: str, confidence: float, expected_reason: str) -> None:
    decision = SupportPolicy(minimum_confidence=0.70).evaluate(message, confidence)

    assert decision.should_escalate is True
    assert decision.reason == expected_reason


def test_policy_allows_safe_confident_request() -> None:
    decision = SupportPolicy(minimum_confidence=0.70).evaluate(
        "Where can I find my statements?",
        0.95,
    )

    assert decision.should_escalate is False
    assert decision.reason is None
