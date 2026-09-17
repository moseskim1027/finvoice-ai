import pytest

from finvoice_ai.evaluation.speech import (
    character_error_rate,
    normalize_transcript,
    word_error_rate,
)


def test_transcript_normalization() -> None:
    assert normalize_transcript("Reset my PIN, please!") == "reset my pin please"


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected_errors", "expected_rate"),
    [
        ("reset my pin", "reset my pin", 0, 0.0),
        ("reset my pin", "reset pin", 1, 1 / 3),
        ("reset my pin", "reset the pin", 1, 1 / 3),
        ("reset my pin", "please reset my pin", 1, 1 / 3),
        ("", "unexpected", 1, 1.0),
    ],
)
def test_word_error_rate(
    reference: str,
    hypothesis: str,
    expected_errors: int,
    expected_rate: float,
) -> None:
    result = word_error_rate(reference, hypothesis)

    assert result.errors == expected_errors
    assert result.rate == pytest.approx(expected_rate)


def test_character_error_rate_ignores_spaces_and_case() -> None:
    result = character_error_rate("PIN Reset", "pin rest")

    assert result.reference_units == 8
    assert result.errors == 1
    assert result.rate == pytest.approx(1 / 8)
