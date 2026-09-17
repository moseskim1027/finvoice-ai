import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorRate:
    errors: int
    reference_units: int

    @property
    def rate(self) -> float:
        return self.errors / max(1, self.reference_units)


def normalize_transcript(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return " ".join(tokens)


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    reference_words = normalize_transcript(reference).split()
    hypothesis_words = normalize_transcript(hypothesis).split()
    return ErrorRate(
        errors=_levenshtein_distance(reference_words, hypothesis_words),
        reference_units=len(reference_words),
    )


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    reference_characters = list(normalize_transcript(reference).replace(" ", ""))
    hypothesis_characters = list(normalize_transcript(hypothesis).replace(" ", ""))
    return ErrorRate(
        errors=_levenshtein_distance(reference_characters, hypothesis_characters),
        reference_units=len(reference_characters),
    )


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_unit in enumerate(reference, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_unit in enumerate(hypothesis, start=1):
            substitution_cost = int(reference_unit != hypothesis_unit)
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[hypothesis_index] + 1,
                    previous_row[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]
