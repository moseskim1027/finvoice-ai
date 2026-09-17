import json
from collections import Counter
from pathlib import Path

TAXONOMY = (
    Path(__file__).parents[1]
    / "src"
    / "finvoice_ai"
    / "evaluation"
    / "data"
    / "bilingual_benchmark_taxonomy.json"
)


def test_taxonomy_balances_languages_and_intents_without_sensitive_examples() -> None:
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    templates = taxonomy["templates"]

    assert Counter(template["language"] for template in templates) == {
        "en": 5,
        "fil": 5,
        "en-fil": 5,
    }
    assert Counter(template["intent_id"] for template in templates) == {
        intent["intent_id"]: 3 for intent in taxonomy["intents"]
    }
    assert {template["language_mode"] for template in templates} == {
        "monolingual",
        "code-switched",
    }
    forbidden = ("DEMO-", "@", "+63", "account number")
    assert not any(token in template["text"] for template in templates for token in forbidden)


def test_escalation_labels_have_explicit_policy_basis() -> None:
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))

    assert all(intent["policy_basis"] for intent in taxonomy["intents"])
    assert {
        intent["intent_id"] for intent in taxonomy["intents"] if intent["escalation_required"]
    } == {"card_security", "money_transfer", "unrelated_request"}
