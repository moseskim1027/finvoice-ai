from finvoice_ai.infrastructure.local_providers import TemplateResponseGenerator


def test_template_generator_abstains_without_context() -> None:
    result = TemplateResponseGenerator().generate("Unknown question", [])

    assert result.confidence == 0.0
    assert result.model == "local-template-v1"
