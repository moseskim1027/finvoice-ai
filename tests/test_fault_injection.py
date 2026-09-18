from fastapi.testclient import TestClient

from finvoice_ai.application.conversation import ConversationService
from finvoice_ai.domain.models import ConversationRequest, Decision
from finvoice_ai.domain.policy import SupportPolicy
from finvoice_ai.infrastructure.document_loader import load_default_documents
from finvoice_ai.infrastructure.local_providers import (
    InMemoryConversationStore,
    TemplateResponseGenerator,
)
from finvoice_ai.infrastructure.retrieval import BM25Retriever
from finvoice_ai.main import app
from finvoice_ai.resilience import (
    FaultInjectingGenerator,
    FaultInjectingRetriever,
    FaultInjectingTranscriber,
)
from finvoice_ai.speech.providers import DeterministicTranscriptionProvider
from finvoice_ai.tools.authorization import AuthenticationRequiredError
from finvoice_ai.tools.demo_tools import DemoSupportRepository, build_demo_tool_definitions
from finvoice_ai.tools.gateway import ToolGateway
from finvoice_ai.tools.models import ToolContext

from .audio_helpers import create_wav


def _request() -> ConversationRequest:
    return ConversationRequest(
        session_id="fault-test",
        message="How do I reset my PIN?",
        confidence=0.95,
    )


def test_retriever_and_generator_faults_escalate_safely() -> None:
    documents = load_default_documents()
    for retriever, generator in (
        (
            FaultInjectingRetriever(BM25Retriever(documents), unavailable=True),
            TemplateResponseGenerator(),
        ),
        (
            BM25Retriever(documents),
            FaultInjectingGenerator(TemplateResponseGenerator(), unavailable=True),
        ),
    ):
        response = ConversationService(
            SupportPolicy(0.7),
            retriever,
            generator,
            InMemoryConversationStore(),
        ).respond(_request())
        assert response.decision is Decision.ESCALATE
        assert response.reason == "provider_unavailable"


def test_asr_timeout_and_malformed_audio_return_traceable_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "finvoice_ai.api.routes.build_transcription_provider",
        lambda _settings: FaultInjectingTranscriber(
            DeterministicTranscriptionProvider(), timeout=True
        ),
    )
    client = TestClient(app)

    timeout = client.post(
        "/v1/audio/analyze",
        files={"file": ("speech.wav", create_wav(), "audio/wav")},
        headers={"x-request-id": "asr-timeout"},
    )
    malformed = client.post(
        "/v1/audio/analyze",
        files={"file": ("bad.wav", b"not wav", "audio/wav")},
        headers={"x-request-id": "bad-audio"},
    )

    assert timeout.status_code == 503
    assert timeout.headers["x-request-id"] == "asr-timeout"
    assert malformed.status_code == 422
    assert malformed.headers["x-request-id"] == "bad-audio"


def test_mcp_denial_is_audited() -> None:
    gateway = ToolGateway(build_demo_tool_definitions(DemoSupportRepository.create()))

    try:
        gateway.execute(
            "get_demo_account_status",
            {"account_id": "DEMO-001"},
            ToolContext(principal_id="anonymous", authenticated=False),
        )
    except AuthenticationRequiredError:
        pass
    else:
        raise AssertionError("unauthenticated MCP call was not denied")

    assert gateway.audit_events[-1].outcome == "denied_or_failed"
