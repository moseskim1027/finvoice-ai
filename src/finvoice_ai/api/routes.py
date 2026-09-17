from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from finvoice_ai.application.conversation import ConversationService
from finvoice_ai.config import Settings, get_settings
from finvoice_ai.domain.models import ConversationRequest, ConversationResponse
from finvoice_ai.domain.policy import SupportPolicy
from finvoice_ai.infrastructure.document_loader import load_default_documents
from finvoice_ai.infrastructure.local_providers import (
    InMemoryConversationStore,
    TemplateResponseGenerator,
)
from finvoice_ai.infrastructure.retrieval import BM25Retriever
from finvoice_ai.speech.asr import AsrDependencyError, build_transcription_provider
from finvoice_ai.speech.schemas import (
    SpeechAnalysisResponse,
    SpeechSegmentResponse,
    TranscriptionResponse,
)
from finvoice_ai.speech.service import SpeechService
from finvoice_ai.speech.vad import EnergyVoiceActivityDetector
from finvoice_ai.speech.wav import InvalidAudioError, PcmWavLoader

router = APIRouter()
MAXIMUM_WAV_BYTES = 6_500_000


@router.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/v1/conversations/respond",
    response_model=ConversationResponse,
    tags=["conversations"],
)
def respond(
    request: ConversationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationResponse:
    service = ConversationService(
        policy=SupportPolicy(settings.minimum_response_confidence),
        retriever=BM25Retriever(load_default_documents()),
        generator=TemplateResponseGenerator(),
        store=InMemoryConversationStore(),
    )
    return service.respond(request)


@router.post(
    "/v1/audio/analyze",
    response_model=SpeechAnalysisResponse,
    tags=["speech"],
)
async def analyze_audio(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SpeechAnalysisResponse:
    data = await file.read(MAXIMUM_WAV_BYTES + 1)
    if len(data) > MAXIMUM_WAV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="WAV upload exceeds size limit",
        )

    try:
        service = SpeechService(
            loader=PcmWavLoader(),
            vad=EnergyVoiceActivityDetector(),
            transcriber=build_transcription_provider(settings),
        )
        analysis = service.analyze(data)
    except AsrDependencyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except InvalidAudioError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    return SpeechAnalysisResponse(
        duration_seconds=analysis.duration_seconds,
        segments=[
            SpeechSegmentResponse(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                mean_rms=segment.mean_rms,
            )
            for segment in analysis.segments
        ],
        transcription=TranscriptionResponse(
            text=analysis.transcription.text,
            confidence=analysis.transcription.confidence,
            language=analysis.transcription.language,
            model=analysis.transcription.model,
        ),
    )
