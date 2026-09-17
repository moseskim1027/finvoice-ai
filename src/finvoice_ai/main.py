from fastapi import FastAPI

from finvoice_ai import __version__
from finvoice_ai.api.routes import router
from finvoice_ai.api.streaming import router as streaming_router
from finvoice_ai.config import get_settings
from finvoice_ai.speech.asr import build_transcription_provider
from finvoice_ai.streaming.session import StreamingSessionManager
from finvoice_ai.streaming.transcription import OfflineStreamingTranscriptionAdapter


def create_app() -> FastAPI:
    application = FastAPI(
        title="FinVoice AI",
        summary="A safe financial-support voice and text agent research platform.",
        version=__version__,
    )
    application.include_router(router)
    application.include_router(streaming_router)
    settings = get_settings()
    application.state.streaming_manager = StreamingSessionManager(
        lambda: OfflineStreamingTranscriptionAdapter(build_transcription_provider(settings))
    )
    return application


app = create_app()
