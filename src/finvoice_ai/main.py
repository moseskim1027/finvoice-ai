import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from finvoice_ai import __version__
from finvoice_ai.api.demo_tools import router as demo_tools_router
from finvoice_ai.api.routes import router
from finvoice_ai.api.streaming import router as streaming_router
from finvoice_ai.config import get_settings
from finvoice_ai.observability import (
    METRICS,
    BoundedStreamingMetrics,
    configure_logging,
    configure_tracing,
    new_request_id,
    operation,
    request_id_context,
)
from finvoice_ai.speech.asr import build_transcription_provider
from finvoice_ai.streaming.session import StreamingSessionManager
from finvoice_ai.streaming.transcription import OfflineStreamingTranscriptionAdapter


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing()
    application = FastAPI(
        title="FinVoice AI",
        summary="A safe financial-support voice and text agent research platform.",
        version=__version__,
    )
    application.include_router(router)
    application.include_router(streaming_router)
    application.include_router(demo_tools_router)
    application.mount("/static", StaticFiles(directory="src/finvoice_ai/static"), name="static")

    @application.get("/", include_in_schema=False)
    def demo_console() -> FileResponse:
        """Serve the local-only portfolio simulation console."""
        return FileResponse("src/finvoice_ai/static/index.html")

    @application.get("/lab", include_in_schema=False)
    def model_lab() -> FileResponse:
        """Serve the separate local-model evaluation workspace."""
        return FileResponse("src/finvoice_ai/static/lab.html")

    application.state.streaming_manager = StreamingSessionManager(
        lambda: OfflineStreamingTranscriptionAdapter(build_transcription_provider(settings)),
        metrics=BoundedStreamingMetrics(),
    )

    @application.middleware("http")
    async def observe_request(request: Request, call_next):
        request_id = new_request_id(request.headers.get("x-request-id"))
        token = request_id_context.set(request_id)
        started = time.perf_counter()
        route = request.url.path
        status_code = 500
        try:
            with operation("http.request", method=request.method, route=route):
                response = await call_next(request)
                status_code = response.status_code
        finally:
            duration = time.perf_counter() - started
            labels = {"method": request.method, "route": route, "status": str(status_code)}
            METRICS.increment("finvoice_http_requests", labels)
            METRICS.observe("finvoice_http_request_duration_seconds", duration, labels)
            logging.getLogger("finvoice.http").info(
                "request_completed",
                extra={
                    "safe_fields": {
                        "event": "request_completed",
                        "request_id": request_id,
                        "method": request.method,
                        "route": route,
                        "status_code": status_code,
                        "duration_ms": round(duration * 1000, 3),
                    }
                },
            )
            request_id_context.reset(token)
        response.headers["x-request-id"] = request_id
        return response

    return application


app = create_app()
