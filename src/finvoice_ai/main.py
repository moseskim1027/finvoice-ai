from fastapi import FastAPI

from finvoice_ai import __version__
from finvoice_ai.api.routes import router


def create_app() -> FastAPI:
    application = FastAPI(
        title="FinVoice AI",
        summary="A safe financial-support voice and text agent research platform.",
        version=__version__,
    )
    application.include_router(router)
    return application


app = create_app()
