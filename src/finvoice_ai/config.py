from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from FINVOICE-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FINVOICE_",
        extra="ignore",
    )

    environment: str = "local"
    log_level: str = "INFO"
    minimum_response_confidence: float = Field(default=0.70, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
