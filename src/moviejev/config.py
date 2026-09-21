from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    # Required when the key is not scoped to a workspace (API returns 400 otherwise).
    anthropic_workspace_id: str | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"

    typesafe_api_key: SecretStr | None = None
    typesafe_base_url: HttpUrl = HttpUrl("https://api.typesafe.ai")
    typesafe_model: str = "jev-latest"

    tmdb_api_key: SecretStr | None = None
    tmdb_base_url: HttpUrl = HttpUrl("https://api.themoviedb.org/3")

    reranker: Literal["jev", "llm_judge", "none"] = "jev"
    candidates: int = Field(default=15, ge=3, le=40)
    top_k: int = Field(default=5, ge=1, le=20)
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    request_timeout_s: float = Field(default=20.0, gt=0, le=120)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
