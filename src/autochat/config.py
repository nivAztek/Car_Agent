"""Application settings, loaded once from environment / `.env` (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Microsoft Foundry (OpenAI-compatible v1 endpoint, used verbatim as base_url)
    foundry_endpoint: str = ""
    foundry_api_key: SecretStr | None = None
    foundry_chat_deployment: str = ""
    foundry_fast_deployment: str = ""
    foundry_extract_deployment: str = ""  # optional; falls back to fast -> chat
    foundry_embed_deployment: str = ""
    embed_dimensions: int = 1536

    # Cosmos DB (vNext emulator by default)
    cosmos_endpoint: str = "http://localhost:8081"
    cosmos_key: SecretStr = SecretStr("")
    cosmos_database: str = "autochat"
    cosmos_cars_container: str = "cars"
    cosmos_chunks_container: str = "chunks"
    cosmos_checkpoints_container: str = "checkpoints"

    # App
    api_base_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    data_dir: Path = Field(default=REPO_ROOT / "data")

    @property
    def fast_deployment(self) -> str:
        return self.foundry_fast_deployment or self.foundry_chat_deployment

    @property
    def extract_deployment(self) -> str:
        return self.foundry_extract_deployment or self.fast_deployment

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def urls_file(self) -> Path:
        return self.data_dir / "urls.txt"


@lru_cache
def get_settings() -> Settings:
    return Settings()
