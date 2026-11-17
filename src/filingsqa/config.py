"""Application settings, loaded from environment variables and `.env`.

All runtime configuration lives here so that nothing reads `os.environ` directly.
The generator LLM and the judge LLM are configured independently: the judge
must never silently fall back to the generator's settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SEC EDGAR
    sec_user_agent: str = "FilingsQA your.name@example.com"
    sec_max_requests_per_second: int = Field(default=10, ge=1, le=10)

    # Generator LLM (system under test)
    llm_provider: LLMProvider = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    anthropic_api_key: SecretStr | None = None

    # Judge LLM (evaluation only)
    judge_provider: LLMProvider = "anthropic"
    judge_model: str = "claude-sonnet-5-5"
    judge_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # LLM cache
    llm_cache_enabled: bool = True
    llm_cache_dir: Path = Path(".cache/llm")

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "filings"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    @field_validator("sec_user_agent")
    @classmethod
    def _user_agent_has_contact(cls, value: str) -> str:
        # SEC blocks anonymous clients; the User-Agent must contain a contact email.
        if "@" not in value:
            raise ValueError("SEC_USER_AGENT must contain a contact email address")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
