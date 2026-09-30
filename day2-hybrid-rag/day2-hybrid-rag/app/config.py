"""Central configuration.

Every tunable value lives here, in ONE place. Values are read (in priority
order) from real environment variables, then from a local ``.env`` file, then
from the defaults below. Nothing else in the project hard-codes chunk sizes,
top-k values, model names or API keys.

Usage::

    from app.config import get_settings
    settings = get_settings()
    print(settings.chunk_size)
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Used when LLM_MODEL is left blank in .env
DEFAULT_LLM_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}


class Settings(BaseSettings):
    """Typed application settings (validated at start-up)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated variables that happen to be in .env
    )

    # --- LLM -----------------------------------------------------------
    llm_provider: Literal["gemini", "openai"] = "gemini"
    google_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = ""  # blank -> DEFAULT_LLM_MODELS[llm_provider]
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- models --------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- chunking ------------------------------------------------------
    chunk_size: int = Field(default=500, ge=50)
    chunk_overlap: int = Field(default=100, ge=0)

    # --- retrieval -----------------------------------------------------
    top_k_vector: int = Field(default=10, ge=1)
    top_k_bm25: int = Field(default=10, ge=1)
    top_k_hybrid: int = Field(default=10, ge=1)
    top_k_rerank: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)

    # --- storage -------------------------------------------------------
    chroma_path: str = "./chroma_db"
    collection_name: str = Field(default="documents", min_length=3)
    documents_dir: str = "./data/documents"

    # --- misc ----------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )
        return self

    # --- derived values ------------------------------------------------
    @property
    def resolved_llm_model(self) -> str:
        """The model name to use: explicit LLM_MODEL or the provider default."""
        return self.llm_model.strip() or DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def active_api_key(self) -> str:
        """The API key belonging to the selected provider."""
        key = self.google_api_key if self.llm_provider == "gemini" else self.openai_api_key
        return key.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object (parsed once, then cached)."""
    return Settings()
