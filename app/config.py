"""
Centralised application configuration.
All values are loaded from environment variables / .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    app_version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── OpenAI ───────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2

    # ── Vector Store ─────────────────────────────────────────────────
    vector_store: Literal["faiss", "pinecone"] = "faiss"
    faiss_index_path: str = "data/faiss_index"

    pinecone_api_key: str = ""
    pinecone_index: str = "rag-index"
    pinecone_environment: str = "us-east-1"

    # ── RAG Parameters ───────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    score_threshold: float = 0.4

    # ── Observability ────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-ai-assistant"
    langchain_tracing_v2: bool = False

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
