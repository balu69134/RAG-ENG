from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "RAG-ENG"
    app_version: str = "1.0.0"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = Field(default=1024, ge=128, le=8192)
    generation_backend: Literal["ollama", "groq"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = Field(default=300, ge=10, le=1800)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_local_path: str | None = "./qdrant_data"
    qdrant_collection: str = "rag_documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: Literal["auto", "sentence-transformers", "hashed-lexical"] = "auto"
    chunk_size: int = Field(default=1200, ge=200, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    default_top_k: int = Field(default=5, ge=1, le=20)
    max_upload_size_mb: int = Field(default=25, ge=1, le=200)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
