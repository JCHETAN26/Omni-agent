"""Runtime configuration, loaded from environment variables / `.env`.

Every external service has a container-free fallback so the whole stack runs
inside a bare GitHub Codespace (or laptop) with zero managed infrastructure:

* QDRANT_URL unset         -> embedded in-memory Qdrant
* ELASTICSEARCH_URL unset  -> in-process BM25 index
* REDIS_URL unset          -> in-memory TTL cache
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Dense vector store (Qdrant) ---
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "omniagent_chunks"

    # --- Sparse / BM25 index (Elasticsearch) ---
    elasticsearch_url: str | None = None
    elasticsearch_api_key: str | None = None
    elasticsearch_index: str = "omniagent_chunks"

    # --- Cache (Redis) ---
    redis_url: str | None = None
    cache_ttl_seconds: int = 300

    # --- Embeddings ---
    embedding_provider: Literal["sentence-transformers", "openai", "hashing"] = (
        "sentence-transformers"
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Chunking ---
    chunk_size: int = 800  # max characters per chunk
    chunk_overlap: int = 120  # characters carried over between adjacent chunks

    # --- Retrieval ---
    rrf_k: int = 60
    retrieval_candidates: int = 50  # per-retriever depth fed into RRF


@lru_cache
def get_settings() -> Settings:
    return Settings()
