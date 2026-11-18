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
    embedding_provider: Literal["sentence-transformers", "openai", "hashing", "triton"] = (
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
    rerank_enabled: bool = False  # cross-encoder rerank via the inference client
    rerank_candidates: int = 10  # fused hits sent to the reranker
    rerank_timeout_s: float = 1.5  # budget; on timeout/error fall back to RRF order

    # --- Model inference (Triton) ---
    inference_backend: Literal["mock", "triton"] = "mock"
    triton_grpc_url: str = "localhost:8001"
    triton_embed_model: str = "minilm_embedder"
    triton_rerank_model: str = "cross_encoder_reranker"
    triton_embed_tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"
    triton_rerank_tokenizer: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    triton_timeout_s: float = 10.0
    triton_max_in_flight: int = 128

    # --- Agent (LangGraph) ---
    # Reasoning backend for routing, grading, rewriting and generation.
    # "heuristic" runs offline (no LLM); "anthropic" uses Claude; "openai" is any
    # OpenAI-compatible endpoint (OpenAI, vLLM, ...).
    llm_provider: Literal["heuristic", "anthropic", "openai"] = "heuristic"
    anthropic_model: str = "claude-opus-5-5"
    anthropic_api_key: str | None = None  # unset: SDK resolves env / `ant` profile
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_timeout_s: float = 60.0
    agent_max_retries: int = 2  # query rewrite + re-retrieval loops per turn
    agent_relevance_threshold: float = 0.5
    agent_grounding_threshold: float = 0.6
    agent_top_k: int = 4
    agent_history_messages: int = 8  # recent messages given to the LLM
    agent_memory_ttl_s: int = 7 * 24 * 3600
    agent_logs_path: str = "samples/logs"

    # --- WebSocket gateway ---
    ws_max_sessions: int = 1000
    ws_buffer_size: int = 256  # ring buffer slots per session (oldest dropped when full)
    ws_max_message_bytes: int = 1_048_576
    ws_heartbeat_interval_s: float = 15.0
    ws_heartbeat_timeout_s: float = 45.0  # close if nothing heard from the client this long

    # --- Rate limiting (token bucket, per session) ---
    rate_limit_capacity: float = 20.0
    rate_limit_refill_per_s: float = 10.0
    rate_limit_audio_cost: float = 0.05  # audio frames are small and frequent

    # --- Startup ---
    seed_path: str | None = None  # ingest *.md from this directory at startup (demos, dev)

    # --- Observability ---
    service_name: str = "omniagent"
    log_format: Literal["json", "text"] = "json"
    log_level: str = "INFO"
    otel_exporter_otlp_endpoint: str | None = None  # e.g. http://localhost:4317
    otel_console_export: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
