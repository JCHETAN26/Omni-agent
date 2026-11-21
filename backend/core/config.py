"""Runtime configuration, loaded from environment variables / `.env`.

Every external service has a container-free fallback so the whole stack runs
inside a bare GitHub Codespace (or laptop) with zero managed infrastructure:

* QDRANT_URL unset         -> embedded in-memory Qdrant
* ELASTICSEARCH_URL unset  -> in-process BM25 index
* REDIS_URL unset          -> in-memory TTL cache
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TRITON_MAX_IN_FLIGHT = 16


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Dense vector store ---
    # "qdrant" uses QDRANT_URL (or embedded in-memory Qdrant when unset);
    # "faiss" uses an in-process HNSW index, persisted under FAISS_PATH when set.
    dense_backend: Literal["qdrant", "faiss"] = "qdrant"
    faiss_path: str | None = None
    faiss_hnsw_m: int = Field(default=32, ge=4)
    faiss_ef_construction: int = Field(default=200, ge=8)
    faiss_ef_search: int = Field(default=128, ge=8)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "omniagent_chunks"
    # The client's REST default is 5 s, which a wait=True upsert exceeds once the
    # optimizer is busy indexing a few hundred thousand points.
    qdrant_timeout_s: int = Field(default=60, ge=1)
    qdrant_upsert_batch: int = Field(default=512, ge=1)  # points per upsert request
    # Search tuning. "int8" scalar quantization keeps 4x smaller vectors in RAM and
    # rescores the top limit*oversampling hits with full-precision vectors. "none"
    # explicitly bypasses any quantization configured on the collection; "auto"
    # leaves it to the collection's configuration.
    qdrant_quantization: Literal["auto", "none", "int8"] = "auto"
    qdrant_oversampling: float = Field(default=2.0, ge=1.0)
    qdrant_hnsw_ef: int | None = Field(default=None, ge=1)  # None = server default

    # --- Sparse / BM25 index (Elasticsearch) ---
    elasticsearch_url: str | None = None
    elasticsearch_api_key: str | None = None
    elasticsearch_index: str = "omniagent_chunks"
    # The client's default is 10 s; a bulk with refresh=wait_for exceeds it when
    # the data volume stalls on fsync (observed 7-10 s on Codespaces disks).
    elasticsearch_timeout_s: float = Field(default=60.0, gt=0)

    # --- Cache (Redis) ---
    redis_url: str | None = None
    cache_ttl_seconds: int = 300

    # --- Embeddings ---
    embedding_provider: Literal["sentence-transformers", "openai", "hashing", "triton"] = (
        "sentence-transformers"
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str | None = None  # e.g. "cpu", "cuda", "mps"; None = auto
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
    # Learning-to-rank: a trained search/ltr.py model directory (model.pt + model.json).
    # Unset disables the stage. A model trained with cross-encoder features needs
    # RERANK_ENABLED=true and is skipped whenever the reranker falls back.
    ltr_model_path: str | None = None

    # --- Model inference (Triton) ---
    inference_backend: Literal["mock", "triton"] = "mock"
    triton_grpc_url: str = "localhost:8001"
    triton_embed_model: str = "minilm_embedder"
    triton_rerank_model: str = "cross_encoder_reranker"
    triton_embed_tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"
    triton_rerank_tokenizer: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    triton_timeout_s: float = 10.0
    triton_max_in_flight: int = Field(default=DEFAULT_TRITON_MAX_IN_FLIGHT, ge=1)

    # --- Vision (live frames over /ws/frames) ---
    # "triton": ViT on a Triton server (TRITON_GRPC_URL); "mock": deterministic
    # labels with no server (tests, local dev); "off": /ws/frames refuses frames.
    vision_backend: Literal["off", "mock", "triton"] = "off"
    triton_vision_model: str = "vit_b16"  # vit_b16 = dynamic batching; vit_b16_nobatch = off
    vision_top_k: int = Field(default=5, ge=1, le=20)
    vision_timeout_s: float = 5.0
    vision_context_ttl_s: int = 120  # how long a thread's latest frame informs the agent
    vision_max_frame_bytes: int = 1_000_000

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
    # 0 makes identical prompts reproducible (and safely cacheable). Leave unset
    # (LLM_TEMPERATURE=) for reasoning models that reject the parameter.
    llm_temperature: float | None = 0.0
    # Exact-match LLM response cache + single-flight coalescing (backend/agent/llm_cache.py),
    # stored in the shared cache (Redis when REDIS_URL is set).
    llm_cache_enabled: bool = False
    llm_cache_ttl_s: int = Field(default=3600, ge=1)
    agent_max_in_flight: int = Field(default=8, ge=1)
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

    # --- Collaborative incident rooms (/rooms, /ws/rooms/{id}) ---
    rooms_max_events: int = Field(default=1000, ge=10)  # replayable history per room
    rooms_ttl_s: int = Field(default=86_400, ge=60)  # room state expires after this much inactivity
    rooms_queue_size: int = Field(default=512, ge=8)  # per-session send queue (overflow -> resync)
    rooms_presence_ttl_s: float = 45.0  # presence without a heartbeat this long is swept
    rooms_max_sessions: int = Field(default=5000, ge=1)  # per replica
    rooms_msgs_per_s: float = 20.0  # per-session rate limit (burst 2x)
    rooms_redis_max_connections: int = Field(default=64, ge=4)  # per replica (blocking pool)

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
