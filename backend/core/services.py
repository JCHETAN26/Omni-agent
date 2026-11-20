"""Builds and owns every long-lived dependency of the API process."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from backend.core.cache import Cache, cache_from_settings
from backend.core.config import Settings
from backend.core.rate_limit import RateLimiter, rate_limiter_from_settings
from backend.core.stores import (
    DenseStore,
    SparseIndex,
    dense_store_from_settings,
    sparse_index_from_settings,
)
from backend.ingestion.embedder import Embedder, embedder_from_settings
from backend.ingestion.processor import IngestionProcessor
from backend.vision.preprocess import Preprocessor
from model_server.inference_client import (
    InferenceClient,
    TritonEmbedder,
    inference_client_from_settings,
)
from model_server.vision_client import VisionClient, vision_client_from_settings
from search.hybrid_retriever import HybridRetriever
from search.ltr import LtrModel


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    dense: DenseStore
    sparse: SparseIndex
    cache: Cache
    inference: InferenceClient
    rate_limiter: RateLimiter
    processor: IngestionProcessor
    retriever: HybridRetriever
    ltr: LtrModel | None = None  # learning-to-rank stage; None disables it
    vision: VisionClient | None = None  # live-frame classifier; None disables /ws/frames
    preprocessor: Preprocessor | None = None
    index_version: int = 0  # bumped on ingest to invalidate cached searches
    index_writes: int = 0
    search_cache_namespace: str = field(default_factory=lambda: uuid.uuid4().hex)
    backends: dict[str, str] = field(default_factory=dict)

    @classmethod
    async def build(cls, s: Settings) -> Services:
        inference = inference_client_from_settings(s)
        await inference.warmup()
        embedder = (
            TritonEmbedder(inference)
            if s.embedding_provider == "triton"
            else (embedder_from_settings(s))
        )
        if s.embedding_provider == "sentence-transformers":
            await embedder.embed(["warm-up"])  # load model weights before serving traffic
        dense = dense_store_from_settings(s)
        sparse = sparse_index_from_settings(s)
        services = cls(
            settings=s,
            embedder=embedder,
            dense=dense,
            sparse=sparse,
            cache=cache_from_settings(s),
            inference=inference,
            rate_limiter=rate_limiter_from_settings(s),
            processor=IngestionProcessor(embedder, dense, sparse, s.chunk_size, s.chunk_overlap),
            retriever=HybridRetriever(embedder, dense, sparse, s.rrf_k, s.retrieval_candidates),
            ltr=LtrModel.load(Path(s.ltr_model_path)) if s.ltr_model_path else None,
            vision=vision_client_from_settings(s),
            preprocessor=Preprocessor() if s.vision_backend != "off" else None,
        )
        if services.ltr is not None and services.ltr.uses_ce and not s.rerank_enabled:
            raise ValueError("LTR_MODEL_PATH needs cross-encoder scores; set RERANK_ENABLED=true")
        services.backends = {
            "dense": (
                "faiss-hnsw"
                if s.dense_backend == "faiss"
                else ("qdrant" if s.qdrant_url else "qdrant-in-memory")
            ),
            "sparse": "elasticsearch" if s.elasticsearch_url else "bm25-in-memory",
            "cache": "redis" if s.redis_url else "in-memory",
            "rate_limiter": "redis" if s.redis_url else "in-memory",
            "embedder": embedder.name,
            "inference": inference.name,
            "ranker": (
                ("ltr+cross-encoder" if services.ltr.uses_ce else "ltr")
                if services.ltr is not None
                else ("cross-encoder" if s.rerank_enabled else "rrf")
            ),
            "vision": (
                f"{services.vision.name} + {services.preprocessor.name}"
                if services.vision is not None and services.preprocessor is not None
                else "off"
            ),
        }
        return services

    async def close(self) -> None:
        await self.dense.close()
        await self.sparse.close()
        await self.cache.close()
        await self.rate_limiter.close()
        await self.inference.close()
        if self.vision is not None:
            await self.vision.close()
