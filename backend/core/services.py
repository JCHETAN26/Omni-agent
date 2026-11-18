"""Builds and owns every long-lived dependency of the API process."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.core.cache import Cache, cache_from_settings
from backend.core.config import Settings
from backend.core.rate_limit import RateLimiter, rate_limiter_from_settings
from backend.core.stores import QdrantDenseStore, SparseIndex, sparse_index_from_settings
from backend.ingestion.embedder import Embedder, embedder_from_settings
from backend.ingestion.processor import IngestionProcessor
from model_server.inference_client import (
    InferenceClient,
    TritonEmbedder,
    inference_client_from_settings,
)
from search.hybrid_retriever import HybridRetriever


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    dense: QdrantDenseStore
    sparse: SparseIndex
    cache: Cache
    inference: InferenceClient
    rate_limiter: RateLimiter
    processor: IngestionProcessor
    retriever: HybridRetriever
    index_version: int = 0  # bumped on ingest to invalidate cached searches
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
        dense = QdrantDenseStore.from_settings(s)
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
        )
        services.backends = {
            "dense": "qdrant" if s.qdrant_url else "qdrant-in-memory",
            "sparse": "elasticsearch" if s.elasticsearch_url else "bm25-in-memory",
            "cache": "redis" if s.redis_url else "in-memory",
            "rate_limiter": "redis" if s.redis_url else "in-memory",
            "embedder": embedder.name,
            "inference": inference.name,
        }
        return services

    async def close(self) -> None:
        await self.dense.close()
        await self.sparse.close()
        await self.cache.close()
        await self.rate_limiter.close()
        await self.inference.close()
