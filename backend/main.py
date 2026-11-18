"""FastAPI gateway: health, ingestion and hybrid search endpoints.

Run:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.core.cache import cache_from_settings
from backend.core.config import get_settings
from backend.core.models import Document
from backend.core.stores import QdrantDenseStore, sparse_index_from_settings
from backend.ingestion.embedder import embedder_from_settings
from backend.ingestion.processor import IngestionProcessor, IngestReport
from search.hybrid_retriever import HybridResult, HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    embedder = embedder_from_settings(s)
    dense = QdrantDenseStore.from_settings(s)
    sparse = sparse_index_from_settings(s)
    cache = cache_from_settings(s)
    if s.embedding_provider == "sentence-transformers":
        await embedder.embed(["warm-up"])  # load model weights before serving traffic

    app.state.cache = cache
    app.state.processor = IngestionProcessor(embedder, dense, sparse, s.chunk_size, s.chunk_overlap)
    app.state.retriever = HybridRetriever(embedder, dense, sparse, s.rrf_k, s.retrieval_candidates)
    app.state.backends = {
        "dense": "qdrant" if s.qdrant_url else "qdrant-in-memory",
        "sparse": "elasticsearch" if s.elasticsearch_url else "bm25-in-memory",
        "cache": "redis" if s.redis_url else "in-memory",
        "embedder": embedder.name,
    }
    app.state.index_version = 0  # bumped on ingest to invalidate cached searches
    yield
    await dense.close()
    await sparse.close()
    await cache.close()


app = FastAPI(title="OmniAgent", version="0.1.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backends": app.state.backends}


@app.post("/ingest", response_model=IngestReport)
async def ingest(req: IngestRequest) -> IngestReport:
    report = await app.state.processor.ingest(req.documents)
    app.state.index_version += 1
    return report


@app.post("/search", response_model=HybridResult)
async def search(req: SearchRequest) -> HybridResult:
    digest = hashlib.sha256(f"{req.top_k}:{req.query}".encode()).hexdigest()[:24]
    key = f"search:v{app.state.index_version}:{digest}"
    if (cached := await app.state.cache.get(key)) is not None:
        return HybridResult.model_validate(cached)
    result = await app.state.retriever.search(req.query, req.top_k)
    await app.state.cache.set(key, result.model_dump(), get_settings().cache_ttl_seconds)
    return result
