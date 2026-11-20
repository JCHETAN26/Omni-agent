"""Asynchronous ingestion: chunk -> (embed -> Qdrant) || (BM25 index).

The sparse branch does not depend on embeddings, so it runs concurrently
with embedding generation and the dense upsert.
"""

from __future__ import annotations

import asyncio
import logging
import time

from pydantic import BaseModel

from backend.core.models import Chunk, Document
from backend.core.stores import DenseStore, SparseIndex
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import Embedder

log = logging.getLogger(__name__)


class IngestReport(BaseModel):
    documents: int
    chunks: int
    embedding_dim: int
    chunk_ms: float
    dense_ms: float
    sparse_ms: float
    total_ms: float


class IngestionProcessor:
    def __init__(
        self,
        embedder: Embedder,
        dense: DenseStore,
        sparse: SparseIndex,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        embed_batch_size: int = 128,
    ):
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_batch_size = embed_batch_size
        self._dense_ready = False
        self._sparse_ready = False

    async def _dense_branch(self, chunks: list[Chunk]) -> tuple[int, float]:
        t0 = time.perf_counter()
        dim = 0
        for i in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[i : i + self.embed_batch_size]
            vectors = await self.embedder.embed([c.text for c in batch])
            dim = len(vectors[0])
            if not self._dense_ready:
                await self.dense.ensure_collection(dim)
                self._dense_ready = True
            await self.dense.upsert(batch, vectors)
        return dim, (time.perf_counter() - t0) * 1000

    async def _sparse_branch(self, chunks: list[Chunk]) -> float:
        t0 = time.perf_counter()
        if not self._sparse_ready:
            await self.sparse.ensure_index()
            self._sparse_ready = True
        await self.sparse.index(chunks)
        return (time.perf_counter() - t0) * 1000

    async def ingest(self, documents: list[Document]) -> IngestReport:
        t0 = time.perf_counter()
        chunks = [
            c for d in documents for c in chunk_document(d, self.chunk_size, self.chunk_overlap)
        ]
        chunk_ms = (time.perf_counter() - t0) * 1000
        if not chunks:
            return IngestReport(
                documents=len(documents),
                chunks=0,
                embedding_dim=0,
                chunk_ms=chunk_ms,
                dense_ms=0,
                sparse_ms=0,
                total_ms=chunk_ms,
            )

        dense_result, sparse_result = await asyncio.gather(
            self._dense_branch(chunks), self._sparse_branch(chunks), return_exceptions=True
        )
        # A failed branch may leave visible writes. Let both finish before the API
        # allows its search cache to be populated again.
        if isinstance(dense_result, BaseException):
            raise dense_result
        if isinstance(sparse_result, BaseException):
            raise sparse_result
        dim, dense_ms = dense_result
        sparse_ms = sparse_result
        report = IngestReport(
            documents=len(documents),
            chunks=len(chunks),
            embedding_dim=dim,
            chunk_ms=round(chunk_ms, 2),
            dense_ms=round(dense_ms, 2),
            sparse_ms=round(sparse_ms, 2),
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        log.info("ingested %s", report.model_dump())
        return report
