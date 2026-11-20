"""Parallel hybrid retrieval: Qdrant (dense ANN) + BM25 (sparse), fused with RRF.

Reciprocal Rank Fusion (Cormack et al., 2009):

    score(d) = sum over retrievers r of  1 / (k + rank_r(d))

where rank_r(d) is the 1-based position of d in retriever r's list, and d
contributes nothing for a retriever that did not return it. RRF only uses
ranks, so it needs no normalisation between cosine and BM25 scores.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from pydantic import BaseModel

from backend.core.models import Chunk, SearchHit
from backend.core.stores import DenseStore, SparseIndex
from backend.ingestion.embedder import Embedder

log = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Returns (id, score) sorted by score, then by best rank.

    Duplicate ids within one list only count at their first (best) position.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in ranked_lists:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]]))


class RetrievalTimings(BaseModel):
    dense_ms: float
    sparse_ms: float
    fusion_ms: float
    total_ms: float


class HybridResult(BaseModel):
    hits: list[SearchHit]
    timings: RetrievalTimings


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        dense: DenseStore,
        sparse: SparseIndex,
        rrf_k: int = DEFAULT_RRF_K,
        candidates: int = 50,
    ):
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.candidates = candidates

    async def _dense(self, query: str) -> tuple[list[tuple[Chunk, float]], float]:
        t0 = time.perf_counter()
        [vector] = await self.embedder.embed([query])
        hits = await self.dense.search(vector, self.candidates)
        return hits, (time.perf_counter() - t0) * 1000

    async def _sparse(self, query: str) -> tuple[list[tuple[Chunk, float]], float]:
        t0 = time.perf_counter()
        hits = await self.sparse.search(query, self.candidates)
        return hits, (time.perf_counter() - t0) * 1000

    async def search(self, query: str, top_k: int = 10) -> HybridResult:
        t0 = time.perf_counter()
        (dense_hits, dense_ms), (sparse_hits, sparse_ms) = await asyncio.gather(
            self._dense(query), self._sparse(query)
        )

        t_fuse = time.perf_counter()
        dense_ids = [c.chunk_id for c, _ in dense_hits]
        sparse_ids = [c.chunk_id for c, _ in sparse_hits]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k)[:top_k]

        chunks = {c.chunk_id: c for c, _ in sparse_hits} | {c.chunk_id: c for c, _ in dense_hits}
        dense_rank = {cid: i for i, cid in enumerate(dense_ids, start=1)}
        sparse_rank = {cid: i for i, cid in enumerate(sparse_ids, start=1)}
        dense_score = {c.chunk_id: score for c, score in dense_hits}
        sparse_score = {c.chunk_id: score for c, score in sparse_hits}
        hits = [
            SearchHit(
                chunk=chunks[cid],
                score=score,
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
                dense_score=dense_score.get(cid),
                sparse_score=sparse_score.get(cid),
            )
            for cid, score in fused
        ]
        fusion_ms = (time.perf_counter() - t_fuse) * 1000

        timings = RetrievalTimings(
            dense_ms=round(dense_ms, 2),
            sparse_ms=round(sparse_ms, 2),
            fusion_ms=round(fusion_ms, 3),
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        log.info("hybrid search q=%r hits=%d %s", query, len(hits), timings.model_dump())
        return HybridResult(hits=hits, timings=timings)
