"""Knowledge-base retrieval used by both the plain stream and the agent:
hybrid retrieval (dense + BM25, RRF) followed by an optional cross-encoder
rerank that degrades to RRF order on timeout or inference error."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from opentelemetry.trace import get_current_span

from backend.core import metrics
from backend.core.config import Settings
from backend.core.models import SearchHit
from model_server.inference_client import InferenceClient, InferenceError
from search.hybrid_retriever import HybridRetriever

log = logging.getLogger(__name__)


@dataclass
class KnowledgeResult:
    hits: list[SearchHit]
    rerank_scores: list[float | None]
    reranked: bool
    rerank_skipped: str | None  # "timeout" | "error" | None
    retrieval_ms: float

    def sources(self) -> list[dict]:
        return [source_dict(h, r) for h, r in zip(self.hits, self.rerank_scores, strict=True)]


def source_dict(hit: SearchHit, rerank_score: float | None) -> dict:
    c = hit.chunk
    return {
        "chunk_id": c.chunk_id,
        "document_title": c.document_title,
        "section_heading": c.section_heading,
        "rrf_score": round(hit.score, 6),
        "rerank_score": None if rerank_score is None else round(rerank_score, 6),
        "dense_rank": hit.dense_rank,
        "sparse_rank": hit.sparse_rank,
    }


async def retrieve_with_rerank(
    retriever: HybridRetriever,
    inference: InferenceClient,
    settings: Settings,
    query: str,
    top_k: int,
    request_id: str | None = None,
) -> KnowledgeResult:
    depth = max(top_k, settings.rerank_candidates) if settings.rerank_enabled else top_k
    result = await retriever.search(query, depth)
    hits = result.hits
    scores: list[float | None] = [None] * len(hits)
    reranked, skipped = False, None
    if settings.rerank_enabled and hits:
        try:
            raw = await asyncio.wait_for(
                inference.rerank(query, [h.chunk.text for h in hits]), settings.rerank_timeout_s
            )
            order = sorted(range(len(hits)), key=lambda i: raw[i], reverse=True)
            hits = [hits[i] for i in order]
            scores = [raw[i] for i in order]
            reranked = True
        except (TimeoutError, InferenceError) as exc:
            # Reranking is an enhancement: degrade to RRF order, never fail.
            skipped = "timeout" if isinstance(exc, TimeoutError) else "error"
            metrics.RERANK_SKIPPED.labels(skipped).inc()
            get_current_span().add_event("rerank.skipped", {"reason": skipped})
            log.warning(
                "rerank skipped (%s), using RRF order", skipped, extra={"request_id": request_id}
            )
    return KnowledgeResult(
        hits=hits[:top_k],
        rerank_scores=scores[:top_k],
        reranked=reranked,
        rerank_skipped=skipped,
        retrieval_ms=result.timings.total_ms,
    )
