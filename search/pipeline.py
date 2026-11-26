"""Knowledge-base retrieval used by both the plain stream and the agent:
hybrid retrieval (dense + BM25, RRF) followed by an optional cross-encoder
rerank that degrades to RRF order on timeout or inference error."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from opentelemetry.trace import get_current_span

from backend.core import metrics
from backend.core.config import Settings
from backend.core.models import SearchHit
from model_server.inference_client import InferenceClient, InferenceError
from search.evidence import exact_identifiers, matches_requested_environment, setting_conflict
from search.hybrid_retriever import HybridRetriever
from search.ltr import features as ltr_features

if TYPE_CHECKING:
    from search.ltr import LtrModel

log = logging.getLogger(__name__)


@dataclass
class KnowledgeResult:
    hits: list[SearchHit]
    rerank_scores: list[float | None]
    reranked: bool
    rerank_skipped: str | None  # "timeout" | "error" | None
    retrieval_ms: float
    conflict: str = ""
    rerank_ms: float = 0.0
    ltr_applied: bool = False
    ltr_skipped: str | None = None  # "needs_rerank_scores" | None
    ltr_ms: float = 0.0

    def sources(self) -> list[dict]:
        return [source_dict(h, r) for h, r in zip(self.hits, self.rerank_scores, strict=True)]


def _diverse(
    hits: list[SearchHit], scores: list[float | None], top_k: int, per_doc: int | None
) -> list[tuple[SearchHit, float | None]]:
    """Take the top_k in ranked order, at most per_doc chunks from any one document.

    When the cap leaves slots empty (few distinct documents), the skipped chunks
    fill them in ranked order, so the context never shrinks because of the cap.
    """
    ranked = list(zip(hits, scores, strict=True))
    if per_doc is None:
        return ranked[:top_k]
    taken: list[tuple[SearchHit, float | None]] = []
    skipped: list[tuple[SearchHit, float | None]] = []
    per: dict[str, int] = {}
    for hit, score in ranked:
        if len(taken) == top_k:
            break
        doc = hit.chunk.doc_id
        if per.get(doc, 0) < per_doc:
            per[doc] = per.get(doc, 0) + 1
            taken.append((hit, score))
        else:
            skipped.append((hit, score))
    return taken + skipped[: top_k - len(taken)]


def source_dict(hit: SearchHit, rerank_score: float | None) -> dict:
    c = hit.chunk
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
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
    *,
    evidence_question: str | None = None,
    ltr: LtrModel | None = None,
) -> KnowledgeResult:
    evidence_question = evidence_question or query
    anchors = set(exact_identifiers(evidence_question))
    missing = anchors.difference(exact_identifiers(query))
    if missing:
        query = f"{query} {' '.join(sorted(missing))}"
    # Any ranking stage, and the per-document cap, needs a pool deeper than what it returns.
    ranking = settings.rerank_enabled or ltr is not None
    deep = ranking or settings.kb_max_chunks_per_doc is not None
    depth = max(top_k, settings.rerank_candidates) if deep else top_k
    # Inspect the bounded hybrid candidate pool before discarding evidence.
    result = await retriever.search(query, max(depth, 2 * settings.retrieval_candidates))
    candidates = [
        hit for hit in result.hits if matches_requested_environment(evidence_question, hit.chunk)
    ]
    if conflict := setting_conflict(evidence_question, [h.chunk for h in candidates]):
        message, ids = conflict
        by_id = {hit.chunk.chunk_id: hit for hit in candidates}
        return KnowledgeResult(
            [by_id[chunk_id] for chunk_id in ids],
            [None] * len(ids),
            False,
            None,
            result.timings.total_ms,
            conflict=message,
        )

    protected: list[str] = []
    if anchors:
        matched = [
            (
                hit,
                anchors.intersection(
                    exact_identifiers(
                        f"{hit.chunk.document_title} {hit.chunk.section_heading} {hit.chunk.text}"
                    )
                ),
            )
            for hit in candidates
        ]
        # Prefer coverage of multiple requested identifiers, then stronger BM25
        # evidence. Spend at most half the context on lexical reservations.
        matched.sort(key=lambda pair: (-len(pair[1]), pair[0].sparse_rank or float("inf")))
        for hit, terms in matched:
            if terms & anchors:
                protected.append(hit.chunk.chunk_id)
                anchors.difference_update(terms)
                if not anchors or len(protected) >= max(1, top_k // 2):
                    break
    elif settings.rerank_enabled:
        # A broad query can still have strong sparse evidence in the fused window.
        protected = [h.chunk.chunk_id for h in candidates[:top_k] if h.sparse_rank == 1]

    by_id = {hit.chunk.chunk_id: hit for hit in candidates}
    required = set(protected)
    capacity = depth - len(protected)
    eligible = [h for h in candidates if h.chunk.chunk_id not in required][:capacity]
    rerank_ids = required | {hit.chunk.chunk_id for hit in eligible}
    hits = [hit for hit in candidates if hit.chunk.chunk_id in rerank_ids]
    scores: list[float | None] = [None] * len(hits)
    reranked, skipped = False, None
    t_rerank = time.perf_counter()
    if settings.rerank_enabled and hits:
        try:
            raw = await asyncio.wait_for(
                inference.rerank(query, [h.chunk.text for h in hits]), settings.rerank_timeout_s
            )
            if len(raw) != len(hits):
                raise InferenceError("reranker returned the wrong number of scores")
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
    rerank_ms = (time.perf_counter() - t_rerank) * 1000 if settings.rerank_enabled else 0.0

    # Learning-to-rank reorders the reranked (or RRF-ordered) candidate pool.
    # A model trained on cross-encoder features cannot run without them.
    ltr_applied, ltr_skipped, ltr_ms = False, None, 0.0
    if ltr is not None and hits:
        if ltr.uses_ce and not reranked:
            ltr_skipped = "needs_rerank_scores"
            metrics.LTR_SKIPPED.labels(ltr_skipped).inc()
        else:
            t_ltr = time.perf_counter()
            learned = ltr.score(ltr_features(query, hits, scores if reranked else None))
            order = np.argsort(-learned, kind="stable").tolist()
            hits = [hits[i] for i in order]
            scores = [scores[i] for i in order]
            ltr_applied = True
            ltr_ms = (time.perf_counter() - t_ltr) * 1000
    selected = _diverse(hits, scores, top_k, settings.kb_max_chunks_per_doc)
    selected_ids = {hit.chunk.chunk_id for hit, _ in selected}
    ranked_scores = {hit.chunk.chunk_id: score for hit, score in zip(hits, scores, strict=True)}
    for chunk_id in protected:
        if chunk_id not in selected_ids:
            slot = next(
                i
                for i in range(len(selected) - 1, -1, -1)
                if selected[i][0].chunk.chunk_id not in required
            )
            selected.pop(slot)
            selected.append((by_id[chunk_id], ranked_scores[chunk_id]))
    return KnowledgeResult(
        hits=[hit for hit, _ in selected],
        rerank_scores=[score for _, score in selected],
        reranked=reranked,
        rerank_skipped=skipped,
        retrieval_ms=result.timings.total_ms,
        rerank_ms=round(rerank_ms, 2),
        ltr_applied=ltr_applied,
        ltr_skipped=ltr_skipped,
        ltr_ms=round(ltr_ms, 3),
    )
