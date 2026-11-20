from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.models import Document, SearchHit
from backend.ingestion.chunker import chunk_document
from search import ltr
from search.hybrid_retriever import HybridResult, RetrievalTimings
from search.pipeline import retrieve_with_rerank
from tests.conftest import load_samples, make_settings


def _retriever(n=6):
    hits = []
    for i in range(1, n + 1):
        (chunk,) = chunk_document(Document(doc_id=f"d{i}", title=f"Doc {i}", text=f"body {i}"))
        hits.append(SearchHit(chunk=chunk, score=1 / i, sparse_rank=i, sparse_score=10.0 / i))
    result = HybridResult(
        hits=hits, timings=RetrievalTimings(dense_ms=0, sparse_ms=0, fusion_ms=0, total_ms=1.0)
    )
    return SimpleNamespace(search=AsyncMock(return_value=result))


class ReverseRanker:
    """Scores the RRF order backwards, so applying it is observable."""

    def __init__(self, uses_ce: bool):
        self.uses_ce = uses_ce
        self.seen = None

    def score(self, x: np.ndarray) -> np.ndarray:
        self.seen = x
        return -x[:, ltr.FEATURES.index("rrf_norm")]


async def test_ltr_reorders_rrf_candidates_without_a_reranker():
    model = ReverseRanker(uses_ce=False)
    kb = await retrieve_with_rerank(_retriever(), None, make_settings(), "body", 3, ltr=model)
    assert kb.ltr_applied and kb.ltr_skipped is None and not kb.reranked
    assert [h.chunk.doc_id for h in kb.hits] == ["d6", "d5", "d4"]
    assert model.seen.shape == (6, len(ltr.FEATURES))


async def test_ltr_sees_cross_encoder_scores_after_rerank():
    model = ReverseRanker(uses_ce=True)
    inference = SimpleNamespace(
        rerank=AsyncMock(side_effect=lambda _, p: [float(len(x)) for x in p])
    )
    kb = await retrieve_with_rerank(
        _retriever(),
        inference,
        make_settings(rerank_enabled=True, rerank_candidates=6),
        "body",
        3,
        ltr=model,
    )
    assert kb.reranked and kb.ltr_applied
    assert (model.seen[:, ltr.FEATURES.index("ce_score")] > 0).all()
    assert kb.rerank_ms >= 0 and kb.ltr_ms >= 0


async def test_ce_model_is_skipped_when_the_reranker_falls_back():
    model = ReverseRanker(uses_ce=True)

    async def slow(*_):
        raise TimeoutError

    kb = await retrieve_with_rerank(
        _retriever(),
        SimpleNamespace(rerank=slow),
        make_settings(rerank_enabled=True, rerank_candidates=6),
        "body",
        3,
        ltr=model,
    )
    assert kb.rerank_skipped == "timeout"
    assert not kb.ltr_applied and kb.ltr_skipped == "needs_rerank_scores"
    assert [h.chunk.doc_id for h in kb.hits] == ["d1", "d2", "d3"]  # RRF order kept
    assert model.seen is None


def _trained_model(tmp_path, uses_ce):
    rng = np.random.default_rng(0)
    groups = [
        ltr.QueryGroup(rng.normal(size=(8, len(ltr.FEATURES))).astype("float32"), np.eye(8)[0])
        for _ in range(10)
    ]
    model, _ = ltr.train(groups, groups, uses_ce=uses_ce, epochs=1, log=lambda *_: None)
    model.save(tmp_path / "ltr")
    return str(tmp_path / "ltr")


def test_search_ranked_endpoint_reports_stages(tmp_path):
    from backend.main import create_app

    path = _trained_model(tmp_path, uses_ce=False)
    with TestClient(create_app(make_settings(ltr_model_path=path))) as client:
        assert client.get("/health").json()["backends"]["ranker"] == "ltr"
        docs = [d.model_dump() for d in load_samples()]
        assert client.post("/ingest", json={"documents": docs}).status_code == 200
        body = client.post("/search/ranked", json={"query": "sk_buff ring", "top_k": 3}).json()
    assert body["ltr_applied"] is True and len(body["hits"]) == 3
    assert set(body["timings"]) == {"retrieval_ms", "rerank_ms", "ltr_ms", "total_ms"}


def test_ce_model_without_reranking_is_rejected_at_startup(tmp_path):
    from backend.main import create_app

    path = _trained_model(tmp_path, uses_ce=True)
    with pytest.raises(ValueError, match="RERANK_ENABLED"):
        with TestClient(create_app(make_settings(ltr_model_path=path))):
            pass
