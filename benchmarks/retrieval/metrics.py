"""IR metrics matching trec_eval / pytrec_eval, which BEIR reports.

* nDCG@k uses linear gain (rel / log2(rank + 1)) and an ideal DCG built from
  the query's judged relevant documents, as trec_eval's ``ndcg_cut`` does.
* Queries without any relevant judgement are skipped (they have no ideal).
* ``ignore_identical_ids`` drops a result whose id equals the query id, which
  BEIR applies to CQADupStack because questions can retrieve themselves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]  # query_id -> {doc_id: relevance}
Run = Mapping[str, Sequence[str]]  # query_id -> ranked doc ids


def _clean(query_id: str, ranking: Sequence[str], ignore_identical_ids: bool) -> list[str]:
    seen: set[str] = set()
    out = []
    for doc_id in ranking:
        if doc_id in seen or (ignore_identical_ids and doc_id == query_id):
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def ndcg_at_k(relevance: Mapping[str, int], ranking: Sequence[str], k: int) -> float:
    dcg = sum(
        relevance.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranking[:k], start=1)
    )
    ideal = sorted((r for r in relevance.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(rank + 1) for rank, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(relevance: Mapping[str, int], ranking: Sequence[str], k: int) -> float:
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevance: Mapping[str, int], ranking: Sequence[str], k: int) -> float:
    relevant = {d for d, r in relevance.items() if r > 0}
    return len(relevant.intersection(ranking[:k])) / len(relevant) if relevant else 0.0


def evaluate(
    qrels: Qrels,
    run: Run,
    ks: Sequence[int] = (10, 100),
    ignore_identical_ids: bool = True,
) -> dict[str, float]:
    """Mean metrics over judged queries; a judged query missing from ``run`` scores 0."""
    judged = [q for q, rel in qrels.items() if any(r > 0 for r in rel.values())]
    if not judged:
        raise ValueError("no judged queries")
    totals: dict[str, float] = {}
    for query_id in judged:
        ranking = _clean(query_id, run.get(query_id, ()), ignore_identical_ids)
        rel = qrels[query_id]
        for k in ks:
            for name, fn in (("ndcg", ndcg_at_k), ("mrr", mrr_at_k), ("recall", recall_at_k)):
                key = f"{name}@{k}"
                totals[key] = totals.get(key, 0.0) + fn(rel, ranking, k)
    return {key: value / len(judged) for key, value in sorted(totals.items())}


def per_query_ndcg(qrels: Qrels, run: Run, k: int = 10, ignore_identical_ids: bool = True):
    """Per-query nDCG@k, for paired significance tests between two systems."""
    return {
        q: ndcg_at_k(rel, _clean(q, run.get(q, ()), ignore_identical_ids), k)
        for q, rel in qrels.items()
        if any(r > 0 for r in rel.values())
    }
