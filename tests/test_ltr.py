import json

import numpy as np
import pytest

from backend.core.models import Chunk, SearchHit
from search import ltr


def _hit(i, title, text, *, rrf, sparse=None, srank=None, dense=None, drank=None, position=0):
    chunk = Chunk(
        chunk_id=f"c{i}",
        doc_id=f"d{i}",
        document_title=title,
        section_heading=title,
        position=position,
        text=text,
    )
    return SearchHit(
        chunk=chunk,
        score=rrf,
        sparse_score=sparse,
        sparse_rank=srank,
        dense_score=dense,
        dense_rank=drank,
    )


def test_features_are_query_relative_and_handle_missing_branches():
    hits = [
        _hit(1, "grep files", "use grep -r", rrf=0.03, sparse=10.0, srank=1, dense=0.8, drank=2),
        _hit(2, "awk", "awk prints", rrf=0.02, dense=0.9, drank=1, position=12),
    ]
    x = ltr.features("grep files", hits, [2.0, None])
    col = {name: i for i, name in enumerate(ltr.FEATURES)}
    assert x.shape == (2, len(ltr.FEATURES))
    assert x[0, col["sparse_score_norm"]] == 1.0 and x[0, col["sparse_rr"]] == 1.0
    assert x[1, col["has_sparse"]] == 0.0 and x[1, col["sparse_score_norm"]] == 0.0
    assert x[1, col["dense_gap"]] == 0.0 and x[0, col["dense_gap"]] == pytest.approx(-0.1)
    assert x[0, col["title_overlap"]] == 1.0 and x[1, col["title_overlap"]] == 0.0
    assert x[0, col["ce_gap"]] == 0.0 and x[1, col["ce_score"]] == 0.0
    assert x[1, col["position"]] == 1.0  # capped at 10 chunks
    assert ltr.features("q", []).shape == (0, len(ltr.FEATURES))


def _synthetic_groups(n_queries, rng):
    """Relevance is driven by text overlap and the CE score; BM25 rank is noise."""
    groups = []
    for _ in range(n_queries):
        n = 20
        x = rng.normal(size=(n, len(ltr.FEATURES))).astype("float32")
        signal = x[:, 12] + x[:, 9]
        y = np.zeros(n, dtype="float32")
        y[np.argsort(-signal)[:2]] = 1.0
        groups.append(ltr.QueryGroup(x, y))
    return groups


def test_training_learns_a_planted_signal_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    train_groups, dev_groups = _synthetic_groups(300, rng), _synthetic_groups(60, rng)
    model, report = ltr.train(
        train_groups, dev_groups, uses_ce=True, epochs=15, log=lambda *_: None
    )
    random_ndcg = np.mean([ltr._ndcg10(rng.normal(size=len(g.y)), g.y) for g in dev_groups])
    assert report["best_dev_ndcg10"] > 0.8 > random_ndcg + 0.3

    model.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "model.json").read_text())
    assert meta["uses_ce"] is True and meta["features"] == list(ltr.FEATURES)
    loaded = ltr.LtrModel.load(tmp_path / "m")
    sample = dev_groups[0].x
    assert np.allclose(loaded.score(sample), model.score(sample), atol=1e-6)


def test_model_without_ce_ignores_ce_columns(tmp_path):
    rng = np.random.default_rng(1)
    groups = _synthetic_groups(40, rng)
    model, _ = ltr.train(groups, groups[:10], uses_ce=False, epochs=2, log=lambda *_: None)
    x = groups[0].x.copy()
    before = model.score(x)
    x[:, 9:11] = 1000.0  # CE columns must have no effect
    assert np.allclose(model.score(x), before)
    model.save(tmp_path / "m")
    assert "ce_score" not in json.loads((tmp_path / "m" / "model.json").read_text())["features"]


def test_loading_a_model_with_a_different_feature_set_fails(tmp_path):
    rng = np.random.default_rng(2)
    groups = _synthetic_groups(20, rng)
    model, _ = ltr.train(groups, groups, uses_ce=True, epochs=1, log=lambda *_: None)
    model.save(tmp_path / "m")
    meta_path = tmp_path / "m" / "model.json"
    meta = json.loads(meta_path.read_text())
    meta["features"] = meta["features"][::-1]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different feature set"):
        ltr.LtrModel.load(tmp_path / "m")


def test_training_requires_a_relevant_candidate():
    empty = [ltr.QueryGroup(np.zeros((3, len(ltr.FEATURES)), "float32"), np.zeros(3, "float32"))]
    with pytest.raises(ValueError):
        ltr.train(empty, empty, uses_ce=True, log=lambda *_: None)
