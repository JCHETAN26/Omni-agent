"""End-to-end smoke test of the BEIR benchmark runner on a tiny two-forum fixture."""

import json

import numpy as np
import pytest

from backend.core.stores import InMemoryBM25Index
from benchmarks.retrieval import run
from search.ltr import FEATURES

TOPICS = {
    "unix": ["grep regex", "ssh tunnel", "cron schedule", "tar archive", "rsync mirror"],
    "webmasters": ["dns record", "nginx redirect", "tls certificate", "robots crawl", "cdn cache"],
}


def _fixture(root):
    for forum, topics in TOPICS.items():
        base = root / "data" / "cqadupstack" / forum
        (base / "qrels").mkdir(parents=True)
        docs, queries, qrels = [], [], []
        for t, topic in enumerate(topics):
            for k in range(6):
                docs.append(
                    {
                        "_id": f"{t}-{k}",
                        "title": f"{topic} question {k}",
                        "text": f"how to {topic} {k}",
                    }
                )
            for q in range(4):
                qid = f"q{t}-{q}"
                queries.append({"_id": qid, "text": f"{topic} help {q}"})
                qrels.append(f"{qid}\t{t}-{q}\t1")
        (base / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
        (base / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
        (base / "qrels" / "test.tsv").write_text(
            "query-id\tcorpus-id\tscore\n" + "\n".join(qrels) + "\n"
        )


@pytest.fixture
def shared_sparse(monkeypatch):
    """Elasticsearch persists between phases; the in-process BM25 must be shared to match."""
    index = InMemoryBM25Index()

    async def close():
        return None

    index.close = close
    monkeypatch.setattr("backend.core.stores.sparse_index_from_settings", lambda _settings: index)
    monkeypatch.delenv("ELASTICSEARCH_URL", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    return index


def test_index_collect_evaluate_end_to_end(tmp_path, shared_sparse):
    _fixture(tmp_path)
    common = [
        "--work",
        str(tmp_path),
        "--forums",
        "unix",
        "webmasters",
        "--embedding-provider",
        "hashing",
    ]
    run.main(["index", *common, "--dense", "faiss", "--batch-size", "16"])
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["documents"] == 60 and index["chunks"] >= 60
    assert (tmp_path / "faiss" / "index.faiss").exists()

    run.main(
        [
            "collect",
            *common,
            "--dense",
            "faiss",
            "--inference",
            "mock",
            "--candidates",
            "20",
            "--pool",
            "8",
        ]
    )
    z = np.load(tmp_path / "collect-faiss" / "candidates.npz")
    assert len(z["query_ids"]) == 40 and z["x"].shape[1] == len(FEATURES)
    assert (z["pools"] <= 8).all() and z["lengths"].sum() == len(z["doc_ids"])

    run.main(["evaluate", *common, "--dense", "faiss", "--epochs", "2"])
    summary = json.loads((tmp_path / "evaluate-faiss.json").read_text())
    systems = set(summary["results"])
    assert systems == {"bm25", "dense", "rrf", "rrf+cross-encoder", "ltr", "ltr+cross-encoder"}
    for metrics in summary["results"].values():
        assert 0.0 <= metrics["ndcg@10"] <= 1.0
        assert set(metrics["same_forum_by_forum"]["per_forum"]) <= set(TOPICS)
    assert sum(summary["queries"].values()) == 40
    assert summary["results"]["bm25"]["ndcg@10"] > 0.3  # lexical topics make BM25 work
    assert (tmp_path / "models" / "ltr_cross-encoder" / "model.pt").exists()
    assert "p_value_vs_rrf" in summary["significance"]["ltr"]


def test_rankings_rerank_only_the_pool_and_dedupe_documents():
    x = np.zeros((5, len(FEATURES)), dtype="float32")
    x[:3, FEATURES.index("ce_score")] = [0.1, 0.9, 0.5]
    group = {
        "x": x,
        "doc_ids": ["a", "b", "a", "c", "d"],  # chunks 0 and 2 belong to doc a
        "dense_rank": np.array([2, 0, 1, 0, 3]),
        "sparse_rank": np.array([0, 1, 0, 2, 0]),
        "pool": 3,
    }
    out = run.rankings(group, {"ltr": np.array([0.0, 1.0, 2.0])})
    assert out["rrf"] == ["a", "b", "c", "d"]
    assert out["rrf+cross-encoder"] == ["b", "a", "c", "d"]
    assert out["ltr"] == ["a", "b", "c", "d"]  # chunk 2 (doc a) first, rest keep RRF order
    assert out["dense"] == ["a", "d"] and out["bm25"] == ["b", "c"]


def test_split_is_deterministic_and_roughly_60_20_20():
    ids = [f"forum/{i}" for i in range(5000)]
    counts = {s: sum(run.split_of(i) == s for i in ids) for s in run.SPLITS}
    assert run.split_of("unix/42") == run.split_of("unix/42")
    assert abs(counts["train"] / 5000 - 0.6) < 0.03 and abs(counts["test"] / 5000 - 0.2) < 0.03


def test_paired_randomization_detects_a_real_shift():
    a = {str(i): 0.3 for i in range(200)}
    assert run.paired_randomization(a, {q: v + 0.1 for q, v in a.items()}) < 0.001
    noisy = {q: v + (0.05 if int(q) % 2 else -0.05) for q, v in a.items()}
    assert run.paired_randomization(a, noisy) > 0.5


class _StubCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: scores term overlap."""

    def __init__(self, name, max_length, device):
        assert (name, max_length) == ("cross-encoder/ms-marco-MiniLM-L-6-v2", 256)

    def predict(self, pairs, batch_size, activation_fn, show_progress_bar):
        return np.array([len(set(q.split()) & set(t.split())) for q, t in pairs], dtype="float32")


def test_split_pipeline_embed_elsewhere_then_score_ce_offline(tmp_path, shared_sparse, monkeypatch):
    _fixture(tmp_path)
    common = ["--work", str(tmp_path), "--forums", "unix", "webmasters"]
    common += ["--embedding-provider", "hashing", "--dense", "faiss"]
    run.main(["embed", *common, "--batch-size", "16"])
    assert (tmp_path / "embed.json").exists() and not (tmp_path / "faiss").exists()
    run.main(["index", *common, "--batch-size", "16"])  # loads the cached vectors
    assert json.loads((tmp_path / "index.json").read_text())["embed_s"] == 0.0

    run.main(["collect", *common, "--no-ce", "--candidates", "20", "--pool", "8"])
    with pytest.raises(RuntimeError, match="score-ce"):
        run.main(["evaluate", *common, "--epochs", "1"])

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _StubCrossEncoder)
    run.main(["score-ce", *common])
    z = np.load(tmp_path / "collect-faiss" / "candidates.npz")
    ce = z["x"][:, FEATURES.index("ce_score")]
    gap = z["x"][:, FEATURES.index("ce_gap")]
    assert bool(z["ce_scored"]) and ce.max() > 0 and gap.max() == 0.0
    run.main(["evaluate", *common, "--epochs", "1"])
    assert (
        "ltr+cross-encoder" in json.loads((tmp_path / "evaluate-faiss.json").read_text())["results"]
    )


def test_cached_vectors_from_different_chunks_are_rejected(tmp_path, shared_sparse):
    _fixture(tmp_path)
    common = ["--work", str(tmp_path), "--forums", "unix", "--embedding-provider", "hashing"]
    run.main(["embed", *common, "--batch-size", "16"])
    (tmp_path / "vectors" / "part-00000.ids").write_text("0" * 64 + "\n")
    with pytest.raises(RuntimeError, match="different chunks"):
        run.main(["index", *common, "--dense", "faiss", "--batch-size", "16"])
