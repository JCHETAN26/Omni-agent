import json
import math

import pytest

from benchmarks.retrieval import beir
from benchmarks.retrieval.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_matches_hand_computed_value():
    # Relevant docs at ranks 2 and 4, two judged relevant docs in total.
    rel = {"a": 1, "b": 1}
    ranking = ["x", "a", "y", "b"]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(rel, ranking, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(rel, ranking, 1) == 0.0
    assert ndcg_at_k(rel, ["a", "b"], 10) == pytest.approx(1.0)


def test_graded_relevance_uses_linear_gain():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(rel, ["b", "a"], 10) == pytest.approx(
        (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    )


def test_mrr_and_recall():
    rel = {"a": 1, "b": 1, "c": 1}
    assert mrr_at_k(rel, ["x", "y", "b"], 10) == pytest.approx(1 / 3)
    assert mrr_at_k(rel, ["x", "y", "b"], 2) == 0.0
    assert recall_at_k(rel, ["a", "x", "c"], 10) == pytest.approx(2 / 3)


def test_evaluate_skips_unjudged_and_drops_the_query_itself():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 0}}  # q2 has no relevant doc
    run = {"q1": ["q1", "d1"]}  # the query's own id ranks first and is ignored
    scores = evaluate(qrels, run, ks=(10,))
    assert scores["ndcg@10"] == pytest.approx(1.0)
    assert evaluate(qrels, run, ks=(10,), ignore_identical_ids=False)["ndcg@10"] < 1.0
    assert evaluate({"q1": {"d1": 1}}, {}, ks=(10,))["ndcg@10"] == 0.0  # missing run scores 0


def test_duplicate_results_only_count_once():
    assert ndcg_at_k({"a": 1}, ["a"], 10) == evaluate({"q": {"a": 1}}, {"q": ["a", "a"]})["ndcg@10"]


def _write_forum(root, forum, docs, queries, qrels):
    base = root / forum
    (base / "qrels").mkdir(parents=True)
    (base / "corpus.jsonl").write_text(
        "".join(json.dumps({"_id": i, "title": t, "text": x}) + "\n" for i, t, x in docs)
    )
    (base / "queries.jsonl").write_text(
        "".join(json.dumps({"_id": i, "text": t}) + "\n" for i, t in queries)
    )
    (base / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t{s}\n" for q, d, s in qrels)
    )


def test_beir_loader_prefixes_ids_per_forum(tmp_path):
    _write_forum(
        tmp_path, "unix", [("1", "grep", "find text")], [("9", "search files")], [("9", "1", 1)]
    )
    _write_forum(tmp_path, "tex", [("1", "tables", "tabular")], [("9", "align")], [("9", "1", 1)])
    data = beir.load(tmp_path, ["unix", "tex"])
    assert data.qrels == {"unix/9": {"unix/1": 1}, "tex/9": {"tex/1": 1}}
    assert data.queries["tex/9"] == "align"
    docs = list(data.documents())
    assert [d.doc_id for d in docs] == ["unix/1", "tex/1"]
    assert docs[0].metadata == {"forum": "unix"} and data.count_documents() == 2
    with pytest.raises(FileNotFoundError):
        beir.load(tmp_path, ["gaming"])
