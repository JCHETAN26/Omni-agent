import json
from types import SimpleNamespace

from backend.agent.llm import strict_json_schema
from benchmarks.answer_quality import run, techqa
from benchmarks.answer_quality.judge import Judgment, judge_messages, label


def test_label_rules():
    assert label(True, True, None, None) == "missed"
    assert label(False, True, None, None) == "abstained_correctly"
    assert label(False, False, None, {"correctness": "correct"}) == "answered_unanswerable"
    assert label(True, False, None, {"correctness": "partial"}) == "partial"
    assert label(True, False, "Timeout", None) == "error"
    assert label(True, False, None, None) == "error"  # answered but not judged


def test_judge_schema_is_strict_and_flat():
    schema = strict_json_schema(Judgment)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(Judgment.model_fields)


def test_judge_messages_mark_missing_reference():
    msgs = judge_messages("q?", "", "an answer", [])
    text = msgs[0]["content"]
    assert "not answerable from the corpus" in text and "(no passages)" in text


def test_parse_technote_strips_title_and_text_markers():
    doc = techqa.parse_technote(
        "swg1", "Title: IBM Fix the thing - United States\n\nText:\n TECHNOTE\nDo X."
    )
    assert doc.title == "IBM Fix the thing"
    assert doc.text.startswith("TECHNOTE")


def test_load_questions(tmp_path):
    rows = [
        {
            "id": "A",
            "question": " q1 ",
            "answer": "fix",
            "is_impossible": False,
            "contexts": [{"filename": "swg2.txt", "text": "..."}],
        },
        {"id": "B", "question": "q2", "answer": "-", "is_impossible": True, "contexts": []},
    ]
    (tmp_path / "train.json").write_text(json.dumps(rows))
    a, b = techqa.load_questions(tmp_path)
    assert (a.question, a.answerable, a.gold_docs, a.reference) == ("q1", True, ("swg2",), "fix")
    assert (b.answerable, b.gold_docs, b.reference) == (False, (), "")


def test_select_is_stratified_and_seeded():
    qs = [SimpleNamespace(qid=str(i), answerable=i % 3 != 0) for i in range(300)]
    a = run.select(qs, 30, 1)
    assert len(a) == 30 and sum(q.answerable for q in a) == 20
    assert [q.qid for q in a] == [q.qid for q in run.select(qs, 30, 1)]
    assert run.select(qs, None, 1) is qs


def test_mcnemar_and_kappa():
    assert run._mcnemar(0, 0) == 1.0
    assert run._mcnemar(10, 0) < 0.01
    assert run._mcnemar(5, 5) == 1.0
    assert run._kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert run._kappa(["a", "b"], ["b", "a"]) < 0


def _row(qid, answerable, lab, faithful=True, gold_hit=True):
    j = (
        None
        if lab in ("missed", "abstained_correctly")
        else {
            "correctness": lab if answerable else "incorrect",
            "faithful": faithful,
            "actionable": True,
        }
    )
    return {
        "qid": qid,
        "answerable": answerable,
        "label": lab,
        "judgment": j,
        "gold_hit": gold_hit,
        "latency_s": 1.0,
        "outcome": "x",
        "tool": "knowledge_base",
    }


def test_summarize_rates():
    rows = [
        _row("1", True, "correct"),
        _row("2", True, "partial"),
        _row("3", True, "missed", gold_hit=False),
        _row("4", False, "abstained_correctly"),
        _row("5", False, "answered_unanswerable", faithful=False),
    ]
    s = run.summarize(rows, 1)
    assert s["answerable_correct"]["rate"] == round(1 / 3, 4)
    assert s["answerable_correct_or_partial"]["rate"] == round(2 / 3, 4)
    assert s["unanswerable_abstained"]["rate"] == 0.5
    assert s["end_to_end_accuracy"]["rate"] == 0.4  # correct + correct abstention, of 5
    assert s["precision_when_answering"]["rate"] == round(1 / 3, 4)  # 1 of 3 answered
    assert s["faithful_when_answering"]["rate"] == round(2 / 3, 4)
    assert s["gold_doc_retrieved"]["rate"] == round(2 / 3, 4)


def test_paired_counts_discordant_questions():
    a = [{"qid": "1", "ok": True}, {"qid": "2", "ok": False}, {"qid": "3", "ok": False}]
    b = [{"qid": "1", "ok": True}, {"qid": "2", "ok": True}, {"qid": "3", "ok": True}]
    p = run._paired(a, b, lambda r: r["ok"], 1)
    assert (p["pairs"], p["agent_only_right"], p["rag_only_right"]) == (3, 2, 0)
    assert p["delta"] == round(2 / 3, 4)


def test_retries_transient_errors_then_succeeds(monkeypatch):
    import asyncio

    async def no_sleep(_):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", no_sleep)
    calls = []

    async def flaky(i):
        calls.append(i)
        if i < 2:
            raise RuntimeError("Client error '429 Too Many Requests'")
        return "ok"

    assert asyncio.run(run._with_retries(flaky)) == "ok"
    assert calls == [0, 1, 2]


def test_exhausted_credits_abort_instead_of_retrying():
    import asyncio

    import pytest

    async def broke(_):
        raise RuntimeError('429 {"code": "credit_balance_exhausted"}')

    with pytest.raises(run.CreditsExhausted):
        asyncio.run(run._with_retries(broke))


def test_agent_turn_with_heuristic_fallback_is_rejected():
    import asyncio

    import pytest

    class Runner:
        async def run_turn(self, thread_id, question, emit):
            await emit({"type": "agent.step", "node": "grade_relevance", "fallback": True})
            return {"answer": "x", "outcome": "answered"}

    q = SimpleNamespace(qid="Q1", question="why?")
    with pytest.raises(run.LLMFallbackUsed):
        asyncio.run(run._answer_agent(Runner(), q))


def test_consistency_regrades_a_sample_and_reports_agreement(tmp_path, monkeypatch):
    import asyncio

    for variant in run.VARIANTS:
        answers = [
            {
                "qid": str(i),
                "question": "q",
                "reference": "r",
                "answer": "a",
                "passages": [],
                "error": None,
                "abstained": False,
            }
            for i in range(4)
        ]
        (tmp_path / f"answers-{variant}.jsonl").write_text(
            "".join(json.dumps(a) + "\n" for a in answers)
        )
        first = [{"qid": str(i), "correctness": "correct", "faithful": True} for i in range(4)]
        (tmp_path / f"judgments-{variant}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in first)
        )

    class Model:
        async def complete_json(self, system, messages, schema):
            return Judgment(
                correctness="partial",
                actionable=True,
                faithful=True,
                unsupported_claims=[],
                reason="",
            )

        async def close(self):
            return None

    monkeypatch.setattr(run, "_judge_model", lambda name: Model())
    args = SimpleNamespace(run=tmp_path, model="m", n=3, seed=1, concurrency=2)
    asyncio.run(run.consistency(args))
    out = json.loads((tmp_path / "judge-consistency.json").read_text())
    assert out["overall"]["regraded"] == 6
    assert out["overall"]["correctness_same"] == 0.0
    assert out["per_variant"]["rag"]["faithful_same"] == 1.0


def test_retrieval_replay_uses_production_pipeline_with_stored_scores():
    import asyncio

    from backend.core.models import Chunk, SearchHit
    from benchmarks.answer_quality import retrieval
    from search.hybrid_retriever import HybridResult, RetrievalTimings

    def hit(doc, i, rank):
        chunk = Chunk(
            chunk_id=f"{doc}-{i}",
            doc_id=doc,
            document_title=doc,
            section_heading="",
            position=i,
            text=f"text {doc} {i}",
        )
        return SearchHit(chunk=chunk, score=1 / rank, sparse_rank=rank, dense_rank=rank)

    hits = [hit("a", 0, 1), hit("a", 1, 2), hit("a", 2, 3), hit("a", 3, 4), hit("gold", 0, 5)]
    timings = RetrievalTimings(dense_ms=0, sparse_ms=0, fusion_ms=0, total_ms=0)
    pools = [
        {
            "qid": "DEV_Q1",
            "question": "why does it fail",
            "gold_docs": ["gold"],
            "result": HybridResult(hits=hits, timings=timings).model_dump(),
        }
    ]
    ce = {"DEV_Q1": {"gold-0": 9.0}}
    base = asyncio.run(retrieval._replay(pools, ce, retrieval._config("b", None, None)))
    assert base["DEV_Q1"] == {4: False, 10: True, 20: True}
    capped = retrieval._config("c", None, None, kb_max_chunks_per_doc=1, rerank_candidates=20)
    assert asyncio.run(retrieval._replay(pools, ce, capped))["DEV_Q1"][4] is True
    reranked = retrieval._config("r", "m", None, rerank_enabled=True, rerank_candidates=20)
    assert asyncio.run(retrieval._replay(pools, ce, reranked))["DEV_Q1"][4] is True


def test_fold_ltr_ranks_each_question_with_the_model_that_never_saw_it(tmp_path):
    import asyncio
    import json

    import numpy as np

    from benchmarks.answer_quality import run

    class Fixed:
        uses_ce = True

        def __init__(self, value):
            self.value = value

        def score(self, x):
            return np.full(len(x), self.value)

    (tmp_path / "folds.json").write_text(json.dumps({"folds": 2, "fold_of": {"TRAIN_Q1": 1}}))
    ltr = run.FoldLtr.__new__(run.FoldLtr)
    ltr.folds, ltr.fold_of = 2, {"TRAIN_Q1": 1}
    ltr.models = [Fixed(0.0), Fixed(1.0)]

    async def scored(qid):
        run._QID.set(qid)
        await asyncio.sleep(0)
        return float(ltr.score(np.zeros((1, 3)))[0])

    async def both():
        return await asyncio.gather(scored("TRAIN_Q1"), scored("DEV_Q9"))

    first, unanswerable = asyncio.run(both())
    assert first == 1.0  # its own held-out fold, not the default
    assert unanswerable == float(ltr.fold("DEV_Q9"))  # stable hash fold
    assert run._settings("m", "ltr-ce").rerank_candidates == 50
    assert run._settings("m").rerank_enabled is False
