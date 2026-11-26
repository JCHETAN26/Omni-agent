"""Answer-quality evaluation on TechQA: plain RAG vs the self-correcting agent.

Both variants use the same corpus, hybrid retrieval (BM25 + dense, RRF), top-k,
LLM and answer prompt. They differ only in what happens around generation:

  rag    retrieve top-k for the question, generate once, return the answer
  agent  the full LangGraph agent: route, grade the passages, rewrite and
         re-retrieve, generate, check grounding, retry or abstain

Steps (each resumable; outputs go to --run):

  fetch       download + checksum the dataset
  answer      index the corpus in-process, then answer every question per variant
  judge       grade answered responses with a different, stronger model
  report      metrics with bootstrap CIs and a paired agent-vs-RAG comparison
  consistency re-grade a seeded sample to measure the judge's own run-to-run agreement
  human-sample / agreement
              export a blind sample for hand grading; measure judge-human agreement

  python -m benchmarks.answer_quality.run fetch --data ~/omni-eval/techqa
  python -m benchmarks.answer_quality.run answer --data ~/omni-eval/techqa --run <dir>
  python -m benchmarks.answer_quality.run judge --run <dir>
  python -m benchmarks.answer_quality.run report --run <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from math import comb
from pathlib import Path

from benchmarks.answer_quality import techqa
from benchmarks.answer_quality.judge import (
    JUDGE_PROMPT,
    Judgment,
    judge_messages,
    label,
)

ANSWER_MODEL = "gpt-4.1-mini-2025-04-14"
JUDGE_MODEL = "gpt-4.1-2025-04-14"
VARIANTS = ("rag", "agent")
# USD per 1M tokens (input, output), OpenAI list prices used for the cost estimate.
PRICES = {
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-2025-04-14": (2.00, 8.00),
}


# -- shared helpers -----------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tokens() -> tuple[float, float]:
    from backend.core import metrics

    return (
        metrics.LLM_TOKENS.labels("prompt")._value.get(),
        metrics.LLM_TOKENS.labels("completion")._value.get(),
    )


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 3)


def _cost(model: str, prompt: float, completion: float) -> float | None:
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    return round(prompt / 1e6 * pin + completion / 1e6 * pout, 4)


def select(questions: list, limit: int | None, seed: int) -> list:
    """All questions, or a seeded sample stratified by answerability."""
    if not limit or limit >= len(questions):
        return questions
    rng = random.Random(seed)
    yes = [q for q in questions if q.answerable]
    no = [q for q in questions if not q.answerable]
    n_yes = round(limit * len(yes) / len(questions))
    return rng.sample(yes, n_yes) + rng.sample(no, limit - n_yes)


# -- answer -------------------------------------------------------------------


# Ranking stacks the answer run can use. "ltr-ce" is the TechQA-selected stack
# (benchmarks/answer_quality/retrieval.py): bge-reranker-base scores over the top
# 50 fused candidates, a learning-to-rank model on top, one chunk per technote.
RANKINGS = {
    "rrf": {},
    "ltr-ce": {
        "inference_backend": "local",
        "local_rerank_model": "BAAI/bge-reranker-base",
        "local_rerank_max_length": 512,
        "rerank_enabled": True,
        "rerank_candidates": 50,
        "kb_max_chunks_per_doc": 1,
        # Offline run: the reranker is serialised in-process and queues under load;
        # a generous budget keeps every turn on the measured stack (fallbacks are counted).
        "rerank_timeout_s": 300.0,
    },
}

_QID: contextvars.ContextVar[str] = contextvars.ContextVar("techqa_qid")


class FoldLtr:
    """Ranks each question with the cross-validation fold model that never saw it.

    ``retrieval.py crossval`` trains one LTR model per fold of the answerable
    questions. Unanswerable questions were never training data, so any fold is
    out-of-sample for them; they get a stable hash-assigned fold.
    """

    def __init__(self, directory: Path):
        from search.ltr import LtrModel

        meta = json.loads((directory / "folds.json").read_text())
        self.folds = meta["folds"]
        self.fold_of = meta["fold_of"]
        self.models = [LtrModel.load(directory / f"fold-{f}") for f in range(self.folds)]
        self.uses_ce = all(m.uses_ce for m in self.models)
        self.directory = directory

    def fold(self, qid: str) -> int:
        if qid in self.fold_of:
            return self.fold_of[qid]
        return int(hashlib.sha256(qid.encode()).hexdigest(), 16) % self.folds

    def score(self, x):
        return self.models[self.fold(_QID.get())].score(x)


def _settings(model: str, ranking: str = "rrf"):
    from backend.core.config import Settings

    return Settings(
        llm_provider="openai",
        llm_model=model,
        llm_temperature=0.0,
        llm_cache_enabled=False,
        embedding_provider="sentence-transformers",
        dense_backend="faiss",
        faiss_path=None,
        qdrant_url=None,
        elasticsearch_url=None,
        redis_url=None,
        inference_backend="mock",
        rerank_enabled=False,
        ltr_model_path=None,
        vision_backend="off",
        otel_exporter_otlp_endpoint=None,
    ).model_copy(update=RANKINGS[ranking])


def _passages_out(passages: list[dict], sources: list[dict]) -> list[dict]:
    doc_by_chunk = {s["chunk_id"]: s.get("doc_id") for s in sources}
    return [
        {
            "doc_id": doc_by_chunk.get(p["chunk_id"]),
            "document_title": p["document_title"],
            "section_heading": p["section_heading"],
            "text": p["text"],
        }
        for p in passages
    ]


async def _answer_rag(services, reasoner, q) -> dict:
    from backend.agent.reasoning import ToolContext
    from backend.agent.workflow import _passage
    from search.pipeline import retrieve_with_rerank

    s = services
    kb = await retrieve_with_rerank(
        s.retriever, s.inference, s.settings, q.question, s.settings.agent_top_k, ltr=s.ltr
    )
    passages = [_passage(h, r) for h, r in zip(kb.hits, kb.rerank_scores, strict=True)]

    async def ignore(_: str) -> None:
        return None

    gen = await reasoner.generate(q.question, ToolContext("knowledge_base", passages), [], ignore)
    return {
        "answer": gen.text,
        "abstained": gen.abstained,
        "outcome": "abstained" if gen.abstained else "answered",
        "tool": "knowledge_base",
        "retries": 0,
        "attempts": 1,
        "passages": _passages_out(passages, kb.sources()),
    }


class LLMFallbackUsed(RuntimeError):
    """The agent answered with its offline heuristic for a step: not an LLM answer."""


class CreditsExhausted(RuntimeError):
    """The API account cannot serve requests; retrying is pointless."""


_FATAL = ("credit_balance_exhausted", "insufficient_quota", "invalid_api_key")
_TRANSIENT = ("429", "500", "502", "503", "504", "timed out", "Timeout", "LLMFallbackUsed")


async def _with_retries(fn, attempts: int = 6):
    """Retry transient API failures with backoff; abort the run when credits are gone."""
    for i in range(attempts):
        try:
            return await fn(i)
        except Exception as exc:  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
            if any(f in text for f in _FATAL):
                raise CreditsExhausted(text[:300]) from exc
            if i == attempts - 1 or not any(t in text for t in _TRANSIENT):
                raise
            await asyncio.sleep(min(60.0, 2.0 * 2**i) + random.random())


async def preflight(settings) -> None:
    """One tiny real call, so a dead key or empty balance fails before indexing."""
    from backend.agent.llm import chat_model_from_settings

    model = chat_model_from_settings(settings)
    try:
        await _with_retries(
            lambda _: _drain(
                model.stream_text("Reply with ok.", [{"role": "user", "content": "ok"}])
            )
        )
    finally:
        await model.close()


async def _drain(stream) -> str:
    return "".join([d async for d in stream])


async def _answer_agent(runner, q, attempt: int = 0) -> dict:
    fallbacks: list[str] = []

    async def watch(event: dict) -> None:
        # The workflow silently substitutes the heuristic reasoner when an LLM step
        # fails. Such a turn is not an LLM answer, so it must not be scored as one.
        if event.get("type") == "agent.step" and event.get("fallback"):
            fallbacks.append(event.get("node", "?"))
        if event.get("type") == "agent.generation.discarded" and "error" in event.get("reason", ""):
            fallbacks.append("generate")

    state = await runner.run_turn(f"techqa-{q.qid}-{attempt}", q.question, watch)
    if fallbacks:
        raise LLMFallbackUsed(f"heuristic fallback in steps {sorted(set(fallbacks))}")
    return {
        "answer": state.get("answer", ""),
        "abstained": bool(state.get("abstained")),
        "outcome": state.get("outcome", ""),
        "tool": state.get("tool"),
        "retries": state.get("retries", 0),
        "attempts": state.get("attempts", 0),
        "rewrites": state.get("rewrites", []),
        "passages": _passages_out(state.get("passages", []), state.get("sources", [])),
    }


async def answer(args) -> None:
    from backend.agent.runner import AgentRunner
    from backend.core.services import Services

    run: Path = args.run
    run.mkdir(parents=True, exist_ok=True)
    questions = select(techqa.load_questions(args.data), args.limit, args.seed)
    settings = _settings(args.model, args.ranking)
    if (args.ranking == "ltr-ce") != (args.ltr_folds is not None):
        raise SystemExit("--ranking ltr-ce needs --ltr-folds (and only it uses them)")
    await preflight(settings)
    services = await Services.build(settings)
    if args.ltr_folds is not None:
        services.ltr = FoldLtr(args.ltr_folds)
        services.backends["ranker"] = f"ltr+cross-encoder (out-of-fold, {services.ltr.folds} folds)"
    try:
        docs = techqa.load_corpus(args.data)
        t0 = time.perf_counter()
        chunks = 0
        for i in range(0, len(docs), args.ingest_batch):
            report = await services.processor.ingest(docs[i : i + args.ingest_batch])
            chunks += report.chunks
        index = {
            "documents": len(docs),
            "chunks": chunks,
            "seconds": round(time.perf_counter() - t0, 1),
            "backends": services.backends,
        }
        print(f"indexed {index}", flush=True)
        runner = AgentRunner.from_services(services)
        for variant in args.variants:
            out = run / f"answers-{variant}.jsonl"
            done = {r["qid"] for r in _read_jsonl(out)}
            todo = [q for q in questions if q.qid not in done]
            sem = asyncio.Semaphore(args.concurrency)
            p0, c0 = _tokens()
            t_variant = time.perf_counter()

            async def one(q, variant=variant, out=out, sem=sem):
                _QID.set(q.qid)  # gather runs each question in its own task and context
                async with sem:
                    t = time.perf_counter()
                    try:
                        if variant == "rag":
                            result = await _with_retries(
                                lambda _: _answer_rag(services, runner.reasoner, q)
                            )
                        else:
                            result = await _with_retries(lambda i: _answer_agent(runner, q, i))
                        error = None
                    except CreditsExhausted:
                        raise
                    except Exception as exc:  # noqa: BLE001 - recorded, scored as error
                        result, error = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
                    retrieved = [p["doc_id"] for p in result.get("passages", [])]
                    _append_jsonl(
                        out,
                        {
                            "qid": q.qid,
                            "variant": variant,
                            "answerable": q.answerable,
                            "question": q.question,
                            "reference": q.reference,
                            "gold_docs": list(q.gold_docs),
                            "gold_hit": bool(set(q.gold_docs) & set(retrieved)),
                            "latency_s": round(time.perf_counter() - t, 3),
                            "error": error,
                            **result,
                        },
                    )

            await asyncio.gather(*(one(q) for q in todo))
            p1, c1 = _tokens()
            meta_path = run / f"meta-{variant}.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"segments": []}
            meta.update(
                model=args.model,
                index=index,
                seed=args.seed,
                limit=args.limit,
                ranking=args.ranking,
                ranking_settings=RANKINGS[args.ranking],
                ltr_folds=str(args.ltr_folds) if args.ltr_folds else None,
                ranking_fallbacks=_ranking_fallbacks(),
            )
            meta["segments"].append(
                {
                    "questions": len(todo),
                    "wall_s": round(time.perf_counter() - t_variant, 1),
                    "prompt_tokens": int(p1 - p0),
                    "completion_tokens": int(c1 - c0),
                    "concurrency": args.concurrency,
                }
            )
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"{variant}: {len(todo)} answered, tokens {int(p1 - p0)}/{int(c1 - c0)}")
    finally:
        await services.close()


def _ranking_fallbacks() -> dict:
    """Cumulative rerank/LTR fallbacks in this process (should all be zero)."""
    from prometheus_client import REGISTRY

    out = {}
    for metric in ("omniagent_rerank_skipped_total", "omniagent_ltr_skipped_total"):
        for family in REGISTRY.collect():
            for sample in family.samples:
                if sample.name == metric:
                    out[f"{metric}{{{sample.labels.get('reason')}}}"] = sample.value
    return out


# -- judge --------------------------------------------------------------------


def _judge_model(name: str):
    from backend.agent.llm_openai import OpenAICompatibleChatModel
    from backend.core.config import Settings

    s = Settings()
    return OpenAICompatibleChatModel(
        s.llm_base_url, s.llm_api_key or s.openai_api_key, name, s.llm_timeout_s, 0.0
    )


async def _judge_all(model, name: str, todo: list[dict], out: Path, concurrency: int) -> dict:
    """Grade each answer in ``todo``, appending one row per answer to ``out``."""
    sem = asyncio.Semaphore(concurrency)
    p0, c0 = _tokens()

    async def one(a):
        async with sem:
            messages = judge_messages(a["question"], a["reference"], a["answer"], a["passages"])
            try:
                j = await _with_retries(
                    lambda _: model.complete_json(JUDGE_PROMPT, messages, Judgment)
                )
                row = {"qid": a["qid"], "judge_model": name, **j.model_dump()}
            except CreditsExhausted:
                raise
            except Exception as exc:  # noqa: BLE001
                row = {"qid": a["qid"], "judge_error": f"{type(exc).__name__}: {exc}"[:300]}
            _append_jsonl(out, row)

    await asyncio.gather(*(one(a) for a in todo))
    p1, c1 = _tokens()
    return {"judged": len(todo), "prompt_tokens": int(p1 - p0), "completion_tokens": int(c1 - c0)}


def _judgeable(answers: list[dict]) -> list[dict]:
    return [a for a in answers if not a["error"] and not a["abstained"]]


async def judge(args) -> None:
    model = _judge_model(args.model)
    for variant in VARIANTS:
        answers = _read_jsonl(args.run / f"answers-{variant}.jsonl")
        out = args.run / f"judgments-{variant}.jsonl"
        done = {r["qid"] for r in _read_jsonl(out)}
        todo = [a for a in _judgeable(answers) if a["qid"] not in done]
        segment = await _judge_all(model, args.model, todo, out, args.concurrency)
        meta_path = args.run / f"judge-meta-{variant}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"segments": []}
        meta["model"] = args.model
        meta["segments"].append(segment)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"{variant}: judged {len(todo)}")
    await model.close()


async def consistency(args) -> None:
    """Re-grade a seeded sample of judged answers; report how often the verdict moves."""
    model = _judge_model(args.model)
    out = {"model": args.model, "per_variant": {}}
    first_all, second_all = [], []
    for variant in VARIANTS:
        first = {
            r["qid"]: r
            for r in _read_jsonl(args.run / f"judgments-{variant}.jsonl")
            if "correctness" in r
        }
        answers = [a for a in _judgeable(_read_jsonl(args.run / f"answers-{variant}.jsonl"))]
        pool = sorted((a for a in answers if a["qid"] in first), key=lambda a: a["qid"])
        sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
        path = args.run / f"judgments-repeat-{variant}.jsonl"
        done = {r["qid"] for r in _read_jsonl(path)}
        segment = await _judge_all(
            model, args.model, [a for a in sample if a["qid"] not in done], path, args.concurrency
        )
        second = {r["qid"]: r for r in _read_jsonl(path) if "correctness" in r}
        qids = [a["qid"] for a in sample if a["qid"] in second]
        a1 = [first[q]["correctness"] for q in qids]
        a2 = [second[q]["correctness"] for q in qids]
        first_all += a1
        second_all += a2
        out["per_variant"][variant] = {
            "regraded": len(qids),
            "correctness_same": _agree(a1, a2),
            "correctness_kappa": _kappa(a1, a2),
            "correct_vs_not_same": _agree(
                [x == "correct" for x in a1], [x == "correct" for x in a2]
            ),
            "faithful_same": _agree(
                [first[q]["faithful"] for q in qids], [second[q]["faithful"] for q in qids]
            ),
            **{f"tokens_{k}": v for k, v in segment.items() if k != "judged"},
        }
    out["overall"] = {
        "regraded": len(first_all),
        "correctness_same": _agree(first_all, second_all),
        "correctness_kappa": _kappa(first_all, second_all),
    }
    (args.run / "judge-consistency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    await model.close()


# -- report -------------------------------------------------------------------


def _scored(run: Path, variant: str) -> list[dict]:
    judgments = {
        r["qid"]: r
        for r in _read_jsonl(run / f"judgments-{variant}.jsonl")
        if "judge_error" not in r
    }
    rows = []
    for a in _read_jsonl(run / f"answers-{variant}.jsonl"):
        j = judgments.get(a["qid"])
        rows.append(
            {**a, "judgment": j, "label": label(a["answerable"], a["abstained"], a["error"], j)}
        )
    return rows


def _bootstrap(values: list[float], seed: int, n: int = 2000) -> list[float]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def _rate(values: list[bool], seed: int) -> dict:
    v = [1.0 if x else 0.0 for x in values]
    return {
        "n": len(v),
        "rate": round(sum(v) / len(v), 4) if v else None,
        "ci95": _bootstrap(v, seed),
    }


def _mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant pair counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 6)


def summarize(rows: list[dict], seed: int) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["label"] not in ("missed", "abstained_correctly", "error")]
    judged = [r for r in answered if r["judgment"]]
    judged_ans = [r for r in judged if r["answerable"]]
    labels = Counter(r["label"] for r in rows)
    return {
        "questions": len(rows),
        "answerable": len(ans),
        "unanswerable": len(una),
        "labels": dict(labels),
        "answerable_correct": _rate([r["label"] == "correct" for r in ans], seed),
        "answerable_correct_or_partial": _rate(
            [r["label"] in ("correct", "partial") for r in ans], seed
        ),
        "answerable_missed": _rate([r["label"] == "missed" for r in ans], seed),
        "answerable_incorrect": _rate([r["label"] == "incorrect" for r in ans], seed),
        "unanswerable_abstained": _rate([r["label"] == "abstained_correctly" for r in una], seed),
        "end_to_end_accuracy": _rate(
            [r["label"] in ("correct", "abstained_correctly") for r in rows], seed
        ),
        "precision_when_answering": _rate([r["label"] == "correct" for r in answered], seed),
        "faithful_when_answering": _rate([r["judgment"]["faithful"] for r in judged], seed),
        "actionable_when_answering_answerable": _rate(
            [r["judgment"]["actionable"] for r in judged_ans], seed
        ),
        "gold_doc_retrieved": _rate([r["gold_hit"] for r in ans], seed),
        "correct_given_gold_retrieved": _rate(
            [r["label"] == "correct" for r in ans if r["gold_hit"]], seed
        ),
        "errors": labels.get("error", 0),
        "latency_s": {
            "p50": _pct([r["latency_s"] for r in rows], 0.5),
            "p95": _pct([r["latency_s"] for r in rows], 0.95),
        },
        "agent_outcomes": dict(Counter(r.get("outcome") for r in rows)),
        "routes": dict(Counter(r.get("tool") for r in rows)),
    }


def _paired(a: list[dict], b: list[dict], ok, seed: int) -> dict:
    """b minus a on the questions both variants answered without error."""
    by_q = {r["qid"]: r for r in a}
    pairs = [(ok(by_q[r["qid"]]), ok(r)) for r in b if r["qid"] in by_q]
    diffs = [float(y) - float(x) for x, y in pairs]
    b_only = sum(1 for x, y in pairs if y and not x)
    a_only = sum(1 for x, y in pairs if x and not y)
    return {
        "pairs": len(pairs),
        "delta": round(sum(diffs) / len(diffs), 4) if diffs else None,
        "ci95": _bootstrap(diffs, seed),
        "agent_only_right": b_only,
        "rag_only_right": a_only,
        "mcnemar_p": _mcnemar(b_only, a_only),
    }


def report(args) -> None:
    out = {"dataset": techqa.DATASET, "seed": args.seed, "variants": {}}
    scored = {}
    for variant in VARIANTS:
        if not (args.run / f"answers-{variant}.jsonl").exists():
            continue
        rows = _scored(args.run, variant)
        scored[variant] = rows
        summary = summarize(rows, args.seed)
        cost = {}
        for kind in ("meta", "judge-meta"):
            path = args.run / f"{kind}-{variant}.json"
            if path.exists():
                meta = json.loads(path.read_text())
                p = sum(s["prompt_tokens"] for s in meta["segments"])
                c = sum(s["completion_tokens"] for s in meta["segments"])
                cost[kind] = {
                    "model": meta.get("model"),
                    "prompt_tokens": p,
                    "completion_tokens": c,
                    "usd_estimate": _cost(meta.get("model", ""), p, c),
                    "tokens_per_question": round((p + c) / max(1, len(rows)), 1),
                }
                if kind == "meta":
                    summary["index"] = meta.get("index")
        summary["cost"] = cost
        out["variants"][variant] = summary
    if {"rag", "agent"} <= scored.keys():
        rag, agent = scored["rag"], scored["agent"]
        out["agent_vs_rag"] = {
            "answerable_correct": _paired(
                [r for r in rag if r["answerable"]],
                [r for r in agent if r["answerable"]],
                lambda r: r["label"] == "correct",
                args.seed,
            ),
            "unanswerable_abstained": _paired(
                [r for r in rag if not r["answerable"]],
                [r for r in agent if not r["answerable"]],
                lambda r: r["label"] == "abstained_correctly",
                args.seed,
            ),
            "end_to_end_accuracy": _paired(
                rag, agent, lambda r: r["label"] in ("correct", "abstained_correctly"), args.seed
            ),
            "faithful_when_both_answered": _paired(
                [r for r in rag if r["judgment"]],
                [r for r in agent if r["judgment"]],
                lambda r: r["judgment"]["faithful"],
                args.seed,
            ),
        }
    path = args.run / "summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# -- human agreement ----------------------------------------------------------

HUMAN_FIELDS = ("item", "question", "reference", "answer", "passages", "correctness", "faithful")


def human_sample(args) -> None:
    """Blind sample of judged answers (both variants mixed) for hand grading."""
    rows = []
    for variant in VARIANTS:
        rows += [r for r in _scored(args.run, variant) if r["judgment"] and r["answerable"]]
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    key = []
    with (args.run / "human-grading.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HUMAN_FIELDS)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow(
                {
                    "item": i,
                    "question": r["question"],
                    "reference": r["reference"],
                    "answer": r["answer"],
                    "passages": "\n\n".join(
                        f"[{k}] {p['document_title']}\n{p['text']}"
                        for k, p in enumerate(r["passages"], 1)
                    ),
                    "correctness": "",
                    "faithful": "",
                }
            )
            key.append({"item": i, "qid": r["qid"], "variant": r["variant"], **r["judgment"]})
    (args.run / "human-grading-key.json").write_text(json.dumps(key, indent=2))
    print(f"wrote {len(sample)} items to {args.run / 'human-grading.csv'}")


def _kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def _agree(a: list, b: list) -> float | None:
    return round(sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 4) if a else None


def agreement(args) -> None:
    key = {k["item"]: k for k in json.loads((args.run / "human-grading-key.json").read_text())}
    human = list(csv.DictReader(args.grades.open()))
    pairs = [
        (key[int(h["item"])], h)
        for h in human
        if h["correctness"].strip() and int(h["item"]) in key
    ]
    jc = [k["correctness"] for k, _ in pairs]
    hc = [h["correctness"].strip().lower() for _, h in pairs]
    jb = [k["correctness"] == "correct" for k, _ in pairs]
    hb = [h == "correct" for h in hc]
    jf = [str(k["faithful"]).lower() for k, h in pairs if h["faithful"].strip()]
    hf = [h["faithful"].strip().lower() for _, h in pairs if h["faithful"].strip()]
    out = {
        "graded": len(pairs),
        "correctness_3way_agreement": _agree(jc, hc),
        "correctness_3way_kappa": _kappa(jc, hc),
        "correct_vs_not_agreement": _agree(jb, hb),
        "correct_vs_not_kappa": _kappa([str(x) for x in jb], [str(x) for x in hb]),
        "faithful_agreement": _agree(jf, hf),
        "faithful_kappa": _kappa(jf, hf),
        "judge_correct_rate": round(statistics.mean(jb), 4) if jb else None,
        "human_correct_rate": round(statistics.mean(hb), 4) if hb else None,
    }
    (args.run / "agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--data", type=Path, required=True)
    a = sub.add_parser("answer")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    a.add_argument("--model", default=ANSWER_MODEL)
    a.add_argument("--limit", type=int, default=None, help="stratified sample size")
    a.add_argument("--seed", type=int, default=2026)
    a.add_argument("--concurrency", type=int, default=8)
    a.add_argument("--ingest-batch", type=int, default=256)
    a.add_argument("--ranking", choices=sorted(RANKINGS), default="rrf")
    a.add_argument("--ltr-folds", type=Path, default=None, help="crossval fold-model directory")
    j = sub.add_parser("judge")
    j.add_argument("--run", type=Path, required=True)
    j.add_argument("--model", default=JUDGE_MODEL)
    j.add_argument("--concurrency", type=int, default=8)
    r = sub.add_parser("report")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--seed", type=int, default=2026)
    h = sub.add_parser("human-sample")
    h.add_argument("--run", type=Path, required=True)
    h.add_argument("--n", type=int, default=40)
    h.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("consistency")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--model", default=JUDGE_MODEL)
    c.add_argument("--n", type=int, default=100, help="answers re-graded per variant")
    c.add_argument("--seed", type=int, default=11)
    c.add_argument("--concurrency", type=int, default=8)
    g = sub.add_parser("agreement")
    g.add_argument("--run", type=Path, required=True)
    g.add_argument("--grades", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        techqa.fetch(args.data)
    elif args.cmd == "answer":
        asyncio.run(answer(args))
    elif args.cmd == "judge":
        asyncio.run(judge(args))
    elif args.cmd == "consistency":
        asyncio.run(consistency(args))
    elif args.cmd == "report":
        report(args)
    elif args.cmd == "human-sample":
        human_sample(args)
    else:
        agreement(args)


if __name__ == "__main__":
    main()
