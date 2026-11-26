"""Retrieval recall on TechQA: does the gold technote reach the answer's context?

Indexing 28k technotes takes ~15 minutes, so the candidate pools are collected
once and every ranking configuration is replayed offline through the production
``search.pipeline.retrieve_with_rerank``: a stub retriever returns the stored
hybrid (BM25 + dense, RRF) pool and a stub inference client returns stored
cross-encoder scores. Nothing about ranking is re-implemented here, so a result
is what the gateway would return with those settings.

  collect    index the corpus in-process, store the top-100 hybrid pool per
             answerable question (the same pool production fetches)
  score-ce   score every (question, chunk) pair with a cross-encoder
  train-ltr  train the learning-to-rank model on TechQA TRAIN questions only
  evaluate   recall@k per configuration; selection on TRAIN, report on DEV
  crossval   out-of-fold recall on all 610 questions for one fixed configuration

Metric: hit@k = at least one of the k chunks returned is from the gold technote.
TechQA ships TRAIN (450 answerable) and DEV (160 answerable) questions; anything
tuned is tuned on TRAIN and the headline comparison is on DEV.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np

from benchmarks.answer_quality import run as aq
from benchmarks.answer_quality import techqa
from search.hybrid_retriever import HybridResult

KS = (4, 10, 20)


def _split(qid: str) -> str:
    return qid.split("_", 1)[0]  # TRAIN | DEV


def _write_gz(path: Path, rows) -> None:
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


# -- collect ------------------------------------------------------------------


async def collect(args) -> None:
    from backend.core.services import Services

    args.work.mkdir(parents=True, exist_ok=True)
    settings = aq._settings(aq.ANSWER_MODEL)
    pool = 2 * settings.retrieval_candidates  # what retrieve_with_rerank fetches
    services = await Services.build(settings)
    try:
        docs = techqa.load_corpus(args.data)
        t0 = time.perf_counter()
        for i in range(0, len(docs), 256):
            await services.processor.ingest(docs[i : i + 256])
        print(f"indexed {len(docs)} docs in {time.perf_counter() - t0:.0f}s", flush=True)
        questions = [q for q in techqa.load_questions(args.data) if q.answerable]
        rows = []
        for q in questions:
            result = await services.retriever.search(q.question, pool)
            rows.append(
                {
                    "qid": q.qid,
                    "split": _split(q.qid),
                    "question": q.question,
                    "gold_docs": list(q.gold_docs),
                    "result": result.model_dump(),
                }
            )
        _write_gz(args.work / "pools.jsonl.gz", rows)
        meta = {
            "questions": len(rows),
            "pool": pool,
            "documents": len(docs),
            "backends": services.backends,
            "index_seconds": round(time.perf_counter() - t0, 1),
        }
        (args.work / "pools-meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))
    finally:
        await services.close()


# -- cross-encoder ------------------------------------------------------------


def _ce_path(work: Path, model: str, max_length: int) -> Path:
    return work / f"ce-{model.replace('/', '__')}-L{max_length}.json.gz"


def score_ce(args) -> None:
    from sentence_transformers import CrossEncoder

    pools = _read_gz(args.work / "pools.jsonl.gz")
    model = CrossEncoder(args.model, max_length=args.max_length, device=args.device)
    out, t0 = {}, time.perf_counter()
    for i, p in enumerate(pools, 1):
        hits = p["result"]["hits"]
        scores = model.predict(
            [(p["question"], h["chunk"]["text"]) for h in hits], batch_size=args.batch
        )
        out[p["qid"]] = {
            h["chunk"]["chunk_id"]: float(s) for h, s in zip(hits, scores, strict=True)
        }
        if i % 100 == 0:
            print(f"{i}/{len(pools)} {time.perf_counter() - t0:.0f}s", flush=True)
    with gzip.open(_ce_path(args.work, args.model, args.max_length), "wt") as f:
        json.dump({"model": args.model, "max_length": args.max_length, "scores": out}, f)


def _load_ce(work: Path, model: str, max_length: int) -> dict[str, dict[str, float]]:
    with gzip.open(_ce_path(work, model, max_length), "rt") as f:
        return json.load(f)["scores"]


# -- replay through the production pipeline ----------------------------------


class _Retriever:
    def __init__(self, result: HybridResult):
        self.result = result

    async def search(self, query: str, top_k: int = 10) -> HybridResult:
        return HybridResult(hits=self.result.hits[:top_k], timings=self.result.timings)


class _Inference:
    """Serves stored cross-encoder scores; texts map to their chunk's score."""

    def __init__(self, by_text: dict[str, float]):
        self.by_text = by_text

    async def rerank(self, query: str, passages) -> list[float]:
        return [self.by_text[t] for t in passages]


def _config(name: str, ce_model: str | None, ltr: str | None, **settings) -> dict:
    return {"name": name, "ce_model": ce_model, "ltr": ltr, "settings": settings}


def configs(ce_model: str, ltr_models: dict[str, str]) -> list[dict]:
    out = [
        _config("baseline (rrf)", None, None),
        _config("rrf, 1 chunk/doc", None, None, kb_max_chunks_per_doc=1, rerank_candidates=20),
    ]
    for n in (20, 50, 100):
        out.append(_config(f"ce@{n}", ce_model, None, rerank_enabled=True, rerank_candidates=n))
        out.append(
            _config(
                f"ce@{n}, 1 chunk/doc",
                ce_model,
                None,
                rerank_enabled=True,
                rerank_candidates=n,
                kb_max_chunks_per_doc=1,
            )
        )
    for name, path in ltr_models.items():
        for n in (20, 50):
            out.append(
                _config(
                    f"{name}@{n}, 1 chunk/doc",
                    ce_model,
                    path,
                    rerank_enabled=True,
                    rerank_candidates=n,
                    kb_max_chunks_per_doc=1,
                )
            )
    return out


async def _replay(pools, ce, cfg) -> dict[str, dict[int, bool]]:
    from backend.core.config import Settings
    from search.ltr import LtrModel
    from search.pipeline import retrieve_with_rerank

    settings = aq._settings(aq.ANSWER_MODEL).model_copy(update=cfg["settings"])
    assert isinstance(settings, Settings)
    ltr = LtrModel.load(Path(cfg["ltr"])) if cfg["ltr"] else None
    hits_at: dict[str, dict[int, bool]] = {}
    for p in pools:
        result = HybridResult.model_validate(p["result"])
        scores = ce.get(p["qid"], {}) if cfg["ce_model"] else {}
        inference = _Inference(
            {h.chunk.text: scores.get(h.chunk.chunk_id, 0.0) for h in result.hits}
        )
        gold = set(p["gold_docs"])
        hits_at[p["qid"]] = {}
        for k in KS:
            kb = await retrieve_with_rerank(
                _Retriever(result), inference, settings, p["question"], k, ltr=ltr
            )
            hits_at[p["qid"]][k] = bool(gold & {h.chunk.doc_id for h in kb.hits})
    return hits_at


# -- learning to rank on TechQA TRAIN ----------------------------------------


def _groups(pools, ce, uses_ce: bool, depth: int) -> dict:
    from backend.core.models import SearchHit
    from search.ltr import QueryGroup, features

    groups = {}
    for p in pools:
        hits = [SearchHit.model_validate(h) for h in p["result"]["hits"][:depth]]
        scores = [ce[p["qid"]][h.chunk.chunk_id] for h in hits] if uses_ce else None
        y = np.array([float(h.chunk.doc_id in p["gold_docs"]) for h in hits], dtype="float32")
        groups[p["qid"]] = QueryGroup(features(p["question"], hits, scores), y)
    return groups


def _fit(groups: dict, uses_ce: bool, seed: int, out: Path, meta: dict) -> float:
    from search.ltr import train

    qids = sorted(groups)
    random.Random(seed).shuffle(qids)
    cut = int(0.8 * len(qids))  # early stopping on a slice of the training questions
    model, info = train(
        [groups[q] for q in qids[:cut]], [groups[q] for q in qids[cut:]], uses_ce=uses_ce
    )
    model.save(out)
    (out / "training.json").write_text(json.dumps({**info, "seed": seed, **meta}, indent=2))
    return info["best_dev_ndcg10"]


def train_ltr(args) -> None:
    pools = [p for p in _read_gz(args.work / "pools.jsonl.gz") if p["split"] == "TRAIN"]
    ce = _load_ce(args.work, args.ce_model, args.ce_max_length) if args.uses_ce else {}
    best = _fit(
        _groups(pools, ce, args.uses_ce, args.depth),
        args.uses_ce,
        args.seed,
        args.out,
        {"depth": args.depth, "split": "TRAIN"},
    )
    print(f"saved {args.out}: best held-out-TRAIN nDCG@10 {best:.4f}")


def crossval(args) -> None:
    """Out-of-fold recall on all answerable questions for one fixed configuration.

    Every question is ranked by an LTR model that never saw it: the 610 questions
    are split into folds, and each fold is ranked by a model trained on the others.
    """
    pools = _read_gz(args.work / "pools.jsonl.gz")
    ce = _load_ce(args.work, args.ce_model, args.ce_max_length)
    qids = sorted(p["qid"] for p in pools)
    random.Random(args.seed).shuffle(qids)
    fold_of = {q: i % args.folds for i, q in enumerate(qids)}
    settings = {
        "rerank_enabled": True,
        "rerank_candidates": args.candidates,
        "kb_max_chunks_per_doc": 1,
    }
    base_cfg = _config("baseline (rrf)", None, None)
    base = asyncio.run(_replay(pools, ce, base_cfg))
    oof: dict[str, dict[int, bool]] = {}
    folds = []
    for f in range(args.folds):
        train_pools = [p for p in pools if fold_of[p["qid"]] != f]
        test_pools = [p for p in pools if fold_of[p["qid"]] == f]
        out = args.work / "cv" / Path(args.out).stem / f"fold-{f}"
        best = _fit(
            _groups(train_pools, ce, args.uses_ce, args.depth),
            args.uses_ce,
            args.seed + f,
            out,
            {"depth": args.depth, "fold": f, "folds": args.folds},
        )
        cfg = _config(f"fold {f}", args.ce_model, str(out), **settings)
        oof.update(asyncio.run(_replay(test_pools, ce, cfg)))
        folds.append(
            {
                "fold": f,
                "train": len(train_pools),
                "test": len(test_pools),
                "best_inner_ndcg10": round(best, 4),
            }
        )
        print(f"fold {f}: {len(test_pools)} questions ranked", flush=True)
    splits = {
        "ALL": qids,
        "TRAIN": [q for q in qids if _split(q) == "TRAIN"],
        "DEV": [q for q in qids if _split(q) == "DEV"],
    }
    report = {
        "ce_model": args.ce_model,
        "ce_max_length": args.ce_max_length,
        "uses_ce": args.uses_ce,
        "settings": settings,
        "folds": folds,
        "baseline": {s: _summary(base, q, args.seed) for s, q in splits.items()},
        "ltr_out_of_fold": {s: _summary(oof, q, args.seed) for s, q in splits.items()},
        "vs_baseline": {
            f"hit@{k}": {s: _vs_baseline(base, oof, q, k) for s, q in splits.items()} for k in KS
        },
    }
    (args.work / args.out).write_text(json.dumps(report, indent=2))
    # Which fold model never saw each question: run.py answer --ltr-folds uses it.
    (args.work / "cv" / Path(args.out).stem / "folds.json").write_text(
        json.dumps({"folds": args.folds, "fold_of": fold_of}, indent=2)
    )
    for s, q in splits.items():
        print(
            f"{s:5s} n={len(q)} baseline hit@4 {report['baseline'][s]['hit@4']['rate']:.3f}"
            f"  LTR hit@4 {report['ltr_out_of_fold'][s]['hit@4']['rate']:.3f}"
            f"  p={report['vs_baseline']['hit@4'][s]['mcnemar_p']}"
        )


# -- evaluate -----------------------------------------------------------------


def _summary(hits_at, qids, seed: int) -> dict:
    out = {}
    for k in KS:
        values = [hits_at[q][k] for q in qids]
        out[f"hit@{k}"] = aq._rate(values, seed)
    return out


def _vs_baseline(base, other, qids, k: int = 4) -> dict:
    b = sum(1 for q in qids if other[q][k] and not base[q][k])
    c = sum(1 for q in qids if base[q][k] and not other[q][k])
    return {
        "delta": round((b - c) / len(qids), 4) if qids else None,
        "gained": b,
        "lost": c,
        "mcnemar_p": aq._mcnemar(b, c),
    }


def evaluate(args) -> None:
    pools = _read_gz(args.work / "pools.jsonl.gz")
    ce = _load_ce(args.work, args.ce_model, args.ce_max_length)
    ltr_models = dict(item.split("=", 1) for item in args.ltr)
    splits = {
        "DEV": [p["qid"] for p in pools if p["split"] == "DEV"],
        "TRAIN": [p["qid"] for p in pools if p["split"] == "TRAIN"],
        "ALL": [p["qid"] for p in pools],
    }
    ceiling = {
        p["qid"]: bool(set(p["gold_docs"]) & {h["chunk"]["doc_id"] for h in p["result"]["hits"]})
        for p in pools
    }
    report = {
        "ce_model": args.ce_model,
        "ce_max_length": args.ce_max_length,
        "questions": {s: len(q) for s, q in splits.items()},
        "pool_ceiling": {
            s: aq._rate([ceiling[q] for q in qids], args.seed) for s, qids in splits.items()
        },
        "configs": [],
    }
    base = None
    for cfg in configs(args.ce_model, ltr_models):
        t0 = time.perf_counter()
        hits_at = asyncio.run(_replay(pools, ce, cfg))
        base = base or hits_at
        row = {
            "name": cfg["name"],
            "settings": cfg["settings"],
            "ltr": cfg["ltr"],
            **{s: _summary(hits_at, qids, args.seed) for s, qids in splits.items()},
            "vs_baseline_hit@4": {
                s: _vs_baseline(base, hits_at, qids) for s, qids in splits.items()
            },
        }
        report["configs"].append(row)
        print(
            f"{cfg['name']:32s} TRAIN hit@4 {row['TRAIN']['hit@4']['rate']:.3f}"
            f"  DEV hit@4 {row['DEV']['hit@4']['rate']:.3f}"
            f"  hit@10 {row['DEV']['hit@10']['rate']:.3f}"
            f"  hit@20 {row['DEV']['hit@20']['rate']:.3f}"
            f"  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    (args.work / args.out).write_text(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--work", type=Path, required=True)
    s = sub.add_parser("score-ce")
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    s.add_argument("--max-length", type=int, default=256)
    s.add_argument("--device", default="mps")
    s.add_argument("--batch", type=int, default=64)
    t = sub.add_parser("train-ltr")
    t.add_argument("--work", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    t.add_argument("--ce-max-length", type=int, default=256)
    t.add_argument("--uses-ce", action="store_true")
    t.add_argument("--depth", type=int, default=50)
    t.add_argument("--seed", type=int, default=13)
    e = sub.add_parser("evaluate")
    e.add_argument("--work", type=Path, required=True)
    e.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    e.add_argument("--ce-max-length", type=int, default=256)
    e.add_argument("--out", default="retrieval-eval.json")
    e.add_argument("--ltr", nargs="*", default=[], help="name=model_dir")
    e.add_argument("--seed", type=int, default=2026)
    x = sub.add_parser("crossval")
    x.add_argument("--work", type=Path, required=True)
    x.add_argument("--ce-model", default="BAAI/bge-reranker-base")
    x.add_argument("--ce-max-length", type=int, default=512)
    x.add_argument("--uses-ce", action="store_true")
    x.add_argument("--candidates", type=int, default=20)
    x.add_argument("--depth", type=int, default=50)
    x.add_argument("--folds", type=int, default=5)
    x.add_argument("--out", default="crossval.json")
    x.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)
    if args.cmd == "collect":
        asyncio.run(collect(args))
    elif args.cmd == "score-ce":
        score_ce(args)
    elif args.cmd == "train-ltr":
        train_ltr(args)
    elif args.cmd == "crossval":
        crossval(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
