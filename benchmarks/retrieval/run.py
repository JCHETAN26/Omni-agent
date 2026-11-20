"""Retrieval-quality and latency benchmark on BEIR CQADupStack.

Phases (each reads the previous phase's output from --work):

  embed     chunk + embed the corpus only (e.g. on a GPU laptop: --device mps);
            each cached vector batch records a hash of its chunk ids
  index     embed (or load verified cached vectors), then load the same vectors
            into Faiss and/or Qdrant, and the text into Elasticsearch
  collect   run every judged query through OmniAgent's HybridRetriever, score the
            top --pool candidates with the served cross-encoder (or skip that with
            --no-ce), and save every candidate's features and chunk ids
  score-ce  fill cross-encoder features offline with the same model weights
  evaluate  split queries 60/20/20 by id hash, train the PyTorch LTR model on
            train (early stop on dev), report every ranker on the held-out test
            split, with BEIR-style per-forum macro averages and paired tests
  latency   load-test /search and /search/ranked on a running gateway

Example (Codespace, native Qdrant + Elasticsearch already running):

  python -m benchmarks.retrieval.run index    --work /data/beir --dense faiss qdrant
  python -m benchmarks.retrieval.run collect  --work /data/beir --dense faiss
  python -m benchmarks.retrieval.run evaluate --work /data/beir
  python -m benchmarks.retrieval.run latency  --server http://127.0.0.1:8000 --work /data/beir
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

from backend.core.config import Settings
from backend.core.models import Chunk, SearchHit
from benchmarks.retrieval import beir
from benchmarks.retrieval.metrics import evaluate as score_run
from benchmarks.retrieval.metrics import per_query_ndcg

SPLITS = ("train", "dev", "test")


def split_of(query_id: str) -> str:
    bucket = int(hashlib.sha256(query_id.encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else ("dev" if bucket < 8 else "test")


def _settings(args, **overrides) -> Settings:
    base = dict(
        embedding_provider=args.embedding_provider,
        qdrant_url=os.environ.get("QDRANT_URL"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
        qdrant_collection=args.index_name,
        elasticsearch_url=os.environ.get("ELASTICSEARCH_URL"),
        elasticsearch_api_key=os.environ.get("ELASTICSEARCH_API_KEY"),
        elasticsearch_index=args.index_name,
        faiss_path=str(Path(args.work) / "faiss"),
        embedding_device=args.device,
    )
    return Settings(_env_file=None, **{**base, **overrides})


def _dataset(args) -> beir.Dataset:
    root = beir.download_cqadupstack(Path(args.work) / "data", args.forums)
    return beir.load(root, args.forums)


# ----------------------------------------------------------------------------
# index
# ----------------------------------------------------------------------------


def _ids_digest(chunks: list[Chunk]) -> str:
    return hashlib.sha256("\n".join(c.chunk_id for c in chunks).encode()).hexdigest()


async def run_index(args, embed_only: bool = False) -> None:
    """Chunk and embed the corpus; unless ``embed_only``, also load every store.

    Vectors are cached per batch with a hash of the batch's chunk ids, so they
    can be computed on one machine (e.g. a GPU laptop) and loaded on another;
    a cached batch whose chunk ids differ is rejected instead of misloaded."""
    from backend.core.stores import (
        dense_store_from_settings,
        sparse_index_from_settings,
    )
    from backend.ingestion.chunker import chunk_document
    from backend.ingestion.embedder import embedder_from_settings

    work = Path(args.work)
    data = _dataset(args)
    settings = _settings(args)
    embedder = embedder_from_settings(settings)
    stores = (
        {}
        if embed_only
        else {
            name: dense_store_from_settings(_settings(args, dense_backend=name))
            for name in args.dense
        }
    )
    sparse = None if embed_only else sparse_index_from_settings(settings)
    if sparse is not None:
        await sparse.ensure_index()
    vectors_dir = work / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "documents": 0,
        "chunks": 0,
        "embed_s": 0.0,
        "dense_s": {n: 0.0 for n in stores},
        "sparse_s": 0.0,
    }
    started = time.perf_counter()
    batch: list[Chunk] = []
    part = 0

    async def flush() -> None:
        nonlocal part
        cache = vectors_dir / f"part-{part:05d}.npy"
        ids_file = cache.with_suffix(".ids")
        digest = _ids_digest(batch)
        if args.wait_for_vectors:
            # Vectors are produced elsewhere and arrive part by part; the .ids
            # file is moved into place last, so its presence means the part is whole.
            deadline = time.monotonic() + args.wait_for_vectors
            while not (cache.exists() and ids_file.exists()):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{cache.name} did not arrive")
                await asyncio.sleep(5)
        if cache.exists():
            if not ids_file.exists() or ids_file.read_text().strip() != digest:
                raise RuntimeError(f"{cache.name} was embedded from different chunks; re-embed")
            vectors = np.load(cache).astype("float32")
        else:
            t = time.perf_counter()
            vectors = np.asarray(await embedder.embed([c.text for c in batch]), dtype="float32")
            totals["embed_s"] += time.perf_counter() - t
            np.save(cache, vectors.astype("float16"))
            ids_file.write_text(digest + "\n")
        rows = vectors.tolist()

        async def dense_write(name: str, store) -> None:
            await store.ensure_collection(vectors.shape[1])
            t = time.perf_counter()
            await store.upsert(batch, rows)
            totals["dense_s"][name] += time.perf_counter() - t

        async def sparse_write() -> None:
            t = time.perf_counter()
            await sparse.index(batch)
            totals["sparse_s"] += time.perf_counter() - t

        # The stores are independent, so a batch is written to all of them at once.
        writes = [dense_write(name, store) for name, store in stores.items()]
        if sparse is not None:
            writes.append(sparse_write())
        await asyncio.gather(*writes)
        totals["chunks"] += len(batch)
        part += 1
        batch.clear()
        if part % 20 == 0:
            rate = totals["chunks"] / (time.perf_counter() - started)
            print(
                f"indexed {totals['documents']:,} docs / {totals['chunks']:,} chunks"
                f" ({rate:.0f}/s)",
                flush=True,
            )

    for doc in data.documents():
        totals["documents"] += 1
        batch.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))
        if len(batch) >= args.batch_size:
            await flush()
    if batch:
        await flush()
    for store in stores.values():
        await store.close()  # Faiss persists its index here
    if sparse is not None:
        await sparse.close()
    totals["wall_s"] = time.perf_counter() - started
    report = {
        "host": platform.platform(),
        "cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "forums": list(args.forums),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "dense_backends": list(stores),
        "index_name": args.index_name,
        **totals,
    }
    (work / ("embed.json" if embed_only else "index.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


# ----------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------


async def run_collect(args) -> None:
    from backend.core.stores import dense_store_from_settings, sparse_index_from_settings
    from backend.ingestion.embedder import embedder_from_settings
    from model_server.inference_client import inference_client_from_settings
    from search import ltr
    from search.hybrid_retriever import HybridRetriever

    work = Path(args.work)
    data = _dataset(args)
    settings = _settings(args, dense_backend=args.dense[0], inference_backend=args.inference)
    embedder = embedder_from_settings(settings)
    dense = dense_store_from_settings(settings)
    sparse = sparse_index_from_settings(settings)
    inference = None if args.no_ce else inference_client_from_settings(settings)
    if inference is not None:
        await inference.warmup()
    retriever = HybridRetriever(embedder, dense, sparse, settings.rrf_k, candidates=args.candidates)

    queries = sorted(data.qrels)
    if args.limit:
        queries = queries[: args.limit]
    out_dir = work / f"collect-{args.dense[0]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    timings: list[dict] = []
    started = time.perf_counter()
    done = 0

    async def one(query_id: str) -> dict:
        nonlocal done
        query = data.queries[query_id]
        async with sem:
            result = await retriever.search(query, top_k=2 * args.candidates)
            hits: list[SearchHit] = result.hits
            pool = hits[: args.pool]
            t = time.perf_counter()
            ce = (
                await inference.rerank(query, [h.chunk.text for h in pool])
                if pool and inference is not None
                else [None] * len(pool)  # scored later by the score-ce phase
            )
            ce_ms = (time.perf_counter() - t) * 1000
        ce_scores = list(ce) + [None] * (len(hits) - len(pool))
        timings.append(
            {"retrieval": result.timings.model_dump(), "ce_ms": ce_ms, "pool": len(pool)}
        )
        done += 1
        if done % 500 == 0:
            print(
                f"collected {done:,}/{len(queries):,} queries"
                f" ({done / (time.perf_counter() - started):.1f}/s)",
                flush=True,
            )
        return {
            "query_id": query_id,
            "x": ltr.features(query, hits, ce_scores),
            "doc_ids": [h.chunk.doc_id for h in hits],
            "chunk_ids": [h.chunk.chunk_id for h in hits],
            "dense_rank": [h.dense_rank or 0 for h in hits],
            "sparse_rank": [h.sparse_rank or 0 for h in hits],
            "pool": len(pool),
        }

    rows = await asyncio.gather(*(one(q) for q in queries))
    if inference is not None:
        await inference.close()
    await dense.close()
    await sparse.close()
    lengths = np.array([len(r["doc_ids"]) for r in rows])
    np.savez_compressed(
        out_dir / "candidates.npz",
        query_ids=np.array([r["query_id"] for r in rows]),
        lengths=lengths,
        pools=np.array([r["pool"] for r in rows]),
        x=np.concatenate([r["x"] for r in rows]) if rows else np.zeros((0, len(ltr.FEATURES))),
        doc_ids=np.array([d for r in rows for d in r["doc_ids"]]),
        chunk_ids=np.array([c for r in rows for c in r["chunk_ids"]]),
        ce_scored=np.array(inference is not None),
        dense_rank=np.array([d for r in rows for d in r["dense_rank"]]),
        sparse_rank=np.array([d for r in rows for d in r["sparse_rank"]]),
    )
    summary = {
        "queries": len(rows),
        "candidates": args.candidates,
        "pool": args.pool,
        "wall_s": time.perf_counter() - started,
        "retrieval_ms_p50": statistics.median(t["retrieval"]["total_ms"] for t in timings),
        "ce_ms_p50": statistics.median(t["ce_ms"] for t in timings) if inference else None,
    }
    (out_dir / "collect.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


# ----------------------------------------------------------------------------
# score-ce
# ----------------------------------------------------------------------------


def run_score_ce(args) -> None:
    """Fill cross-encoder features for collected pools, on any device.

    Uses the same model and pair tokenisation as serving (query/passage pair,
    longest-first truncation to 256 tokens, raw logits); export_onnx.py checks
    the served ONNX graph against these PyTorch weights. Chunk texts are rebuilt
    from the corpus, since chunk ids are deterministic.
    """
    from sentence_transformers import CrossEncoder
    from torch import nn

    from backend.ingestion.chunker import chunk_document
    from search import ltr

    work = Path(args.work)
    path = work / f"collect-{args.dense[0]}" / "candidates.npz"
    z = dict(np.load(path, allow_pickle=False))
    data = _dataset(args)
    settings = _settings(args)
    offsets = np.concatenate([[0], np.cumsum(z["lengths"])])
    needed = {
        z["chunk_ids"][j]
        for i in range(len(z["query_ids"]))
        for j in range(offsets[i], offsets[i] + z["pools"][i])
    }
    texts: dict[str, str] = {}
    for doc in data.documents():
        for chunk in chunk_document(doc, settings.chunk_size, settings.chunk_overlap):
            if chunk.chunk_id in needed:
                texts[chunk.chunk_id] = chunk.text
    missing = needed.difference(texts)
    if missing:
        raise RuntimeError(f"{len(missing)} pooled chunks are not in the local corpus")

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256, device=args.device)
    pairs, rows = [], []
    for i, query_id in enumerate(z["query_ids"].tolist()):
        for j in range(offsets[i], offsets[i] + z["pools"][i]):
            pairs.append((data.queries[query_id], texts[z["chunk_ids"][j]]))
            rows.append(j)
    started = time.perf_counter()
    scores = model.predict(
        pairs, batch_size=128, activation_fn=nn.Identity(), show_progress_bar=False
    )
    x = z["x"]
    ce_col, gap_col = ltr.FEATURES.index("ce_score"), ltr.FEATURES.index("ce_gap")
    x[rows, ce_col] = scores
    for i in range(len(z["query_ids"])):
        a, b = offsets[i], offsets[i] + z["pools"][i]
        if b > a:
            x[a:b, gap_col] = x[a:b, ce_col] - x[a:b, ce_col].max()
    z["x"], z["ce_scored"] = x, np.array(True)
    np.savez_compressed(path, **z)
    report = {"pairs": len(pairs), "device": args.device, "wall_s": time.perf_counter() - started}
    (path.parent / "score-ce.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


# ----------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------


def _load_candidates(path: Path):
    # Read each array once: indexing an NpzFile decompresses the whole array on
    # every access, and a slice of that copy would pin all of it per query.
    with np.load(path, allow_pickle=False) as npz:
        z = {
            k: npz[k]
            for k in ("query_ids", "lengths", "pools", "x", "doc_ids", "dense_rank", "sparse_rank")
        }
    offsets = np.concatenate([[0], np.cumsum(z["lengths"])])
    doc_ids = z["doc_ids"].tolist()
    groups = {}
    for i, query_id in enumerate(z["query_ids"].tolist()):
        a, b = offsets[i], offsets[i + 1]
        groups[query_id] = {
            "x": z["x"][a:b],
            "doc_ids": doc_ids[a:b],
            "dense_rank": z["dense_rank"][a:b],
            "sparse_rank": z["sparse_rank"][a:b],
            "pool": int(z["pools"][i]),
        }
    return groups


def _dedupe(doc_ids: list[str], order) -> list[str]:
    seen, out = set(), []
    for i in order:
        d = doc_ids[i]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def rankings(group: dict, ltr_scores: dict[str, np.ndarray]) -> dict[str, list[str]]:
    """Chunk candidates (in RRF order) -> document rankings for every system.

    Re-rankers reorder only the first ``pool`` candidates, as the serving
    pipeline does; the rest keep RRF order behind them.
    """
    from search.ltr import FEATURES

    n, pool = len(group["doc_ids"]), group["pool"]
    rrf = list(range(n))

    def by_rank(ranks) -> list[int]:
        return [i for i in sorted(range(n), key=lambda i: ranks[i]) if ranks[i] > 0]

    ce_col = group["x"][:pool, FEATURES.index("ce_score")]
    systems = {
        "bm25": by_rank(group["sparse_rank"]),
        "dense": by_rank(group["dense_rank"]),
        "rrf": rrf,
        "rrf+cross-encoder": list(np.argsort(-ce_col, kind="stable")) + rrf[pool:],
    }
    for name, scores in ltr_scores.items():
        systems[name] = list(np.argsort(-scores[:pool], kind="stable")) + rrf[pool:]
    return {name: _dedupe(group["doc_ids"], order) for name, order in systems.items()}


def paired_randomization(
    a: dict[str, float], b: dict[str, float], trials: int = 10000, seed: int = 7
) -> float:
    """Two-sided p-value for mean(b - a) under random sign flips of per-query deltas."""
    deltas = np.array([b[q] - a[q] for q in a])
    observed = abs(deltas.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(trials, len(deltas)))
    return float(((np.abs((signs * deltas).mean(axis=1)) >= observed).sum() + 1) / (trials + 1))


def _macro_by_forum(qrels, run, forum_of, same_forum: bool) -> dict:
    forums = sorted(set(forum_of.values()))
    per = {}
    for forum in forums:
        q = {qid: rel for qid, rel in qrels.items() if forum_of[qid] == forum}
        r = {qid: [d for d in run[qid] if not same_forum or d.startswith(forum + "/")] for qid in q}
        if q:
            per[forum] = score_run(q, r, ks=(10,))["ndcg@10"]
    return {"macro_ndcg@10": float(np.mean(list(per.values()))), "per_forum": per}


def run_evaluate(args) -> None:
    from search import ltr

    work = Path(args.work)
    data = _dataset(args)
    candidates = work / f"collect-{args.dense[0]}" / "candidates.npz"
    if not bool(np.load(candidates)["ce_scored"]):
        raise RuntimeError("candidates lack cross-encoder scores; run the score-ce phase first")
    groups = _load_candidates(candidates)
    split = {q: split_of(q) for q in groups}

    def labelled(query_ids):
        out = []
        for q in query_ids:
            g = groups[q]
            pool = g["pool"]
            rel = data.qrels[q]
            y = np.array([rel.get(d, 0) for d in g["doc_ids"][:pool]], dtype="float32")
            # BEIR ignores a question retrieving itself; never train on it either.
            keep = np.array([d != q for d in g["doc_ids"][:pool]])
            out.append(ltr.QueryGroup(g["x"][:pool][keep], y[keep]))
        return out

    by_split = {s: [q for q in groups if split[q] == s] for s in SPLITS}
    models, reports = {}, {}
    for name, uses_ce in (("ltr", False), ("ltr+cross-encoder", True)):
        print(f"training {name}", flush=True)
        model, report = ltr.train(
            labelled(by_split["train"]),
            labelled(by_split["dev"]),
            uses_ce=uses_ce,
            epochs=args.epochs,
        )
        model.save(work / "models" / name.replace("+", "_"))
        models[name], reports[name] = model, report

    test_ids = by_split["test"]
    runs: dict[str, dict[str, list[str]]] = {}
    for q in test_ids:
        g = groups[q]
        scores = {name: m.score(g["x"][: g["pool"]]) for name, m in models.items()}
        for system, ranking in rankings(g, scores).items():
            runs.setdefault(system, {})[q] = ranking
    qrels_test = {q: data.qrels[q] for q in test_ids}
    forum_of = {q: data.forum_of[q] for q in test_ids}
    results = {}
    for system, run in runs.items():
        results[system] = {
            **score_run(qrels_test, run, ks=(10, 100)),
            "combined_index_by_forum": _macro_by_forum(qrels_test, run, forum_of, same_forum=False),
            "same_forum_by_forum": _macro_by_forum(qrels_test, run, forum_of, same_forum=True),
        }
    base = per_query_ndcg(qrels_test, runs["rrf"])
    significance = {
        system: {
            "delta_ndcg@10_vs_rrf": results[system]["ndcg@10"] - results["rrf"]["ndcg@10"],
            "relative_gain_vs_rrf": results[system]["ndcg@10"] / results["rrf"]["ndcg@10"] - 1,
            "relative_gain_vs_bm25": results[system]["ndcg@10"] / results["bm25"]["ndcg@10"] - 1,
            "p_value_vs_rrf": paired_randomization(base, per_query_ndcg(qrels_test, run)),
        }
        for system, run in runs.items()
        if system != "rrf"
    }
    summary = {
        "dense_backend": args.dense[0],
        "queries": {s: len(v) for s, v in by_split.items()},
        "judged_test_queries": sum(
            1 for q in test_ids if any(r > 0 for r in data.qrels[q].values())
        ),
        "results": results,
        "significance": significance,
        "training": reports,
    }
    (work / f"evaluate-{args.dense[0]}.json").write_text(json.dumps(summary, indent=2) + "\n")
    for system, r in results.items():
        print(
            f"{system:22s} nDCG@10 {r['ndcg@10']:.4f}  MRR@10 {r['mrr@10']:.4f}  "
            f"R@100 {r['recall@100']:.4f}  "
            f"macro(same-forum) {r['same_forum_by_forum']['macro_ndcg@10']:.4f}",
            flush=True,
        )


# ----------------------------------------------------------------------------
# latency
# ----------------------------------------------------------------------------


def _pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))]


_PERCENTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))


async def _load_level(client, path: str, queries: list[str], clients: int, requests: int) -> dict:
    """Closed-loop load: ``clients`` workers share ``requests`` queries."""
    latencies: list[float] = []
    server_ms: list[float] = []
    errors = 0
    cursor = iter(range(requests))

    async def worker() -> None:
        nonlocal errors
        for i in cursor:
            t = time.perf_counter()
            r = await client.post(path, json={"query": queries[i % len(queries)], "top_k": 10})
            latencies.append((time.perf_counter() - t) * 1000)
            if r.status_code != 200:
                errors += 1
                continue
            server_ms.append(r.json()["timings"]["total_ms"])

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(clients)))
    wall = time.perf_counter() - t0
    return {
        "path": path,
        "clients": clients,
        "requests": len(latencies),
        "errors": errors,
        "qps": round(len(latencies) / wall, 2),
        "client_ms": {p: round(_pct(latencies, q), 2) for p, q in _PERCENTILES},
        "server_ms": (
            {p: round(_pct(server_ms, q), 2) for p, q in _PERCENTILES} if server_ms else None
        ),
    }


async def run_latency(args) -> None:
    import httpx

    data = _dataset(args)
    queries = [data.queries[q] for q in sorted(data.qrels)]
    random.Random(11).shuffle(queries)
    report = {"server": args.server, "requests_per_level": args.requests, "levels": []}
    async with httpx.AsyncClient(base_url=args.server, timeout=60) as client:
        report["backends"] = (await client.get("/health")).json().get("backends")
        for path in args.paths:
            for _ in range(20):  # warm caches, model sessions and connection pools
                await client.post(path, json={"query": queries[0], "top_k": 10})
            for clients in args.concurrency:
                level = await _load_level(client, path, queries, clients, args.requests)
                report["levels"].append(level)
                print(json.dumps(level), flush=True)
    out = Path(args.work) / f"latency-{args.label}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")


# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "phase", choices=("embed", "index", "collect", "score-ce", "evaluate", "latency")
    )
    parser.add_argument(
        "--work", required=True, help="working directory for data, vectors and results"
    )
    parser.add_argument("--forums", nargs="+", default=list(beir.CQADUPSTACK_FORUMS))
    parser.add_argument("--dense", nargs="+", default=["faiss"], choices=("faiss", "qdrant"))
    parser.add_argument("--index-name", default="beir_cqadupstack")
    parser.add_argument(
        "--embedding-provider",
        default="sentence-transformers",
        choices=("sentence-transformers", "hashing"),
    )
    parser.add_argument("--inference", default="triton", choices=("triton", "mock"))
    parser.add_argument("--device", default=None, help="torch device for embed/score-ce")
    parser.add_argument("--no-ce", action="store_true", help="collect without the reranker")
    parser.add_argument(
        "--wait-for-vectors",
        type=float,
        default=0,
        help="index: wait up to this many seconds per part for vectors embedded elsewhere",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="chunks per embed/ingest batch"
    )
    parser.add_argument("--candidates", type=int, default=100, help="per-branch retrieval depth")
    parser.add_argument("--pool", type=int, default=50, help="candidates the re-rankers reorder")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[4])
    parser.add_argument("--limit", type=int, default=0, help="collect only the first N queries")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--paths", nargs="+", default=["/search", "/search/ranked"])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--label", default="run")
    args = parser.parse_args(argv)
    if args.phase == "collect":
        args.concurrency = args.concurrency[0]
    if args.phase in ("embed", "index"):
        asyncio.run(run_index(args, embed_only=args.phase == "embed"))
    elif args.phase == "score-ce":
        run_score_ce(args)
    elif args.phase == "collect":
        asyncio.run(run_collect(args))
    elif args.phase == "evaluate":
        run_evaluate(args)
    else:
        asyncio.run(run_latency(args))


if __name__ == "__main__":
    main()
