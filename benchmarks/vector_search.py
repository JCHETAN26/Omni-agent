"""Qdrant search tuning: recall@10 against exact search, and QPS under load.

Uses OmniAgent's own QdrantDenseStore against a populated collection. Ground
truth is Qdrant's exact (brute-force) search for the same query vectors, so
recall measures only what approximation (HNSW, quantization) costs.

Each configuration is measured with the same query vectors at a fixed number of
concurrent clients. int8 quantization is enabled on the collection once, up front;
"none" configurations then bypass it explicitly, so every configuration searches
the same collection and the configurations can be interleaved. Each round runs
every configuration once (in a rotated order) and the report keeps the median
of each metric across rounds, which removes order and cache-warmth effects.

  python -m benchmarks.vector_search --collection beir_cqadupstack \\
      --queries query_vectors.npy --concurrency 16 --output results/vector-search.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np

CONFIGS = (
    # (label, quantization, hnsw_ef)
    ("hnsw default", "none", None),
    ("hnsw ef=64", "none", 64),
    ("hnsw ef=32", "none", 32),
    ("int8 + rescore, ef=64", "int8", 64),
    ("int8 + rescore, ef=32", "int8", 32),
)


def _store(collection: str, quantization: str, hnsw_ef: int | None):
    from backend.core.config import Settings
    from backend.core.stores import QdrantDenseStore

    settings = Settings(
        _env_file=None,
        qdrant_url=os.environ["QDRANT_URL"],
        qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
        qdrant_collection=collection,
        qdrant_quantization=quantization,
        qdrant_hnsw_ef=hnsw_ef,
    )
    return QdrantDenseStore.from_settings(settings)


async def _exact(store, vectors: np.ndarray, k: int) -> list[set[str]]:
    from qdrant_client.models import SearchParams

    truth = []
    for v in vectors:
        resp = await store.client.query_points(
            store.collection, query=v.tolist(), limit=k, search_params=SearchParams(exact=True)
        )
        truth.append({str(p.id) for p in resp.points})
    return truth


async def _wait_green(store, timeout_s: float = 1800) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = await store.client.get_collection(store.collection)
        if str(info.status).lower().endswith("green"):
            return
        await asyncio.sleep(5)
    raise RuntimeError("collection did not return to green")


async def measure(store, vectors: np.ndarray, truth: list[set[str]], k: int, clients: int) -> dict:
    queue = list(range(len(vectors)))
    latencies: list[float] = []
    found: dict[int, set[str]] = {}

    async def worker() -> None:
        while queue:
            i = queue.pop()
            t = time.perf_counter()
            hits = await store.search(vectors[i].tolist(), k)
            latencies.append((time.perf_counter() - t) * 1000)
            found[i] = {c.chunk_id for c, _ in hits}

    for v in vectors[:20]:  # warm-up
        await store.search(v.tolist(), k)
    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(clients)))
    wall = time.perf_counter() - t0
    recall = np.mean([len(found[i] & truth[i]) / k for i in range(len(vectors))])
    ordered = sorted(latencies)
    pick = lambda p: round(ordered[round(p * (len(ordered) - 1))], 2)  # noqa: E731
    return {
        "recall@10": round(float(recall), 4),
        "qps": round(len(vectors) / wall, 1),
        "latency_ms": {"p50": pick(0.5), "p95": pick(0.95), "p99": pick(0.99)},
    }


async def run(args) -> dict:
    vectors = np.load(args.queries).astype("float32")[: args.limit]
    base = _store(args.collection, "none", None)
    quantizer = _store(args.collection, "int8", None)
    await quantizer.apply_quantization()  # idempotent; Qdrant rebuilds once
    await _wait_green(base)
    info = await base.client.get_collection(args.collection)
    truth = await _exact(base, vectors, args.k)
    rounds: dict[str, list[dict]] = {label: [] for label, _, _ in CONFIGS}
    for r in range(args.rounds):
        order = CONFIGS[r % len(CONFIGS) :] + CONFIGS[: r % len(CONFIGS)]
        for label, quantization, ef in order:
            store = _store(args.collection, quantization, ef)
            result = await measure(store, vectors, truth, args.k, args.concurrency)
            rounds[label].append(result)
            print(json.dumps({"round": r + 1, "config": label, **result}), flush=True)
            await store.close()
    med = lambda xs: round(float(np.median(xs)), 2)  # noqa: E731
    report = {
        "collection": args.collection,
        "points": info.points_count,
        "queries": len(vectors),
        "clients": args.concurrency,
        "rounds": args.rounds,
        "configs": [
            {
                "config": label,
                "recall@10": med([x["recall@10"] for x in runs]),
                "qps_median": med([x["qps"] for x in runs]),
                "qps_min_max": [min(x["qps"] for x in runs), max(x["qps"] for x in runs)],
                "latency_ms_median": {
                    p: med([x["latency_ms"][p] for x in runs]) for p in ("p50", "p95", "p99")
                },
            }
            for label, runs in rounds.items()
        ],
        "raw": rounds,
    }
    await quantizer.close()
    await base.close()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--queries", type=Path, required=True, help=".npy of query vectors")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
