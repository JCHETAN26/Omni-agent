"""Agent throughput under concurrent load, through the real /sse/agent endpoint.

Every turn goes through the CLI's ``ask``, which verifies the SSE event
sequence, token count and answer checksum, so a turn only counts when the full
answer arrived intact. Each turn is a fresh conversation.

Workloads (same seeded sequence for every configuration being compared):

  unique  every turn asks a different question: caching cannot help, so this
          measures the cache's overhead
  storm   turns are drawn from --distinct questions with a Zipf(--zipf-s)
          distribution, as when many engineers ask about the same outage

Prometheus counters are sampled before and after each level, so LLM tokens and
cache hits/misses/coalesced calls are exact per level.

The LLM cache persists across levels on one gateway. To compare configurations
from a cold cache, run one --concurrency level per gateway start.

  python -m benchmarks.agent_throughput --server http://127.0.0.1:8000 \\
      --questions questions.json --workload storm --turns 300 --concurrency 4 16 \\
      --label cache-on --output results/agent-cache-on.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

_METRIC = re.compile(r"^(omniagent_llm_(?:tokens|cache)_total)\{([^}]*)\} ([0-9.e+]+)$")


def scrape(server: str) -> dict[str, float]:
    """LLM token and cache counters from /metrics, keyed 'name{labels}'."""
    with urlopen(f"{server}/metrics", timeout=10) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        if m := _METRIC.match(line):
            out[f"{m.group(1)}{{{m.group(2)}}}"] = float(m.group(3))
    return out


def workload(questions: list[str], kind: str, turns: int, distinct: int, s: float, seed: int):
    rng = random.Random(seed)
    if kind == "unique":
        if turns > len(questions):
            raise ValueError(f"unique workload needs {turns} questions, have {len(questions)}")
        return rng.sample(questions, turns)
    pool = rng.sample(questions, distinct)
    weights = [1 / (rank**s) for rank in range(1, distinct + 1)]
    return rng.choices(pool, weights=weights, k=turns)


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))], 2)


def run_level(server: str, sequence: list[str], clients: int) -> dict:
    from benchmarks.agent_client import ask

    def one(question: str) -> dict:
        try:
            r = ask(server, question)
            return {
                "ok": True,
                "ttft": r["ttft_ms"],
                "total": r["total_ms"],
                "outcome": r["outcome"],
            }
        except Exception as exc:  # noqa: BLE001 - every failure counts against throughput
            return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:160]}

    before = scrape(server)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        results = list(pool.map(one, sequence))
    wall = time.perf_counter() - t0
    after = scrape(server)
    ok = [r for r in results if r["ok"]]
    deltas = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(after) | set(before)}
    return {
        "clients": clients,
        "turns": len(sequence),
        "completed": len(ok),
        "errors": Counter(r["error"].split(":")[0] for r in results if not r["ok"]),
        "error_samples": [r["error"] for r in results if not r["ok"]][:3],
        "wall_s": round(wall, 2),
        "turns_per_s": round(len(ok) / wall, 3),
        "ttft_ms": {
            p: _pct([r["ttft"] for r in ok if r["ttft"]], q)
            for p, q in (("p50", 0.5), ("p95", 0.95))
        },
        "total_ms": {
            p: _pct([r["total"] for r in ok], q)
            for p, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
        "outcomes": Counter(r["outcome"] for r in ok),
        "llm": {k: round(v) for k, v in sorted(deltas.items()) if v},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--questions", type=Path, required=True, help="JSON list of question strings"
    )
    parser.add_argument("--workload", choices=("unique", "storm"), required=True)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--distinct", type=int, default=40)
    parser.add_argument("--zipf-s", type=float, default=1.1)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    questions = json.loads(args.questions.read_text())
    sequence = workload(questions, args.workload, args.turns, args.distinct, args.zipf_s, args.seed)
    with urlopen(f"{args.server}/health", timeout=10) as response:
        backends = json.load(response).get("backends")
    report = {
        "label": args.label,
        "workload": args.workload,
        "distinct_questions": len(set(sequence)),
        "zipf_s": args.zipf_s if args.workload == "storm" else None,
        "seed": args.seed,
        "backends": backends,
        "levels": [],
    }
    for clients in args.concurrency:
        level = run_level(args.server, sequence, clients)
        report["levels"].append(level)
        print(json.dumps({k: v for k, v in level.items() if k != "error_samples"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
