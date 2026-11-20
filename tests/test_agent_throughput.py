import asyncio
import json
from collections import Counter

import pytest

from benchmarks import agent_throughput as bench
from tests.conftest import live_server

QUESTIONS = [f"question {i} about the network queue timeout" for i in range(50)]


def test_unique_workload_never_repeats_and_storm_is_skewed():
    unique = bench.workload(QUESTIONS, "unique", 30, 0, 0, seed=1)
    assert len(set(unique)) == 30
    assert unique == bench.workload(QUESTIONS, "unique", 30, 0, 0, seed=1)  # seeded
    with pytest.raises(ValueError):
        bench.workload(QUESTIONS, "unique", 51, 0, 0, seed=1)
    storm = bench.workload(QUESTIONS, "storm", 2000, 10, 1.1, seed=1)
    counts = Counter(storm).most_common()
    assert len(counts) <= 10 and counts[0][1] > 4 * counts[-1][1]


async def test_level_counts_only_intact_answers_and_reads_metrics(tmp_path):
    async with live_server() as (_, http_url, _):
        questions = tmp_path / "q.json"
        questions.write_text(json.dumps(QUESTIONS[:6]))
        out = tmp_path / "r.json"
        await asyncio.to_thread(
            bench.main,
            [
                "--server", http_url, "--questions", str(questions), "--workload", "unique",
                "--turns", "6", "--concurrency", "3", "--label", "t", "--output", str(out),
            ],
        )  # fmt: skip
    report = json.loads(out.read_text())
    level = report["levels"][0]
    assert level["completed"] == 6 and not level["errors"] and level["turns_per_s"] > 0
    assert level["total_ms"]["p50"] is not None and report["backends"]["inference"] == "mock"
    assert report["distinct_questions"] == 6


def test_scrape_parses_llm_counters(monkeypatch):
    text = "\n".join(
        [
            "# HELP omniagent_llm_tokens_total LLM tokens",
            'omniagent_llm_tokens_total{type="prompt"} 1234.0',
            'omniagent_llm_cache_total{kind="json",outcome="hit"} 7.0',
            'omniagent_other_total{x="y"} 9.0',
        ]
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return text.encode()

    monkeypatch.setattr(bench, "urlopen", lambda url, timeout: Response())
    assert bench.scrape("http://x") == {
        'omniagent_llm_tokens_total{type="prompt"}': 1234.0,
        'omniagent_llm_cache_total{kind="json",outcome="hit"}': 7.0,
    }
