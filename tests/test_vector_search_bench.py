from types import SimpleNamespace

import numpy as np

from benchmarks.vector_search import measure


class FakeStore:
    """Returns the true top-10 for even queries and only 5 of them for odd ones."""

    def __init__(self, truth):
        self.truth = truth

    async def search(self, vector, k):
        i = int(vector[0])
        ids = sorted(self.truth[i])
        if i % 2:
            ids = ids[:5] + [f"wrong-{n}" for n in range(5)]
        return [(SimpleNamespace(chunk_id=c), 1.0) for c in ids]


async def test_measure_reports_recall_against_exact_and_latency():
    vectors = np.array([[i, 0.0] for i in range(40)], dtype="float32")
    truth = [{f"q{i}-{n}" for n in range(10)} for i in range(40)]
    result = await measure(FakeStore(truth), vectors, truth, 10, clients=4)
    assert result["recall@10"] == 0.75  # half the queries at 1.0, half at 0.5
    assert result["qps"] > 0 and result["latency_ms"]["p50"] <= result["latency_ms"]["p99"]
