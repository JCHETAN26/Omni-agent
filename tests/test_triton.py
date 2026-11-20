"""Real tritonclient.grpc.aio client against the local Triton-protocol server."""

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
from opentelemetry import trace

from model_server.inference_client import InferenceError, TritonInferenceClient, bucket_length
from model_server.local_triton import LocalTritonServer, load_config

REPO = Path(__file__).resolve().parent.parent / "model_server" / "model_repository"


@pytest.fixture
async def triton():
    server = LocalTritonServer(use_onnx=False, synthetic_base_ms=1.0, synthetic_per_item_ms=0.05)
    port = await server.start()
    client = TritonInferenceClient(f"127.0.0.1:{port}")
    yield server, client
    await client.close()
    await server.stop()


def test_model_configs_declare_dynamic_batching():
    for name in ("minilm_embedder", "cross_encoder_reranker"):
        cfg = load_config(REPO / name)
        assert cfg.name == name
        assert cfg.backend == "onnxruntime"
        assert cfg.max_batch_size == 64
        assert cfg.dynamic_batching.max_queue_delay_microseconds == 5000
    emb = load_config(REPO / "minilm_embedder")
    assert [i.name for i in emb.input] == ["input_ids", "attention_mask"]
    assert emb.output[0].name == "sentence_embedding" and list(emb.output[0].dims) == [384]


def test_bucket_length():
    assert [bucket_length(n) for n in (1, 32, 33, 100, 256, 999)] == [32, 32, 64, 128, 256, 256]


async def test_readiness(triton):
    server, client = triton
    assert await client.ready()
    assert await client.client.is_model_ready("cross_encoder_reranker")
    assert not await client.client.is_model_ready("nope")


async def test_embed_shapes_and_normalisation(triton):
    _, client = triton
    vecs = await client.embed(["disk failure", "disk failure", "thermal throttling on rack 4"])
    assert vecs.shape == (3, 384) and vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(vecs[0], vecs[1])


async def test_large_input_is_split_into_max_batch_requests(triton):
    server, client = triton
    vecs = await client.embed([f"sentence number {i}" for i in range(150)])
    assert vecs.shape == (150, 384)
    assert server.stats()["minilm_embedder"]["requests"] == 3  # 64 + 64 + 22


async def test_dynamic_batching_merges_concurrent_requests(triton):
    server, client = triton
    results = await asyncio.gather(*(client.embed([f"query {i}"]) for i in range(128)))
    assert all(r.shape == (1, 384) for r in results)
    stats = server.stats()["minilm_embedder"]
    assert stats["requests"] == 128 and stats["items"] == 128
    assert stats["batches"] < 128 // 4  # many requests per executed batch
    assert stats["max_batch_size"] > 1


async def test_different_length_buckets_are_not_merged(triton):
    server, client = triton
    long_text = " ".join(["word"] * 50)  # > 32 tokens -> 64 bucket
    await asyncio.gather(client.embed(["short"]), client.embed([long_text]))
    assert server.stats()["minilm_embedder"]["batches"] == 2


async def test_rerank_prefers_overlapping_passage(triton):
    _, client = triton
    scores = await client.rerank(
        "disk controller timeout",
        ["power supply redundancy lost", "the disk controller hit a timeout", "dns ndots"],
    )
    assert len(scores) == 3
    assert int(np.argmax(scores)) == 1


async def test_trace_context_propagates_over_grpc(triton, spans):
    server, client = triton
    with trace.get_tracer("test").start_as_current_span("test.root") as root:
        await client.embed(["trace me"])
    trace_id = format(root.get_span_context().trace_id, "032x")
    assert any(tp.split("-")[1] == trace_id for tp in server.received_traceparents)
    finished = {s.name: s for s in spans.get_finished_spans()}
    client_span, server_span = finished["triton.infer"], finished["triton.server.ModelInfer"]
    assert client_span.context.trace_id == server_span.context.trace_id == root.context.trace_id
    assert server_span.parent.span_id == client_span.context.span_id
    assert client_span.attributes["triton.model"] == "minilm_embedder"


async def test_errors_surface_as_inference_error(triton):
    _, client = triton
    client.embed_model = "missing_model"
    with pytest.raises(InferenceError, match="missing_model"):
        await client.embed(["x"])
    dead = TritonInferenceClient("127.0.0.1:1", timeout_s=0.5)
    with pytest.raises(InferenceError):
        await dead.embed(["x"])
    assert not await dead.ready()
    await dead.close()


async def test_event_loop_stays_responsive_under_load(triton, record_property):
    _, client = triton
    # Production loads tokenizers at startup (Services.build -> warmup) before serving;
    # the first-use load holds the GIL in its worker thread and is not what we measure.
    await client.warmup()
    lags: list[float] = []
    stop = asyncio.Event()

    async def ticker():
        while not stop.is_set():
            t0 = time.perf_counter()
            await asyncio.sleep(0.005)
            lags.append(time.perf_counter() - t0 - 0.005)

    tick = asyncio.create_task(ticker())
    start = time.perf_counter()
    await asyncio.gather(*(client.embed([f"load {i}"] * 4) for i in range(256)))
    record_property("burst_wall_ms", round((time.perf_counter() - start) * 1000, 2))
    stop.set()
    await tick
    record_property("event_loop_lag_max_ms", round(max(lags) * 1000, 2))
    assert max(lags) < 0.25, f"event loop blocked for {max(lags) * 1000:.0f} ms"


@pytest.mark.slow
async def test_onnx_models_match_sentence_transformers():
    """Needs `python -m model_server.export_onnx` first; skipped otherwise."""
    if not (REPO / "minilm_embedder" / "1" / "model.onnx").exists():
        pytest.skip("model.onnx not exported")
    from sentence_transformers import SentenceTransformer

    server = LocalTritonServer(use_onnx=True)
    port = await server.start()
    client = TritonInferenceClient(f"127.0.0.1:{port}")
    try:
        texts = ["EIO-5 controller timeout", "machines get hot and become sluggish"]
        got = await client.embed(texts)
        ref = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
            texts, normalize_embeddings=True
        )
        np.testing.assert_allclose(got, ref, atol=1e-4)
        scores = await client.rerank(
            "disk controller timeout",
            [
                "EIO-5 means the controller timed out waiting for a block write.",
                "Servers throttle CPU frequency when inlet temperature exceeds 35C.",
            ],
        )
        assert scores[0] > scores[1]
    finally:
        await client.close()
        await server.stop()
