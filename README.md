# OmniAgent

Low-latency, multi-modal agentic copilot for enterprise technical support and diagnostics.

- **Phase 1:** monorepo scaffold, GitHub Codespaces dev environment, async document
  ingestion, and parallel hybrid retrieval (dense + BM25, fused with RRF).
- **Phase 2:** WebSocket and SSE streaming gateway (sessions, heartbeats, ring buffer,
  token-bucket rate limiting), Triton model repository and async gRPC inference client,
  cross-encoder reranking, structured logging, and OpenTelemetry tracing across frames
  and gRPC calls.

## Layout

```
backend/
  api/           websocket.py (/ws/stream), streaming.py (event stream + /sse/stream),
                 ring_buffer.py
  core/          config, models, stores (Qdrant/ES), cache, rate_limit, telemetry, services
  ingestion/     chunker.py, embedder.py, processor.py (async ingestion pipeline)
  main.py        FastAPI app factory: REST + SSE + WebSocket
search/
  hybrid_retriever.py   parallel dense + sparse retrieval, Reciprocal Rank Fusion
model_server/
  model_repository/     Triton configs: minilm_embedder, cross_encoder_reranker
  inference_client.py   async Triton gRPC client (+ in-process mock)
  local_triton.py       local server speaking Triton's gRPC protocol (dev/test/bench)
  export_onnx.py        exports both models to ONNX and verifies them against PyTorch
samples/         small technical-support corpus used by tests and demos
scripts/         demo_phase1.py, bench_phase2.py
tests/           pytest suite
.devcontainer/   Codespaces config (Python 3.12, Node 22, CPU-only PyTorch)
```

## Quick start

**Codespaces:** open the repo in a Codespace. `postCreateCommand` installs everything and
creates `.env` from `.env.example`.

**Locally:**

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
cp .env.example .env
```

Then:

```bash
pytest -m "not slow"              # fast suite
pytest                            # + real MiniLM and ONNX parity tests
python -m scripts.demo_phase1     # ingest samples + run hybrid searches
uvicorn backend.main:app --reload # API on :8000, docs at /docs
```

To serve the real models over gRPC:

```bash
pip install -r model_server/requirements-export.txt
python -m model_server.export_onnx          # writes model_repository/*/1/model.onnx (~87 MB each)
python -m model_server.local_triton --port 8001
INFERENCE_BACKEND=triton EMBEDDING_PROVIDER=triton RERANK_ENABLED=true \
  uvicorn backend.main:app
```

## Services, with no Docker required

Each backend switches on when its URL is set. Otherwise a fallback runs in-process:

| Service       | Env var                                    | Fallback when unset                     |
|---------------|--------------------------------------------|-----------------------------------------|
| Qdrant        | `QDRANT_URL`                               | embedded in-memory Qdrant (`:memory:`)  |
| Elasticsearch | `ELASTICSEARCH_URL`                        | pure-Python Okapi BM25 (k1=1.2, b=0.75) |
| Redis         | `REDIS_URL`                                | in-process cache and rate limiter       |
| Triton        | `INFERENCE_BACKEND=triton`, `TRITON_GRPC_URL` | in-process mock inference client     |
| OTLP collector| `OTEL_EXPORTER_OTLP_ENDPOINT`              | spans created and propagated, not exported |

In Codespaces, set these as Codespaces secrets. The devcontainer lists them as recommended
secrets. Keep the `elasticsearch` client major version equal to your cluster's major version.

## How it works

**Ingestion** (`backend/ingestion/processor.py`)
1. Chunking splits on Markdown headings and records the heading path as `section_heading`.
   It packs paragraphs up to `CHUNK_SIZE` characters, with `CHUNK_OVERLAP` overlap.
   Chunk IDs are deterministic UUIDv5s, so re-ingesting a document overwrites its chunks.
2. Two branches run concurrently with `asyncio.gather`:
   - embed (MiniLM, OpenAI, Triton, or the offline hashing embedder), then upsert into
     Qdrant with payload `chunk_id`, `document_title`, `section_heading`, `text`, ...
   - index raw text into Elasticsearch (BM25 similarity) or the in-process BM25 index.

**Retrieval** (`search/hybrid_retriever.py`)
- Dense ANN and BM25 queries run concurrently. Each returns `RETRIEVAL_CANDIDATES` results.
- The results are fused with RRF: `score(d) = Σ 1 / (k + rank_r(d))`, `k = 60`.
- Each hit reports its fused score plus its rank in each list, so you can debug which
  retriever found it.
- With `RERANK_ENABLED=true`, the top `RERANK_CANDIDATES` go to the cross-encoder. If it
  doesn't answer within `RERANK_TIMEOUT_S`, the response uses RRF order and is flagged
  `reranked: false` rather than failing.

## Streaming gateway

### WebSocket: `/ws/stream`

Client → server (JSON text frames, plus binary frames for audio):

| Frame | Meaning |
|---|---|
| `{"type":"query","id":"q1","text":"...","top_k":5,"traceparent"?:"00-…"}` | ask a question |
| `{"type":"ping","ts":…}` | answered immediately with `pong` |
| `{"type":"pong","ts":…}` | reply to a server heartbeat `ping` |
| binary frame | one audio chunk for the current utterance |
| `{"type":"audio.end","id":"a1"}` | end of utterance → `audio.received` with byte count + SHA-256 |

Server → client: every frame carries a per-session, strictly increasing `seq`.
`session.created`, `pong`, `ping`, `rate_limited`, `buffer.overflow`, `error`,
`audio.received`, and per request:
`response.start` → `response.sources` → `response.token`* → `response.end` (or
`response.error`). Tokens carry a contiguous `index`. `response.end` carries the token
count and a SHA-256 of the concatenated text, so clients can verify the stream.

Each connection runs three tasks in an anyio task group:
- **reader:** validates frames, applies the per-session token bucket (Redis Lua script,
  or in-memory), and pushes into a ring buffer (`WS_BUFFER_SIZE` slots). When the buffer
  is full, the oldest item is dropped and the client gets `buffer.overflow`.
- **worker:** drains the buffer in order and streams responses.
- **heartbeat:** sends `ping` every `WS_HEARTBEAT_INTERVAL_S` and closes with code `4408`
  after `WS_HEARTBEAT_TIMEOUT_S` of silence.

Other close codes: `1009` message too big, `1013` server at `WS_MAX_SESSIONS`.

### SSE: `GET /sse/stream?q=...&top_k=5`

The same event sequence as `text/event-stream`. It honours a `traceparent` request header.

> **The answer text is a placeholder.** It streams the best source passage word by word
> until LLM generation lands. The event contract stays the same when a generator
> replaces it. Audio is acknowledged and verified, but not transcribed yet.

## Triton inference

`model_server/model_repository/` holds the Triton model repository:

| Model | Source | Inputs → output | Batching |
|---|---|---|---|
| `minilm_embedder` | all-MiniLM-L6-v2 + mean pooling + L2 norm | `input_ids`, `attention_mask` → `sentence_embedding [384]` | max 64, 5 ms queue delay, 2 CPU instances |
| `cross_encoder_reranker` | ms-marco-MiniLM-L-6-v2 | `+ token_type_ids` → `logits [1]` | max 64, 5 ms queue delay |

The `.onnx` files are generated by `export_onnx.py` and are not committed. The script
checks every graph's I/O against its `config.pbtxt`, and checks numerical parity with
PyTorch (max |Δ| ≈ 1e-7 for the embedder, ≈ 5e-6 for the reranker).

**Client** (`inference_client.py`) uses `tritonclient.grpc.aio`:
- Tokenization, including importing `transformers`, runs in worker threads, and compute
  happens in Triton, so the API event loop never runs model code.
- Sequences are padded to 32/64/128/256-token buckets. Triton only batches requests with
  identical shapes, so bucketing is what lets concurrent requests merge.
- A semaphore caps in-flight requests (`TRITON_MAX_IN_FLIGHT`), and inputs larger than
  `max_batch_size` are split.
- Each call gets a `triton.infer` client span and a W3C `traceparent` in its gRPC metadata.

**Local server** (`local_triton.py`) is *not* Triton. It speaks Triton's gRPC protocol
using Triton's own protobufs: ServerLive/Ready, ModelReady, ModelConfig, ModelInfer and
ModelStatistics. It loads the same `config.pbtxt` files and implements dynamic batching
(queue delay, shape-matched merging, `max_batch_size`, instance concurrency). It runs
ONNX models with onnxruntime when they're exported, otherwise synthetic compute. Tests
and benchmarks use it on machines that can't run Triton.

**Real Triton** runs on Linux, typically with a GPU:
```bash
docker run --gpus all -p 8001:8001 -v $PWD/model_server/model_repository:/models \
  nvcr.io/nvidia/tritonserver:<version>-py3 tritonserver --model-repository=/models
python -m scripts.bench_phase2 --triton-url <host>:8001
```
For GPU serving, change `instance_group.kind` to `KIND_GPU`.

## Observability

- **Logs:** with `LOG_FORMAT=json`, logs are one JSON object per line, including
  `trace_id`/`span_id` from the active span.
- **Traces:** OpenTelemetry spans: `ws.session`; `omniagent.answer` per request (a child
  of the client's `traceparent` if sent, otherwise a new root linked to the session);
  `triton.infer` on the client, and `triton.server.ModelInfer` on the local server,
  parented across gRPC. HTTP spans come from FastAPI instrumentation. Export to OTLP with
  `OTEL_EXPORTER_OTLP_ENDPOINT`, or to the console with `OTEL_CONSOLE_EXPORT=true`.
- **`/health`:** backends, inference readiness, WebSocket session counts, and event-loop
  lag percentiles. The lag figure is a direct check that nothing blocks the loop.

## Benchmark

`python -m scripts.bench_phase2` runs the inference server and the gateway as separate
processes. Results on an 8-core Apple Silicon laptop, CPU only, real ONNX models:

| gRPC embed, 1 sentence/request | conc 1 | conc 8 | conc 64 | conc 128 |
|---|---|---|---|---|
| requests/s | 87 | 188 | 323 | 316 |
| p50 latency (ms) | 11.4 | 42.0 | 194.8 | 431.2 |
| avg dynamic batch size | 1.0 | 8.0 | 20.5 | 29.3 |

| WebSocket: 100 concurrent sessions × 5 queries | embed only | embed + rerank |
|---|---|---|
| completed / errors / integrity failures | 500 / 0 / 0 | 500 / 0 / 0 |
| queries/s | 267 | 53 |
| time to first token p50 / p95 (ms) | 300 / 389 | 1613 / 1996 |
| gateway event-loop lag p99 / max (ms) | 13 / 71 | 9 / 146 |

On this CPU, the cross-encoder manages about 75 pairs/s, so under 100 concurrent sessions
491 of 500 queries hit the 1.5 s rerank budget and fell back to RRF order. That's the
designed degradation. Reranking at that load needs a GPU Triton.

## API

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"doc_id":"kb1","title":"KB","text":"# KB\n## Disk\nEIO-5 means ..."}]}'
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"EIO-5 controller timeout","top_k":5}'
curl -N 'localhost:8000/sse/stream?q=EIO-5+controller+timeout'
```

## Known limitations

- The search cache key is invalidated by an in-process counter on each ingest. With Redis
  and several API workers, other workers can serve stale cached results until the TTL
  expires.
- Ingestion is a synchronous request, with no background job queue yet.
- Answers are extractive placeholders until LLM generation lands, and audio isn't
  transcribed yet.
- There's no authentication on `/ws/stream` or the REST endpoints yet.
- The in-process Qdrant and BM25 fallbacks run their search math on the event loop. That's
  fine for development, but production should point at real Qdrant and Elasticsearch.
- The gRPC path has been tested against the local Triton-protocol server, not a real
  Triton instance. Run `bench_phase2.py --triton-url` once one is available.
