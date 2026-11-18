# OmniAgent

Low-latency, multi-modal agentic copilot for enterprise technical support and diagnostics.

**Phase 1 (this commit):** monorepo scaffold, GitHub Codespaces dev environment, async
document ingestion, and parallel hybrid retrieval (dense + BM25, fused with RRF).

## Layout

```
backend/
  core/          config (env-driven), data models, Qdrant/Elasticsearch adapters, cache
  ingestion/     chunker.py, embedder.py, processor.py (async ingestion pipeline)
  main.py        FastAPI gateway: /health, /ingest, /search
search/
  hybrid_retriever.py   parallel dense + sparse retrieval, Reciprocal Rank Fusion
samples/         small technical-support corpus used by tests and the demo
scripts/         demo_phase1.py (end-to-end smoke demo)
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
pytest -m "not slow"              # offline suite (no model download)
pytest -m slow                    # real all-MiniLM-L6-v2 embeddings
python -m scripts.demo_phase1     # ingest samples + run hybrid searches
uvicorn backend.main:app --reload # API on :8000, docs at /docs
```

## Services, with no Docker required

Each backend switches on when its URL is set. Otherwise a fallback runs in-process:

| Service       | Env var             | Fallback when unset                         |
|---------------|---------------------|---------------------------------------------|
| Qdrant        | `QDRANT_URL`        | embedded in-memory Qdrant (`:memory:`)      |
| Elasticsearch | `ELASTICSEARCH_URL` | pure-Python Okapi BM25 (k1=1.2, b=0.75)     |
| Redis         | `REDIS_URL`         | in-process TTL cache                        |

In Codespaces, set these as Codespaces secrets. The devcontainer lists them as recommended
secrets. Keep the `elasticsearch` client major version equal to your cluster's major version.

## How it works

**Ingestion** (`backend/ingestion/processor.py`)
1. Chunking splits on Markdown headings and records the heading path as `section_heading`.
   It packs paragraphs up to `CHUNK_SIZE` characters, with `CHUNK_OVERLAP` overlap.
   Chunk IDs are deterministic UUIDv5s, so re-ingesting a document overwrites its chunks.
2. Two branches run concurrently with `asyncio.gather`:
   - embed (MiniLM, OpenAI, or the offline hashing embedder), then upsert into Qdrant with
     payload `chunk_id`, `document_title`, `section_heading`, `text`, ...
   - index raw text into Elasticsearch (BM25 similarity) or the in-process BM25 index.

**Retrieval** (`search/hybrid_retriever.py`)
- Dense ANN and BM25 queries run concurrently. Each returns `RETRIEVAL_CANDIDATES` results.
- The results are fused with RRF: `score(d) = Σ 1 / (k + rank_r(d))`, `k = 60`.
- Each hit reports its fused score plus its rank in each list, so you can debug which
  retriever found it.

## API

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"doc_id":"kb1","title":"KB","text":"# KB\n## Disk\nEIO-5 means ..."}]}'
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"EIO-5 controller timeout","top_k":5}'
```

## Known Phase 1 limitations

- The search cache key is invalidated by an in-process counter on each ingest. With Redis
  and several API workers, other workers can serve stale cached results until the TTL
  expires.
- Ingestion is a synchronous request, with no background job queue yet.
- There's no cross-encoder reranking or LLM generation yet. Those come in later phases.
