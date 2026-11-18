"""Phase 1 smoke demo: ingest the sample corpus, then run hybrid searches.

Usage:
    python -m scripts.demo_phase1                 # uses .env (default: MiniLM embeddings)
    EMBEDDING_PROVIDER=hashing python -m scripts.demo_phase1   # fully offline
"""

import asyncio
import logging

from backend.core.config import get_settings
from backend.core.stores import QdrantDenseStore, sparse_index_from_settings
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import embedder_from_settings
from backend.ingestion.processor import IngestionProcessor
from search.hybrid_retriever import HybridRetriever
from tests.conftest import load_samples

QUERIES = [
    "EIO-5 controller timeout",  # exact error code: BM25 should lead
    "machines get hot and become sluggish",  # paraphrase: dense should lead
    "pods take forever to resolve hostnames",  # paraphrase of the ndots section
    "PSU redundancy lost",  # exact alert text
]


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    s = get_settings()
    embedder = embedder_from_settings(s)
    dense = QdrantDenseStore.from_settings(s)
    sparse = sparse_index_from_settings(s)
    docs = load_samples()

    print("=" * 78)
    print(f"Embedder : {embedder.name}")
    dense_desc = f"Qdrant @ {s.qdrant_url}" if s.qdrant_url else "Qdrant (in-memory)"
    sparse_desc = (
        f"Elasticsearch @ {s.elasticsearch_url}" if s.elasticsearch_url else "BM25 (in-process)"
    )
    print(f"Dense    : {dense_desc}")
    print(f"Sparse   : {sparse_desc}")
    print(f"RRF k    : {s.rrf_k}")
    print("=" * 78)

    print("\n[1] Chunking")
    for d in docs:
        chunks = chunk_document(d, s.chunk_size, s.chunk_overlap)
        print(f"  {d.doc_id:<26} {len(chunks):>2} chunks")
        for c in chunks:
            print(
                f"      #{c.position} {c.chunk_id[:8]}  {len(c.text):>4} chars"
                f"  [{c.section_heading}]"
            )

    print("\n[2] Ingestion (embed -> Qdrant  ||  BM25 index)")
    report = await IngestionProcessor(
        embedder, dense, sparse, s.chunk_size, s.chunk_overlap
    ).ingest(docs)
    for k, v in report.model_dump().items():
        print(f"  {k:<14} {v}")

    print("\n[3] Hybrid retrieval (parallel dense + sparse, RRF fusion)")
    retriever = HybridRetriever(embedder, dense, sparse, s.rrf_k, s.retrieval_candidates)
    for q in QUERIES:
        res = await retriever.search(q, top_k=3)
        t = res.timings
        print(f"\n  Q: {q!r}")
        print(
            f"     dense {t.dense_ms} ms | sparse {t.sparse_ms} ms"
            f" | fusion {t.fusion_ms} ms | total {t.total_ms} ms"
        )
        for i, h in enumerate(res.hits, 1):
            dr = h.dense_rank or "-"
            sr = h.sparse_rank or "-"
            print(
                f"     {i}. rrf={h.score:.5f}  dense#{dr:<2} bm25#{sr:<2} {h.chunk.section_heading}"
            )

    await dense.close()
    await sparse.close()


if __name__ == "__main__":
    asyncio.run(main())
