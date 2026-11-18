"""Real all-MiniLM-L6-v2 embeddings. Run with: pytest -m slow"""

import pytest
from qdrant_client import AsyncQdrantClient

from backend.core.stores import InMemoryBM25Index, QdrantDenseStore
from backend.ingestion.embedder import SentenceTransformerEmbedder
from backend.ingestion.processor import IngestionProcessor
from search.hybrid_retriever import HybridRetriever

pytestmark = pytest.mark.slow


async def test_minilm_semantic_match_without_keyword_overlap(sample_docs):
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    dense = QdrantDenseStore(AsyncQdrantClient(location=":memory:"), "real")
    sparse = InMemoryBM25Index()
    report = await IngestionProcessor(embedder, dense, sparse, 400, 60).ingest(sample_docs)
    assert report.embedding_dim == 384

    retriever = HybridRetriever(embedder, dense, sparse, rrf_k=60, candidates=20)
    # The query shares no content words with the "Thermal Throttling" section.
    result = await retriever.search("machines get hot and become sluggish", top_k=3)
    assert result.hits[0].chunk.section_heading.endswith("Thermal Throttling")
    await dense.close()
