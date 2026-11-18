from pathlib import Path

import pytest

from backend.core.models import Document
from backend.core.stores import InMemoryBM25Index, QdrantDenseStore
from backend.ingestion.embedder import HashingEmbedder
from backend.ingestion.processor import IngestionProcessor
from search.hybrid_retriever import HybridRetriever

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def load_samples() -> list[Document]:
    docs = []
    for path in sorted(SAMPLES.glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        docs.append(Document(doc_id=path.stem, title=title, text=text))
    return docs


@pytest.fixture
def sample_docs() -> list[Document]:
    return load_samples()


@pytest.fixture
async def stack():
    """Offline stack: hashing embedder + in-memory Qdrant + in-process BM25."""
    from qdrant_client import AsyncQdrantClient

    embedder = HashingEmbedder()
    dense = QdrantDenseStore(AsyncQdrantClient(location=":memory:"), "test_chunks")
    sparse = InMemoryBM25Index()
    processor = IngestionProcessor(embedder, dense, sparse, chunk_size=400, chunk_overlap=60)
    retriever = HybridRetriever(embedder, dense, sparse, rrf_k=60, candidates=20)
    yield processor, retriever
    await dense.close()
