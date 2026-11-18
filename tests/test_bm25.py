from backend.core.models import Chunk
from backend.core.stores import InMemoryBM25Index


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", document_title="t", section_heading="s", position=0, text=text
    )


async def test_bm25_ranks_rare_term_match_first():
    idx = InMemoryBM25Index()
    await idx.index(
        [
            _chunk("1", "the disk controller timed out"),
            _chunk("2", "the network is slow the network is congested"),
            _chunk("3", "the replica is stale"),
        ]
    )
    hits = await idx.search("controller timeout disk", limit=3)
    assert hits[0][0].chunk_id == "1"
    assert all(score > 0 for _, score in hits)


async def test_bm25_reindex_overwrites_chunk():
    idx = InMemoryBM25Index()
    await idx.index([_chunk("1", "alpha"), _chunk("2", "beta")])
    await idx.index([_chunk("1", "gamma")])
    assert await idx.search("alpha", 5) == []
    assert (await idx.search("gamma", 5))[0][0].chunk_id == "1"
