from itertools import pairwise

import pytest

from backend.core.models import Document
from backend.ingestion.chunker import chunk_document


def test_section_headings_follow_markdown_hierarchy(sample_docs):
    storage = next(d for d in sample_docs if d.doc_id == "storage_runbook")
    chunks = chunk_document(storage, chunk_size=400, chunk_overlap=60)
    headings = {c.section_heading for c in chunks}
    assert "Storage Node Runbook > Disk Errors > I/O Timeouts (EIO-5)" in headings
    assert "Storage Node Runbook > Replication Lag" in headings
    assert all(c.document_title == "Storage Node Runbook" for c in chunks)


def test_heading_inside_code_fence_is_not_a_section(sample_docs):
    storage = next(d for d in sample_docs if d.doc_id == "storage_runbook")
    chunks = chunk_document(storage)
    assert not any("This is a comment" in c.section_heading for c in chunks)
    assert any("# This is a comment" in c.text for c in chunks)


def test_chunks_respect_size_and_overlap():
    words = " ".join(f"word{i}" for i in range(600))
    doc = Document(doc_id="long", title="Long", text=f"# Long\n\n{words}")
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_paragraph_overlap_is_carried_between_chunks():
    paras = [f"Paragraph {i} " + "alpha beta gamma delta " * 8 for i in range(6)]
    doc = Document(doc_id="p", title="P", text="\n\n".join(paras))
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=80)
    assert len(chunks) >= 2
    for prev, nxt in pairwise(chunks):
        tail_words = prev.text.split()[-3:]
        assert " ".join(tail_words) in nxt.text


def test_chunk_ids_are_deterministic_uuids(sample_docs):
    a = chunk_document(sample_docs[0])
    b = chunk_document(sample_docs[0])
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_document(Document(doc_id="x", title="x", text="x"), chunk_size=100, chunk_overlap=100)
