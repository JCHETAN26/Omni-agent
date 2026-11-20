"""Shared data models for ingestion and retrieval."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str  # UUID string: valid as both a Qdrant point id and an ES _id
    doc_id: str
    document_title: str
    section_heading: str
    position: int  # 0-based order of the chunk inside its document
    text: str

    def payload(self) -> dict:
        return self.model_dump()


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None  # 1-based rank in the dense list, None if absent
    sparse_rank: int | None = None  # 1-based rank in the BM25 list, None if absent
    dense_score: float | None = None  # cosine similarity from the dense store
    sparse_score: float | None = None  # raw BM25 score from the sparse index
