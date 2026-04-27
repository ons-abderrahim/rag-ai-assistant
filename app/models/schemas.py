"""
Pydantic schemas for all API request and response models.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Document Ingestion ────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str = "indexed"
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    document_id: str
    status: str = "deleted"


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to answer using the indexed documents.",
        examples=["What is unsupervised domain adaptation?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve.",
    )
    score_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for a chunk to be included.",
    )
    stream: bool = Field(
        default=False,
        description="Enable server-sent-events streaming (reserved for future use).",
    )


class SourceChunk(BaseModel):
    chunk_id: str
    document: str
    document_id: str
    page: int | None = None
    score: float
    excerpt: str


class QueryMetadata(BaseModel):
    model: str
    latency_ms: int
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    retrieval_strategy: str
    chunks_retrieved: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    metadata: QueryMetadata


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store: str
    documents_indexed: int
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Errors ───────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: Any = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
