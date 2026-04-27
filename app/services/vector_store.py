"""
Vector store abstraction layer.

Provides a unified interface over FAISS (local) and Pinecone (cloud).
Select the backend via the VECTOR_STORE environment variable.
"""

from __future__ import annotations

import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.utils.logger import LoggerMixin, log_event

settings = get_settings()


# ── Data Transfer Objects ─────────────────────────────────────────────────────

@dataclass
class VectorRecord:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page: int | None
    chunk_index: int
    token_count: int


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page: int | None
    score: float


# ── Abstract Base ─────────────────────────────────────────────────────────────

class BaseVectorStore(ABC, LoggerMixin):

    @abstractmethod
    async def upsert(
        self, vectors: list[np.ndarray], records: list[VectorRecord]
    ) -> None: ...

    @abstractmethod
    async def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> int: ...

    @abstractmethod
    def list_documents(self) -> list[dict]: ...

    @abstractmethod
    def document_count(self) -> int: ...


# ── FAISS Implementation ──────────────────────────────────────────────────────

class FAISSVectorStore(BaseVectorStore):
    """
    Local FAISS vector store with disk persistence.

    Index type: IndexFlatIP (inner product = cosine similarity on unit vectors)
    """

    _METADATA_FILE = "metadata.pkl"
    _INDEX_FILE = "index.faiss"

    def __init__(self) -> None:
        import faiss  # local import so Pinecone-only setups don't need it

        self._faiss = faiss
        self._dims = settings.embedding_dimensions
        self._index_path = Path(settings.faiss_index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)

        self._index: faiss.Index = faiss.IndexFlatIP(self._dims)
        self._metadata: list[VectorRecord] = []  # positional: index i → record i
        self._load()

    # ── Write ─────────────────────────────────────────────────────────

    async def upsert(
        self, vectors: list[np.ndarray], records: list[VectorRecord]
    ) -> None:
        if not vectors:
            return

        matrix = np.stack(vectors).astype(np.float32)
        self._index.add(matrix)
        self._metadata.extend(records)
        self._save()
        log_event(
            self.logger, "faiss_upsert", count=len(vectors), total=len(self._metadata)
        )

    async def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to *document_id*.
        FAISS does not support in-place deletion, so we rebuild the index.
        """
        keep_indices = [
            i for i, r in enumerate(self._metadata) if r.document_id != document_id
        ]
        removed = len(self._metadata) - len(keep_indices)

        if removed == 0:
            return 0

        kept_meta = [self._metadata[i] for i in keep_indices]

        # Reconstruct vectors for kept records
        all_vecs = self._index.reconstruct_n(0, self._index.ntotal)
        kept_vecs = np.array([all_vecs[i] for i in keep_indices], dtype=np.float32)

        new_index = self._faiss.IndexFlatIP(self._dims)
        if len(kept_vecs):
            new_index.add(kept_vecs)

        self._index = new_index
        self._metadata = kept_meta
        self._save()

        log_event(self.logger, "faiss_delete", doc_id=document_id, removed=removed)
        return removed

    # ── Read ──────────────────────────────────────────────────────────

    async def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            rec = self._metadata[idx]
            results.append(
                SearchResult(
                    chunk_id=rec.chunk_id,
                    document_id=rec.document_id,
                    filename=rec.filename,
                    text=rec.text,
                    page=rec.page,
                    score=float(score),
                )
            )
        return results

    def list_documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for rec in self._metadata:
            if rec.document_id not in seen:
                seen[rec.document_id] = {
                    "document_id": rec.document_id,
                    "filename": rec.filename,
                    "chunks": 0,
                }
            seen[rec.document_id]["chunks"] += 1
        return list(seen.values())

    def document_count(self) -> int:
        return len({r.document_id for r in self._metadata})

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        self._faiss.write_index(
            self._index, str(self._index_path / self._INDEX_FILE)
        )
        with open(self._index_path / self._METADATA_FILE, "wb") as f:
            pickle.dump(self._metadata, f)

    def _load(self) -> None:
        index_file = self._index_path / self._INDEX_FILE
        meta_file = self._index_path / self._METADATA_FILE
        if index_file.exists() and meta_file.exists():
            self._index = self._faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                self._metadata = pickle.load(f)
            log_event(
                self.logger,
                "faiss_loaded",
                vectors=self._index.ntotal,
                records=len(self._metadata),
            )


# ── Pinecone Implementation ───────────────────────────────────────────────────

class PineconeVectorStore(BaseVectorStore):
    """Pinecone cloud vector store via the official Python SDK."""

    def __init__(self) -> None:
        from pinecone import Pinecone  # type: ignore[import-untyped]

        pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = pc.Index(settings.pinecone_index)
        self._dims = settings.embedding_dimensions

    async def upsert(
        self, vectors: list[np.ndarray], records: list[VectorRecord]
    ) -> None:
        batch = [
            {
                "id": rec.chunk_id,
                "values": vec.tolist(),
                "metadata": {
                    "document_id": rec.document_id,
                    "filename": rec.filename,
                    "text": rec.text,
                    "page": rec.page,
                    "chunk_index": rec.chunk_index,
                    "token_count": rec.token_count,
                },
            }
            for vec, rec in zip(vectors, records)
        ]
        self._index.upsert(vectors=batch)

    async def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[SearchResult]:
        response = self._index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
        )
        results = []
        for match in response.matches:
            meta = match.metadata
            results.append(
                SearchResult(
                    chunk_id=match.id,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    text=meta["text"],
                    page=meta.get("page"),
                    score=float(match.score),
                )
            )
        return results

    async def delete_document(self, document_id: str) -> int:
        # Pinecone supports delete by metadata filter
        self._index.delete(filter={"document_id": {"$eq": document_id}})
        return -1  # count not available via this API

    def list_documents(self) -> list[dict]:
        # Pinecone does not offer a native list-by-metadata; return empty for now
        return []

    def document_count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.total_vector_count


# ── Factory ───────────────────────────────────────────────────────────────────

_store: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """Return a singleton vector store instance (chosen via config)."""
    global _store
    if _store is None:
        if settings.vector_store == "pinecone":
            _store = PineconeVectorStore()
        else:
            _store = FAISSVectorStore()
    return _store
