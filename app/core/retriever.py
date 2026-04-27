"""
Hybrid retriever: dense (FAISS/Pinecone) + sparse (BM25).
Results fused with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.embeddings import EmbeddingModel
from app.models.schemas import SourceChunk
from app.services.vector_store import BaseVectorStore, SearchResult
from app.utils.logger import LoggerMixin, log_event


@dataclass
class RetrievalResult:
    chunks: list[SourceChunk]
    strategy: str


class HybridRetriever(LoggerMixin):
    """
    Two-stage retrieval:

    1. Dense retrieval   — embedding similarity via the vector store
    2. Sparse retrieval  — BM25 over all indexed chunk texts (in-memory)
    3. RRF fusion        — merge and rerank the two ranked lists

    The BM25 index is rebuilt on each call from the vector store metadata.
    For production with large corpora, consider a persistent sparse index
    (e.g. Elasticsearch, Tantivy).
    """

    RRF_K = 60  # RRF constant — higher = smoother rank blending

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: BaseVectorStore,
        score_threshold: float = 0.0,
    ) -> None:
        self._embedder = embedding_model
        self._store = vector_store
        self._threshold = score_threshold

    # ── Public API ────────────────────────────────────────────────────

    async def retrieve(
        self, query: str, top_k: int, score_threshold: float | None = None
    ) -> RetrievalResult:
        """
        Retrieve the top-k most relevant chunks for *query*.

        Returns
        -------
        RetrievalResult with a list of SourceChunk objects and strategy label.
        """
        threshold = score_threshold if score_threshold is not None else self._threshold

        log_event(self.logger, "retrieval_start", query_len=len(query), top_k=top_k)

        # ── Step 1: Dense retrieval ─────────────────────────────────
        query_vec = await self._embedder.embed_query(query)
        dense_results = await self._store.search(query_vec, top_k=top_k * 2)

        # ── Step 2: Sparse / BM25 retrieval ────────────────────────
        sparse_results = self._bm25_search(query, dense_results, top_k=top_k * 2)

        # ── Step 3: Fuse with RRF ───────────────────────────────────
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # ── Step 4: Apply score threshold and top_k ─────────────────
        filtered = [r for r in fused if r.score >= threshold][:top_k]

        strategy = "hybrid_rrf" if sparse_results else "dense_only"
        log_event(
            self.logger,
            "retrieval_complete",
            strategy=strategy,
            returned=len(filtered),
        )

        chunks = [self._to_source_chunk(r) for r in filtered]
        return RetrievalResult(chunks=chunks, strategy=strategy)

    # ── BM25 ──────────────────────────────────────────────────────────

    def _bm25_search(
        self,
        query: str,
        dense_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Run BM25 over the *dense_results* corpus (not the full index).
        This scoped BM25 avoids the overhead of indexing all chunks.
        """
        if not dense_results:
            return []

        corpus = [r.text.lower().split() for r in dense_results]
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        scored = sorted(
            zip(scores, dense_results), key=lambda x: x[0], reverse=True
        )
        return [r for _, r in scored[:top_k]]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
    ) -> list[SearchResult]:
        """Combine two ranked lists using RRF scoring."""
        rrf_scores: dict[str, float] = {}
        index: dict[str, SearchResult] = {}

        for rank, result in enumerate(dense, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + (
                1.0 / (self.RRF_K + rank)
            )
            index[result.chunk_id] = result

        for rank, result in enumerate(sparse, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + (
                1.0 / (self.RRF_K + rank)
            )
            index[result.chunk_id] = result

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        fused: list[SearchResult] = []
        for cid in sorted_ids:
            r = index[cid]
            # Replace raw cosine score with normalised RRF score [0, 1]
            r.score = min(1.0, rrf_scores[cid] * self.RRF_K)
            fused.append(r)

        return fused

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_source_chunk(result: SearchResult) -> SourceChunk:
        excerpt = result.text[:300] + "…" if len(result.text) > 300 else result.text
        return SourceChunk(
            chunk_id=result.chunk_id,
            document=result.filename,
            document_id=result.document_id,
            page=result.page,
            score=round(result.score, 4),
            excerpt=excerpt,
        )
