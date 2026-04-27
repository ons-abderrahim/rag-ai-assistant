"""
Unit tests for the RAG pipeline components:
  - EmbeddingModel (mocked API)
  - DocumentProcessor (chunking logic)
  - HybridRetriever (retrieval + RRF)
  - ResponseGenerator (prompt building)
  - RAGPipeline (end-to-end)
"""

import pytest
import numpy as np

from app.services.document_processor import DocumentProcessor
from app.utils.helpers import (
    clean_text,
    count_tokens,
    generate_chunk_id,
    generate_document_id,
    truncate_to_tokens,
)


# ── DocumentProcessor ────────────────────────────────────────────────────────

class TestDocumentProcessor:

    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    def test_process_txt_basic(self):
        text = "Machine learning is a subset of artificial intelligence. " * 20
        doc_id, chunks = self.processor.process_file(text.encode(), "sample.txt")

        assert doc_id.startswith("doc_")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.token_count > 0
            assert chunk.chunk_id.startswith(doc_id)

    def test_process_markdown(self):
        md = "# Introduction\n\nThis is a test document.\n\n## Section 2\n\nMore content here. " * 10
        doc_id, chunks = self.processor.process_file(md.encode(), "notes.md")
        assert len(chunks) >= 1

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            self.processor.process_file(b"data", "file.csv")

    def test_chunk_overlap_produces_more_chunks(self):
        """Higher overlap → more chunks for the same text."""
        text = ("This is a sentence about domain adaptation and smart buildings. " * 30).encode()
        proc_no_overlap = DocumentProcessor(chunk_size=50, chunk_overlap=0)
        proc_overlap = DocumentProcessor(chunk_size=50, chunk_overlap=25)
        _, chunks_no_overlap = proc_no_overlap.process_file(text, "doc.txt")
        _, chunks_overlap = proc_overlap.process_file(text, "doc.txt")
        assert len(chunks_overlap) >= len(chunks_no_overlap)

    def test_chunk_ids_are_unique(self):
        text = ("Word " * 500).encode()
        _, chunks = self.processor.process_file(text, "unique.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_file_produces_no_chunks(self):
        _, chunks = self.processor.process_file(b"   \n  ", "blank.txt")
        assert chunks == []


# ── Utility helpers ───────────────────────────────────────────────────────────

class TestHelpers:

    def test_clean_text_strips_whitespace(self):
        dirty = "  hello    world  \n\n\n   foo  "
        cleaned = clean_text(dirty)
        assert "  " not in cleaned
        assert cleaned.startswith("hello")

    def test_count_tokens_non_zero(self):
        assert count_tokens("Hello, world!") > 0

    def test_truncate_to_tokens_respects_limit(self):
        long_text = "word " * 1000
        truncated = truncate_to_tokens(long_text, max_tokens=50)
        assert count_tokens(truncated) <= 50

    def test_truncate_short_text_unchanged(self):
        short = "Short sentence."
        result = truncate_to_tokens(short, max_tokens=200)
        assert result == short

    def test_generate_document_id_deterministic(self):
        id1 = generate_document_id("report.pdf")
        id2 = generate_document_id("report.pdf")
        assert id1 == id2
        assert id1.startswith("doc_")

    def test_generate_document_id_differs_by_name(self):
        assert generate_document_id("a.pdf") != generate_document_id("b.pdf")

    def test_generate_chunk_id_format(self):
        cid = generate_chunk_id("doc_abc123", 7)
        assert cid == "doc_abc123_chunk_0007"


# ── Retriever ─────────────────────────────────────────────────────────────────

class TestHybridRetriever:

    @pytest.mark.asyncio
    async def test_retrieve_returns_source_chunks(
        self, mock_embedder, mock_store
    ):
        from app.core.retriever import HybridRetriever

        retriever = HybridRetriever(
            embedding_model=mock_embedder,
            vector_store=mock_store,
            score_threshold=0.0,
        )
        result = await retriever.retrieve("What is domain adaptation?", top_k=5)

        assert len(result.chunks) >= 1
        assert result.strategy in ("hybrid_rrf", "dense_only")
        chunk = result.chunks[0]
        assert chunk.score >= 0.0
        assert chunk.document
        assert chunk.excerpt

    @pytest.mark.asyncio
    async def test_score_threshold_filters(self, mock_embedder, mock_store):
        from app.core.retriever import HybridRetriever
        from app.services.vector_store import SearchResult

        # Return a low-score result
        mock_store.search.return_value = [
            SearchResult(
                chunk_id="low_score_chunk",
                document_id="doc_x",
                filename="x.txt",
                text="irrelevant text",
                page=None,
                score=0.1,
            )
        ]
        retriever = HybridRetriever(
            embedding_model=mock_embedder,
            vector_store=mock_store,
            score_threshold=0.8,  # high threshold
        )
        result = await retriever.retrieve("something", top_k=5)
        assert result.chunks == []


# ── RAG Pipeline (end-to-end) ─────────────────────────────────────────────────

class TestRAGPipeline:

    @pytest.mark.asyncio
    async def test_pipeline_run_returns_response(
        self, mock_embedder, mock_store, mock_openai_completion
    ):
        from app.core.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            embedding_model=mock_embedder, vector_store=mock_store
        )
        response = await pipeline.run("Explain domain adaptation.", top_k=3)

        assert response.question == "Explain domain adaptation."
        assert response.answer
        assert isinstance(response.sources, list)
        assert response.metadata.latency_ms >= 0
        assert response.metadata.model

    @pytest.mark.asyncio
    async def test_pipeline_metadata_populated(
        self, mock_embedder, mock_store, mock_openai_completion
    ):
        from app.core.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            embedding_model=mock_embedder, vector_store=mock_store
        )
        response = await pipeline.run("Test question for metadata.")
        meta = response.metadata
        assert meta.tokens_used > 0
        assert meta.retrieval_strategy in ("hybrid_rrf", "dense_only")
        assert meta.chunks_retrieved >= 0
