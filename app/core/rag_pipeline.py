"""
RAG Pipeline — top-level orchestrator.

Coordinates: embed → retrieve → generate → return response.
Optionally traces the full run to LangSmith.
"""

from __future__ import annotations

import time

from app.config import get_settings
from app.core.embeddings import EmbeddingModel
from app.core.generator import ResponseGenerator
from app.core.retriever import HybridRetriever
from app.models.schemas import QueryMetadata, QueryResponse, SourceChunk
from app.services.vector_store import BaseVectorStore
from app.utils.helpers import generate_run_id
from app.utils.logger import LoggerMixin, log_event

settings = get_settings()


class RAGPipeline(LoggerMixin):
    """
    End-to-end RAG pipeline.

    Usage
    -----
    pipeline = RAGPipeline(embedding_model, vector_store)
    response = await pipeline.run(question="...", top_k=5)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedder = embedding_model
        self._store = vector_store
        self._retriever = HybridRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            score_threshold=settings.score_threshold,
        )
        self._generator = ResponseGenerator()
        self._tracer = self._init_tracer()

    # ── Public API ────────────────────────────────────────────────────

    async def run(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> QueryResponse:
        """
        Execute the full RAG pipeline for a user question.

        Parameters
        ----------
        question : str
            Natural language question.
        top_k : int, optional
            Number of chunks to retrieve (overrides config default).
        score_threshold : float, optional
            Minimum retrieval score (overrides config default).

        Returns
        -------
        QueryResponse — fully populated API response object.
        """
        run_id = generate_run_id()
        top_k = top_k or settings.top_k

        log_event(
            self.logger,
            "pipeline_start",
            run_id=run_id,
            question_len=len(question),
            top_k=top_k,
        )

        start_time = time.perf_counter()

        # ── 1. Retrieve ─────────────────────────────────────────────
        retrieval = await self._retriever.retrieve(
            query=question,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # ── 2. Generate ─────────────────────────────────────────────
        generation = await self._generator.generate(
            question=question,
            chunks=retrieval.chunks,
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        log_event(
            self.logger,
            "pipeline_complete",
            run_id=run_id,
            latency_ms=elapsed_ms,
            chunks_used=len(retrieval.chunks),
            tokens=generation.total_tokens,
        )

        # ── 3. Trace (LangSmith) ────────────────────────────────────
        if self._tracer:
            self._trace(run_id, question, retrieval.chunks, generation.answer)

        return QueryResponse(
            question=question,
            answer=generation.answer,
            sources=retrieval.chunks,
            metadata=QueryMetadata(
                model=generation.model,
                latency_ms=elapsed_ms,
                tokens_used=generation.total_tokens,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                retrieval_strategy=retrieval.strategy,
                chunks_retrieved=len(retrieval.chunks),
            ),
        )

    # ── LangSmith Tracing ─────────────────────────────────────────────

    def _init_tracer(self):
        """Initialise LangSmith client if credentials are set."""
        if not settings.langsmith_api_key or not settings.langchain_tracing_v2:
            return None
        try:
            from langsmith import Client  # type: ignore[import-untyped]
            client = Client(api_key=settings.langsmith_api_key)
            log_event(self.logger, "langsmith_enabled", project=settings.langsmith_project)
            return client
        except Exception as exc:
            self.logger.warning(f"LangSmith init failed — tracing disabled: {exc}")
            return None

    def _trace(
        self,
        run_id: str,
        question: str,
        chunks: list[SourceChunk],
        answer: str,
    ) -> None:
        """Fire-and-forget LangSmith trace (errors are logged, not raised)."""
        try:
            self._tracer.create_run(
                name="rag_pipeline",
                run_type="chain",
                id=run_id,
                inputs={"question": question},
                outputs={
                    "answer": answer,
                    "source_count": len(chunks),
                },
                project_name=settings.langsmith_project,
            )
        except Exception as exc:
            self.logger.warning(f"LangSmith trace failed: {exc}")
