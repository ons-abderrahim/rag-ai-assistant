"""
Query route — POST /query

Handles the main RAG query endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.rag_pipeline import RAGPipeline
from app.models.schemas import QueryRequest, QueryResponse
from app.utils.logger import get_logger

router = APIRouter(tags=["Query"])
logger = get_logger(__name__)


def get_pipeline() -> RAGPipeline:
    """Dependency injector — resolved via app state set in main.py."""
    from app.main import app
    pipeline: RAGPipeline | None = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialised.",
        )
    return pipeline


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description=(
        "Submit a natural language question. The system retrieves relevant "
        "document chunks and generates a grounded answer with source citations."
    ),
)
async def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    Run the RAG pipeline for the given question.

    - **question**: natural language question (3–2000 chars)
    - **top_k**: number of chunks to retrieve (1–20, default 5)
    - **score_threshold**: minimum retrieval score (0–1, default 0.4)
    """
    try:
        return await pipeline.run(
            question=body.question,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
        )
    except Exception as exc:
        logger.exception(f"Query failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(exc)}",
        )
