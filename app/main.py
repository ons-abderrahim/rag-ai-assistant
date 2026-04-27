"""
RAG AI Assistant — FastAPI application entry point.

Startup sequence
----------------
1. Load settings from environment / .env
2. Initialise the embedding model (shared singleton)
3. Initialise the vector store (FAISS or Pinecone)
4. Assemble the RAG pipeline and attach to app.state
5. Mount all API routers
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.config import get_settings
from app.core.embeddings import EmbeddingModel
from app.core.rag_pipeline import RAGPipeline
from app.models.schemas import HealthResponse
from app.services.vector_store import get_vector_store
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise expensive resources once on startup; clean up on shutdown."""
    logger.info(f"Starting RAG AI Assistant v{settings.app_version}")

    # Shared embedding model
    embedder = EmbeddingModel()
    app.state.embedder = embedder
    logger.info(f"Embedding model ready: {settings.embedding_model}")

    # Vector store
    store = get_vector_store()
    app.state.store = store
    logger.info(f"Vector store ready: {settings.vector_store}")

    # RAG pipeline
    pipeline = RAGPipeline(embedding_model=embedder, vector_store=store)
    app.state.pipeline = pipeline
    logger.info("RAG pipeline ready")

    yield  # application runs here

    logger.info("Shutting down RAG AI Assistant")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG AI Assistant",
    description=(
        "Production-ready Retrieval-Augmented Generation system. "
        "Ingest documents, query them in natural language, and receive "
        "grounded answers with source citations."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(query_router)
app.include_router(documents_router)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    store = getattr(app.state, "store", None)
    return HealthResponse(
        status="healthy",
        vector_store=settings.vector_store,
        documents_indexed=store.document_count() if store else 0,
        version=settings.app_version,
    )


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
    )
