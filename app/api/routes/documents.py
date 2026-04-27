"""
Document management routes.

  POST   /ingest                — ingest a document into the vector store
  GET    /documents             — list all indexed documents
  DELETE /documents/{doc_id}   — remove a document from the index
"""

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.embeddings import EmbeddingModel
from app.models.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import BaseVectorStore, VectorRecord, get_vector_store
from app.utils.logger import get_logger

router = APIRouter(tags=["Documents"])
logger = get_logger(__name__)

# ── Dependency helpers ────────────────────────────────────────────────────────

def get_store() -> BaseVectorStore:
    return get_vector_store()


def get_embedder() -> EmbeddingModel:
    from app.main import app
    embedder: EmbeddingModel | None = getattr(app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model is not initialised.",
        )
    return embedder


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description=(
        "Upload a PDF, TXT, or Markdown file. It will be chunked, embedded, "
        "and stored in the vector index."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest (PDF, TXT, MD)"),
    chunk_size: int = Form(default=512, ge=64, le=2048),
    chunk_overlap: int = Form(default=64, ge=0, le=512),
    store: BaseVectorStore = Depends(get_store),
    embedder: EmbeddingModel = Depends(get_embedder),
) -> IngestResponse:
    """
    Process and index an uploaded document.

    - **file**: PDF, TXT, or Markdown file
    - **chunk_size**: tokens per chunk (default 512)
    - **chunk_overlap**: overlapping tokens between chunks (default 64)
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        document_id, chunks = processor.process_file(file_bytes, file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Document processing failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(exc)}",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded file.",
        )

    # Embed all chunks
    texts = [c.text for c in chunks]
    vectors = await embedder.embed_documents(texts)

    # Build VectorRecord objects
    records = [
        VectorRecord(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            filename=c.filename,
            text=c.text,
            page=c.page,
            chunk_index=c.chunk_index,
            token_count=c.token_count,
        )
        for c in chunks
    ]

    # Store in vector index
    await store.upsert(vectors, records)

    logger.info(f"Ingested '{file.filename}' → {len(chunks)} chunks (id={document_id})")

    return IngestResponse(
        document_id=document_id,
        filename=file.filename or "upload",
        chunks_created=len(chunks),
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def list_documents(
    store: BaseVectorStore = Depends(get_store),
) -> DocumentListResponse:
    docs_raw = store.list_documents()
    docs = [
        DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            chunks=d["chunks"],
            ingested_at=datetime.utcnow(),  # FAISS does not persist timestamps
        )
        for d in docs_raw
    ]
    return DocumentListResponse(total=len(docs), documents=docs)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Remove a document from the index",
)
async def delete_document(
    document_id: str,
    store: BaseVectorStore = Depends(get_store),
) -> DeleteResponse:
    removed = await store.delete_document(document_id)
    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in the index.",
        )
    logger.info(f"Deleted document id={document_id} ({removed} chunks removed)")
    return DeleteResponse(document_id=document_id)
