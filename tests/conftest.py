"""
Shared pytest fixtures.
All LLM / embedding / vector-store calls are mocked so tests run
without an OpenAI key and without a real FAISS index.
"""

import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# ── Environment stubs (must be set before importing app) ─────────────────────
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("VECTOR_STORE", "faiss")
os.environ.setdefault("FAISS_INDEX_PATH", "/tmp/test_faiss_index")

from app.main import app  # noqa: E402  (import after env is patched)


# ── Fake embeddings ───────────────────────────────────────────────────────────

FAKE_VECTOR = np.random.rand(1536).astype(np.float32)
FAKE_VECTOR /= np.linalg.norm(FAKE_VECTOR)


@pytest.fixture
def mock_embedder():
    """Return an EmbeddingModel whose API calls are stubbed."""
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=FAKE_VECTOR)
    embedder.embed_documents = AsyncMock(return_value=[FAKE_VECTOR])
    embedder.dimensions = 1536
    return embedder


# ── Fake vector store ────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    from app.services.vector_store import SearchResult

    store = MagicMock()
    store.search = AsyncMock(
        return_value=[
            SearchResult(
                chunk_id="doc_test_chunk_0000",
                document_id="doc_test",
                filename="test.pdf",
                text="Domain adaptation is a machine learning technique that...",
                page=1,
                score=0.92,
            )
        ]
    )
    store.upsert = AsyncMock(return_value=None)
    store.delete_document = AsyncMock(return_value=1)
    store.list_documents = MagicMock(
        return_value=[
            {"document_id": "doc_test", "filename": "test.pdf", "chunks": 10}
        ]
    )
    store.document_count = MagicMock(return_value=1)
    return store


# ── Fake OpenAI chat completion ───────────────────────────────────────────────

@pytest.fixture
def mock_openai_completion():
    """Patch the AsyncOpenAI client used by ResponseGenerator."""
    completion = MagicMock()
    completion.choices = [
        MagicMock(message=MagicMock(content="This is a test answer."))
    ]
    completion.usage = MagicMock(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    )

    with patch("app.core.generator.AsyncOpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.chat = MagicMock()
        instance.chat.completions = MagicMock()
        instance.chat.completions.create = AsyncMock(return_value=completion)
        yield instance


# ── HTTP test clients ─────────────────────────────────────────────────────────

@pytest.fixture
def sync_client(mock_embedder, mock_store, mock_openai_completion):
    """Sync TestClient with patched state."""
    app.state.embedder = mock_embedder
    app.state.store = mock_store
    app.state.pipeline = _build_pipeline(mock_embedder, mock_store)
    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_client(
    mock_embedder, mock_store, mock_openai_completion
) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTPX client for async test cases."""
    app.state.embedder = mock_embedder
    app.state.store = mock_store
    app.state.pipeline = _build_pipeline(mock_embedder, mock_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_pipeline(embedder, store):
    from app.core.rag_pipeline import RAGPipeline
    return RAGPipeline(embedding_model=embedder, vector_store=store)
