"""
Integration tests for all FastAPI endpoints.
All external services (OpenAI, FAISS) are mocked via conftest.py.
"""

import io
import pytest


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_ok(sync_client):
    resp = sync_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "vector_store" in body
    assert "documents_indexed" in body
    assert "version" in body


# ── /query ───────────────────────────────────────────────────────────────────

def test_query_success(sync_client):
    resp = sync_client.post(
        "/query",
        json={"question": "What is domain adaptation?", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert "metadata" in body
    assert body["question"] == "What is domain adaptation?"
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0


def test_query_missing_question(sync_client):
    resp = sync_client.post("/query", json={})
    assert resp.status_code == 422


def test_query_too_short(sync_client):
    resp = sync_client.post("/query", json={"question": "Hi"})
    assert resp.status_code == 422  # min_length=3 — "Hi" is 2 chars


def test_query_top_k_bounds(sync_client):
    # top_k must be between 1 and 20
    resp = sync_client.post(
        "/query", json={"question": "Valid question here?", "top_k": 0}
    )
    assert resp.status_code == 422

    resp = sync_client.post(
        "/query", json={"question": "Valid question here?", "top_k": 21}
    )
    assert resp.status_code == 422


def test_query_metadata_fields(sync_client):
    resp = sync_client.post("/query", json={"question": "Explain RAG systems."})
    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    for field in ("model", "latency_ms", "tokens_used", "retrieval_strategy"):
        assert field in meta, f"Missing metadata field: {field}"


# ── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_txt_success(sync_client):
    content = b"Domain adaptation is a machine learning technique used to transfer knowledge across distributions."
    resp = sync_client.post(
        "/ingest",
        files={"file": ("test_doc.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "test_doc.txt"
    assert body["status"] == "indexed"
    assert body["chunks_created"] >= 1


def test_ingest_empty_file(sync_client):
    resp = sync_client.post(
        "/ingest",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code == 422


def test_ingest_unsupported_format(sync_client):
    resp = sync_client.post(
        "/ingest",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert resp.status_code == 415


# ── /documents ───────────────────────────────────────────────────────────────

def test_list_documents(sync_client):
    resp = sync_client.get("/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert "total" in body
    assert "documents" in body
    assert isinstance(body["documents"], list)


def test_delete_document_success(sync_client):
    resp = sync_client.delete("/documents/doc_test")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_delete_document_not_found(sync_client, mock_store):
    from unittest.mock import AsyncMock
    mock_store.delete_document = AsyncMock(return_value=0)
    resp = sync_client.delete("/documents/nonexistent_id")
    assert resp.status_code == 404


# ── Async tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_async(async_client):
    resp = await async_client.post(
        "/query",
        json={"question": "How does retrieval augmented generation work?"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["answer"]) > 0
