# Architecture

## Overview

The system follows a clean layered architecture:

```
┌─────────────┐
│  API Layer  │  FastAPI routes — input validation, HTTP concerns
├─────────────┤
│  Core Layer │  RAG pipeline orchestration, retriever, generator
├─────────────┤
│  Services   │  Vector store I/O, document chunking
├─────────────┤
│  Utils      │  Token counting, logging, ID generation
└─────────────┘
```

## Component Responsibilities

### `app/core/rag_pipeline.py`
Top-level orchestrator. Calls retriever → generator in sequence, measures
end-to-end latency, and optionally traces to LangSmith.

### `app/core/embeddings.py`
Wraps the OpenAI embeddings API with automatic batching (≤512 texts/request)
and exponential back-off retries. Returns L2-normalised float32 vectors.

### `app/core/retriever.py`
Two-stage hybrid retrieval:
1. **Dense** — cosine similarity via FAISS/Pinecone inner product search
2. **Sparse** — BM25 applied to the dense-retrieved candidate set
3. **Fusion** — Reciprocal Rank Fusion (RRF, k=60) merges both ranked lists

### `app/core/generator.py`
Builds the context string from ranked chunks (respecting a 6k token budget)
and calls GPT-4o-mini. Prompt separates system instructions from the
retrieved context to maximise grounding.

### `app/services/vector_store.py`
Unified `BaseVectorStore` interface with two backends:
- **FAISSVectorStore** — local `IndexFlatIP` (exact IP search on unit vectors = cosine). Persists index + metadata to disk as `.faiss` + `.pkl`.
- **PineconeVectorStore** — cloud-managed index via official SDK.

### `app/services/document_processor.py`
Token-aware sliding window chunker. Supports PDF (via pypdf), TXT, and Markdown. Each chunk carries its source document ID, filename, page number, and token count.

## Data Flow

```
User request
  │
  ▼
QueryRequest (Pydantic validation)
  │
  ├─ embed_query()        → float32[1536]
  │
  ├─ dense_search()       → top 2k candidates
  ├─ bm25_search()        → top 2k candidates
  ├─ rrf_fusion()         → ranked merged list
  ├─ filter(threshold)    → top_k SourceChunks
  │
  ├─ build_context()      → string (≤6000 tokens)
  ├─ chat.completions()   → answer string
  │
  └─ QueryResponse        → JSON to client
```

## Token Budget

| Component | Budget |
|-----------|--------|
| System prompt | ~300 tokens |
| Retrieved context | ≤6 000 tokens |
| User question | ≤500 tokens |
| LLM answer | ≤1 024 tokens (configurable) |
| **Total** | ~8 000 tokens |

gpt-4o-mini supports 128k context, so this budget is deliberately conservative
to keep costs low and latency short.

## Extending the System

**Swap embedding model:** Set `EMBEDDING_MODEL` in `.env` to any OpenAI model
or implement `EmbeddingModel` for HuggingFace/local models.

**Swap LLM:** Change `LLM_MODEL`. Any OpenAI-compatible model works.

**Swap vector store:** Set `VECTOR_STORE=pinecone` and provide Pinecone credentials.

**Add reranking:** Insert a cross-encoder reranker between retrieval and generation
in `rag_pipeline.py`.

**Add conversation memory:** Pass conversation history as additional context in
`generator.py` and maintain session state in the API layer.
