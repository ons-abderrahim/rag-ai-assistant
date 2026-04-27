# rag-ai-assistant
Retrieval-Augmented Generation system with FAISS/Pinecone vector database and FastAPI backend. Optimized latency and response quality through embedding search + LLM integration with full logging and evaluation.


# 🔍 RAG-based AI Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-00599C?style=for-the-badge&logo=meta&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, and OpenAI — featuring real-time query handling, structured response evaluation, and optional Pinecone support.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Evaluation](#-evaluation) · [Deployment](#-deployment)

</div>

---

## 📌 Overview

This project implements a complete **RAG pipeline** that grounds LLM responses in a custom document corpus. It combines dense vector retrieval (FAISS / Pinecone) with OpenAI's GPT models to produce factually grounded, context-aware answers with full source attribution.

### What makes this different

- **Modular architecture** — swap embedding models, vector stores, or LLMs with one config change
- **Built-in evaluation** — automated faithfulness, relevance, and latency benchmarking
- **LangSmith tracing** — every query traced end-to-end for observability
- **Production-ready** — async FastAPI, structured logging, Docker, CI/CD

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 **Document ingestion** | PDF, TXT, Markdown with recursive chunking and overlap |
| 🧠 **Embedding models** | OpenAI `text-embedding-3-small` (default) or any SentenceTransformer |
| 🗄️ **Vector stores** | FAISS (local) or Pinecone (cloud) — configurable via `.env` |
| 🤖 **LLM generation** | OpenAI GPT-4o-mini with streaming support |
| 🔍 **Hybrid retrieval** | Dense + BM25 sparse retrieval with reciprocal rank fusion |
| 📊 **Evaluation suite** | Faithfulness, answer relevancy, context precision metrics |
| 🔭 **Observability** | LangSmith tracing + structured JSON logging |
| 🐳 **Containerized** | Docker + docker-compose for one-command deployment |
| ✅ **Tested** | Pytest test suite with async support and mocked LLM calls |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Request                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI App   │  ← Async REST API
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │         RAG Pipeline         │
              │                             │
              │  1. Embed Query             │
              │     └─ OpenAI Embeddings    │
              │                             │
              │  2. Retrieve Chunks         │
              │     └─ FAISS / Pinecone     │
              │     └─ BM25 (sparse)        │
              │     └─ RRF Fusion           │
              │                             │
              │  3. Build Context           │
              │     └─ Reranking            │
              │     └─ Token budget mgmt    │
              │                             │
              │  4. Generate Response       │
              │     └─ GPT-4o-mini          │
              │     └─ Source attribution   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      LangSmith Tracing       │  ← Observability
              └─────────────────────────────┘
```

---

## 📁 Project Structure

```
rag-ai-assistant/
│
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Settings & environment config
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── query.py           # POST /query — ask a question
│   │       └── documents.py       # POST /ingest, GET /documents
│   │
│   ├── core/
│   │   ├── rag_pipeline.py        # Orchestrates the full RAG flow
│   │   ├── embeddings.py          # Embedding model wrapper
│   │   ├── retriever.py           # Hybrid retrieval logic (FAISS + BM25)
│   │   └── generator.py           # LLM generation + prompt templates
│   │
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response schemas
│   │
│   ├── services/
│   │   ├── vector_store.py        # FAISS / Pinecone abstraction layer
│   │   └── document_processor.py  # Ingestion, chunking, preprocessing
│   │
│   └── utils/
│       ├── logger.py              # Structured JSON logging
│       └── helpers.py             # Token counting, text utilities
│
├── data/
│   └── sample_docs/               # Drop documents here for ingestion
│
├── tests/
│   ├── conftest.py                # Pytest fixtures
│   ├── test_rag_pipeline.py       # Unit tests for the RAG core
│   ├── test_api.py                # Integration tests for API endpoints
│   └── test_document_processor.py # Tests for chunking & ingestion
│
├── scripts/
│   ├── ingest_documents.py        # CLI: bulk ingest a folder of docs
│   └── evaluate_rag.py            # CLI: run evaluation suite
│
├── docs/
│   ├── architecture.md            # Detailed architecture notes
│   └── api_reference.md           # Full API reference
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Docker (optional, for containerized run)

### 1. Clone & install

```bash
git clone https://github.com/ons-abderrahim/rag-ai-assistant.git
cd rag-ai-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Ingest your documents

```bash
# Drop PDFs / TXT / MD files into data/sample_docs/
python scripts/ingest_documents.py --source data/sample_docs/
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Query the assistant

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is domain adaptation?", "top_k": 5}'
```

---

## 📡 API Reference

### `POST /query`

Ask a question and receive a grounded answer with source citations.

**Request**
```json
{
  "question": "Explain unsupervised domain adaptation",
  "top_k": 5,
  "stream": false
}
```

**Response**
```json
{
  "answer": "Unsupervised domain adaptation is a technique that...",
  "sources": [
    {
      "chunk_id": "doc_001_chunk_3",
      "document": "domain_adaptation.pdf",
      "page": 4,
      "score": 0.91,
      "excerpt": "In unsupervised settings, the model adapts..."
    }
  ],
  "metadata": {
    "model": "gpt-4o-mini",
    "latency_ms": 843,
    "tokens_used": 1247,
    "retrieval_strategy": "hybrid_rrf"
  }
}
```

---

### `POST /ingest`

Ingest a document into the vector store.

**Request** — `multipart/form-data`
```
file: <your_document.pdf>
chunk_size: 512        (optional, default: 512)
chunk_overlap: 64      (optional, default: 64)
```

**Response**
```json
{
  "document_id": "doc_f3a9c1",
  "filename": "your_document.pdf",
  "chunks_created": 42,
  "status": "indexed"
}
```

---

### `GET /documents`

List all indexed documents.

```json
{
  "total": 7,
  "documents": [
    {
      "document_id": "doc_f3a9c1",
      "filename": "your_document.pdf",
      "chunks": 42,
      "ingested_at": "2025-05-01T14:23:00Z"
    }
  ]
}
```

---

### `DELETE /documents/{document_id}`

Remove a document from the index.

---

### `GET /health`

```json
{
  "status": "healthy",
  "vector_store": "faiss",
  "documents_indexed": 7,
  "version": "1.0.0"
}
```

---

## 📊 Evaluation

Run the built-in evaluation suite against a labeled Q&A dataset:

```bash
python scripts/evaluate_rag.py \
  --dataset data/eval_dataset.json \
  --output results/eval_report.json
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Does the answer stay within retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Are the retrieved chunks actually useful? |
| **Context Recall** | Did retrieval capture the necessary information? |
| **Latency (P50/P95)** | Response time percentiles |

**Sample output:**
```
┌─────────────────────────┬────────┐
│ Metric                  │ Score  │
├─────────────────────────┼────────┤
│ Faithfulness            │  0.91  │
│ Answer Relevancy        │  0.88  │
│ Context Precision       │  0.85  │
│ Context Recall          │  0.87  │
│ Avg Latency             │ 723ms  │
│ P95 Latency             │ 1.2s   │
└─────────────────────────┴────────┘
```

---

## 🐳 Deployment

### Docker (recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Manual Docker

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `VECTOR_STORE` | ❌ | `faiss` | `faiss` or `pinecone` |
| `PINECONE_API_KEY` | ❌ | — | Required if `VECTOR_STORE=pinecone` |
| `PINECONE_INDEX` | ❌ | `rag-index` | Pinecone index name |
| `EMBEDDING_MODEL` | ❌ | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | ❌ | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | ❌ | `512` | Tokens per document chunk |
| `CHUNK_OVERLAP` | ❌ | `64` | Overlap between consecutive chunks |
| `TOP_K` | ❌ | `5` | Default number of chunks to retrieve |
| `LANGSMITH_API_KEY` | ❌ | — | Enable LangSmith tracing |
| `LOG_LEVEL` | ❌ | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 🗺️ Roadmap

- [x] FAISS vector store with persistence
- [x] Pinecone cloud vector store
- [x] Hybrid BM25 + dense retrieval
- [x] LangSmith observability
- [x] Evaluation suite
- [ ] Streaming responses (SSE)
- [ ] Multi-modal support (images in PDFs)
- [ ] Conversation memory / multi-turn chat
- [ ] Web UI (Streamlit)
- [ ] Fine-tuned reranker

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Ons Abderrahim** — ML Engineer & AI Researcher

[![Portfolio](https://img.shields.io/badge/Portfolio-onsabderrahim.github.io-ff1a8c?style=flat-square)](https://onsabderrahim.github.io)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ons--abderrahim-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/ons-abderrahim)
