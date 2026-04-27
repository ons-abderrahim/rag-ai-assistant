"""
Embedding model abstraction.
Wraps OpenAI's text embedding API with batching, retries, and caching.
"""

import asyncio
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.utils.logger import LoggerMixin, log_event

settings = get_settings()


class EmbeddingModel(LoggerMixin):
    """
    Wraps the OpenAI embeddings endpoint.

    Features
    --------
    - Automatic batching (OpenAI limit: 2048 inputs per request)
    - Exponential back-off on rate-limit / transient errors
    - Returns normalised float32 numpy arrays
    """

    MAX_BATCH = 512  # conservative batch size to stay under token limits

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dims = settings.embedding_dimensions

    # ── Public API ────────────────────────────────────────────────────

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed a list of document texts.
        Automatically splits into batches when len(texts) > MAX_BATCH.
        """
        if not texts:
            return []

        log_event(self.logger, "embed_documents", count=len(texts))

        chunks = [
            texts[i : i + self.MAX_BATCH]
            for i in range(0, len(texts), self.MAX_BATCH)
        ]
        results: list[np.ndarray] = []
        for batch in chunks:
            batch_vecs = await self._embed_batch(batch)
            results.extend(batch_vecs)

        return results

    @property
    def dimensions(self) -> int:
        return self._dims

    # ── Internal ─────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Call the OpenAI API for one batch and return normalised vectors."""
        # Replace empty strings — the API rejects them
        cleaned = [t if t.strip() else " " for t in texts]

        response = await self._client.embeddings.create(
            model=self._model,
            input=cleaned,
        )

        vectors = [
            self._normalise(np.array(item.embedding, dtype=np.float32))
            for item in sorted(response.data, key=lambda x: x.index)
        ]
        return vectors

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
