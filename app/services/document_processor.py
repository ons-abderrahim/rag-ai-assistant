"""
Document ingestion pipeline.

Supports: PDF, plain text (.txt), Markdown (.md)

Flow
----
1. Load raw text from file
2. Clean and normalise text
3. Split into overlapping token-based chunks
4. Return list of chunk dicts ready for vector store insertion
"""

import io
from dataclasses import dataclass, field
from pathlib import Path

import pypdf

from app.config import get_settings
from app.utils.helpers import (
    clean_text,
    count_tokens,
    generate_chunk_id,
    generate_document_id,
    get_encoder,
)
from app.utils.logger import LoggerMixin, log_event

settings = get_settings()


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page: int | None
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


class DocumentProcessor(LoggerMixin):
    """
    Converts raw files into lists of DocumentChunk objects.

    Parameters
    ----------
    chunk_size : int
        Maximum tokens per chunk.
    chunk_overlap : int
        Token overlap between consecutive chunks.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._encoder = get_encoder()

    # ── Public API ────────────────────────────────────────────────────

    def process_file(
        self, file_bytes: bytes, filename: str
    ) -> tuple[str, list[DocumentChunk]]:
        """
        Process a file and return (document_id, chunks).

        Parameters
        ----------
        file_bytes : bytes
            Raw file content.
        filename : str
            Original filename (used to detect format and generate ID).
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        document_id = generate_document_id(filename)
        log_event(self.logger, "process_file", filename=filename, doc_id=document_id)

        if suffix == ".pdf":
            pages = self._extract_pdf(file_bytes)
        else:
            pages = [(file_bytes.decode("utf-8", errors="replace"), None)]

        chunks = self._chunk_pages(pages, document_id, filename)
        log_event(
            self.logger,
            "process_file_complete",
            doc_id=document_id,
            chunks=len(chunks),
        )
        return document_id, chunks

    # ── Extraction ───────────────────────────────────────────────────

    def _extract_pdf(self, file_bytes: bytes) -> list[tuple[str, int]]:
        """Return list of (page_text, page_number) tuples."""
        pages: list[tuple[str, int]] = []
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, page_num))
        return pages

    # ── Chunking ─────────────────────────────────────────────────────

    def _chunk_pages(
        self,
        pages: list[tuple[str, int | None]],
        document_id: str,
        filename: str,
    ) -> list[DocumentChunk]:
        """Slide a token window over all pages and emit chunks."""
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for raw_text, page_num in pages:
            text = clean_text(raw_text)
            if not text:
                continue

            tokens = self._encoder.encode(text)
            start = 0

            while start < len(tokens):
                end = min(start + self._chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self._encoder.decode(chunk_tokens)

                chunk = DocumentChunk(
                    chunk_id=generate_chunk_id(document_id, chunk_index),
                    document_id=document_id,
                    filename=filename,
                    text=chunk_text,
                    page=page_num,
                    chunk_index=chunk_index,
                    token_count=len(chunk_tokens),
                )
                chunks.append(chunk)
                chunk_index += 1

                # Advance by (chunk_size - overlap) to create overlap
                step = max(1, self._chunk_size - self._chunk_overlap)
                start += step

        return chunks

    # ── Helpers ──────────────────────────────────────────────────────

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def get_chunk_overlap(self) -> int:
        return self._chunk_overlap
