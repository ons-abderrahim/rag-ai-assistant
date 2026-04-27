"""
Utility functions: token counting, text normalisation, ID generation.
"""

import hashlib
import re
import uuid

import tiktoken

from app.config import get_settings

_settings = get_settings()
_ENCODER: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Return a cached tiktoken encoder for the configured embedding model."""
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.encoding_for_model(_settings.embedding_model)
        except KeyError:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count the number of tokens in *text* using the configured encoder."""
    return len(get_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens."""
    enc = get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove non-printable characters.
    Keeps newlines as sentence boundaries.
    """
    text = re.sub(r"[^\S\n]+", " ", text)      # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse triple+ newlines
    text = re.sub(r"[^\x20-\x7E\n]", "", text)  # drop non-ASCII non-printable
    return text.strip()


def generate_document_id(filename: str) -> str:
    """Deterministic document ID based on filename."""
    digest = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"doc_{digest}"


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Deterministic chunk ID based on document and position."""
    return f"{document_id}_chunk_{chunk_index:04d}"


def generate_run_id() -> str:
    """Random UUID for a single query run (for tracing)."""
    return str(uuid.uuid4())
