#!/usr/bin/env python3
"""
CLI: Bulk-ingest a folder of documents into the vector store.

Usage
-----
    python scripts/ingest_documents.py --source data/sample_docs/
    python scripts/ingest_documents.py --source docs/ --chunk-size 256 --chunk-overlap 32
    python scripts/ingest_documents.py --source report.pdf          # single file
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from app.core.embeddings import EmbeddingModel
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorRecord, get_vector_store

console = Console()

SUPPORTED = {".pdf", ".txt", ".md"}


async def ingest_file(
    path: Path,
    processor: DocumentProcessor,
    embedder: EmbeddingModel,
    store,
) -> dict:
    """Ingest a single file. Returns a result dict."""
    file_bytes = path.read_bytes()
    doc_id, chunks = processor.process_file(file_bytes, path.name)

    if not chunks:
        return {"file": path.name, "status": "skipped", "chunks": 0, "reason": "no text extracted"}

    texts = [c.text for c in chunks]
    vectors = await embedder.embed_documents(texts)

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
    await store.upsert(vectors, records)

    return {"file": path.name, "doc_id": doc_id, "status": "ok", "chunks": len(chunks)}


async def main(args: argparse.Namespace) -> None:
    source = Path(args.source)

    # Collect files
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = [f for f in source.rglob("*") if f.suffix.lower() in SUPPORTED]
    else:
        console.print(f"[red]Source not found: {source}[/red]")
        sys.exit(1)

    if not files:
        console.print(f"[yellow]No supported files found in {source}[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold cyan]RAG AI Assistant — Document Ingestion[/bold cyan]")
    console.print(f"Found [bold]{len(files)}[/bold] file(s) to ingest\n")

    processor = DocumentProcessor(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    embedder = EmbeddingModel()
    store = get_vector_store()

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting…", total=len(files))
        for f in files:
            progress.update(task, description=f"  {f.name[:50]}")
            try:
                result = await ingest_file(f, processor, embedder, store)
            except Exception as exc:
                result = {"file": f.name, "status": "error", "chunks": 0, "reason": str(exc)}
            results.append(result)
            progress.advance(task)

    # Summary table
    table = Table(title="Ingestion Summary", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Chunks", justify="right")

    ok = err = skip = 0
    for r in results:
        status = r["status"]
        colour = {"ok": "green", "error": "red", "skipped": "yellow"}.get(status, "white")
        table.add_row(r["file"], f"[{colour}]{status}[/{colour}]", str(r.get("chunks", 0)))
        if status == "ok":
            ok += 1
        elif status == "error":
            err += 1
        else:
            skip += 1

    console.print(table)
    console.print(
        f"\n✓ [green]{ok} ingested[/green]  "
        f"⚠ [yellow]{skip} skipped[/yellow]  "
        f"✗ [red]{err} errors[/red]\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest documents into the RAG vector store.")
    parser.add_argument("--source", required=True, help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=512, help="Tokens per chunk (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Token overlap (default: 64)")
    asyncio.run(main(parser.parse_args()))
