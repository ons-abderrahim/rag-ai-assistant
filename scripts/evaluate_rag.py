#!/usr/bin/env python3
"""
CLI: Evaluate RAG pipeline quality against a labeled Q&A dataset.

Dataset format (JSON):
[
  {
    "question": "What is domain adaptation?",
    "ground_truth": "Domain adaptation is a technique that...",
    "relevant_docs": ["paper.pdf"]   (optional)
  },
  ...
]

Usage
-----
    python scripts/evaluate_rag.py --dataset data/eval_dataset.json
    python scripts/evaluate_rag.py --dataset data/eval_dataset.json --output results/report.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from app.core.embeddings import EmbeddingModel
from app.core.rag_pipeline import RAGPipeline
from app.services.vector_store import get_vector_store

console = Console()


# ── Metrics ───────────────────────────────────────────────────────────────────

def token_overlap_score(prediction: str, ground_truth: str) -> float:
    """Approximate faithfulness via token-level F1 overlap."""
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def relevancy_score(question: str, answer: str) -> float:
    """
    Heuristic answer relevancy: checks that question keywords appear in answer.
    In production, replace with an LLM-based judge.
    """
    q_words = {w.lower() for w in question.split() if len(w) > 3}
    a_words = {w.lower() for w in answer.split()}
    if not q_words:
        return 1.0
    return len(q_words & a_words) / len(q_words)


def context_precision(sources: list, relevant_docs: list[str]) -> float:
    """Fraction of retrieved chunks that come from a relevant document."""
    if not relevant_docs or not sources:
        return 0.0
    relevant_set = {d.lower() for d in relevant_docs}
    hits = sum(1 for s in sources if s["document"].lower() in relevant_set)
    return hits / len(sources)


# ── Runner ────────────────────────────────────────────────────────────────────

async def evaluate(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        sys.exit(1)

    with open(dataset_path) as f:
        dataset = json.load(f)

    console.print(f"\n[bold cyan]RAG AI Assistant — Evaluation Suite[/bold cyan]")
    console.print(f"Dataset: [bold]{dataset_path}[/bold]  ({len(dataset)} examples)\n")

    embedder = EmbeddingModel()
    store = get_vector_store()
    pipeline = RAGPipeline(embedding_model=embedder, vector_store=store)

    records = []
    latencies = []

    for i, example in enumerate(dataset, start=1):
        question = example["question"]
        ground_truth = example.get("ground_truth", "")
        relevant_docs = example.get("relevant_docs", [])

        console.print(f"[{i}/{len(dataset)}] {question[:80]}…")

        t0 = time.perf_counter()
        try:
            response = await pipeline.run(question=question, top_k=args.top_k)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")
            continue

        latencies.append(latency)

        sources_dicts = [s.model_dump() for s in response.sources]

        record = {
            "question": question,
            "answer": response.answer,
            "ground_truth": ground_truth,
            "faithfulness": token_overlap_score(response.answer, ground_truth) if ground_truth else None,
            "answer_relevancy": relevancy_score(question, response.answer),
            "context_precision": context_precision(sources_dicts, relevant_docs) if relevant_docs else None,
            "latency_ms": round(latency, 1),
            "tokens_used": response.metadata.tokens_used,
            "chunks_retrieved": response.metadata.chunks_retrieved,
        }
        records.append(record)

    # ── Aggregate ────────────────────────────────────────────────────
    def avg(values):
        v = [x for x in values if x is not None]
        return sum(v) / len(v) if v else None

    agg = {
        "faithfulness": avg(r["faithfulness"] for r in records),
        "answer_relevancy": avg(r["answer_relevancy"] for r in records),
        "context_precision": avg(r["context_precision"] for r in records),
        "latency_p50": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else None,
        "avg_tokens": avg(r["tokens_used"] for r in records),
        "total_examples": len(records),
    }

    # ── Display ──────────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Score", justify="right", style="bold")

    def fmt(v, pct=True):
        if v is None:
            return "[dim]N/A[/dim]"
        if pct:
            return f"[green]{v:.2%}[/green]" if v >= 0.7 else f"[yellow]{v:.2%}[/yellow]"
        return f"{v:.1f}"

    table.add_row("Faithfulness",     fmt(agg["faithfulness"]))
    table.add_row("Answer Relevancy", fmt(agg["answer_relevancy"]))
    table.add_row("Context Precision",fmt(agg["context_precision"]))
    table.add_row("Avg Latency (ms)", fmt(agg["latency_p50"], pct=False))
    table.add_row("P95 Latency (ms)", fmt(agg["latency_p95"], pct=False))
    table.add_row("Avg Tokens Used",  fmt(agg["avg_tokens"], pct=False))
    table.add_row("Total Examples",   str(agg["total_examples"]))

    console.print(table)

    # ── Save report ──────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {"summary": agg, "details": records}
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        console.print(f"\n[green]Report saved → {out}[/green]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality.")
    parser.add_argument("--dataset", required=True, help="Path to JSON evaluation dataset")
    parser.add_argument("--output", default=None, help="Output path for the JSON report")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per query")
    asyncio.run(evaluate(parser.parse_args()))
