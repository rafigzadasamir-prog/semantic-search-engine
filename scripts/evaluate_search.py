#!/usr/bin/env python
"""Evaluate lexical-only, dense-only, hybrid, and hybrid+rerank retrieval.

Runs every query in a hand-labeled query -> relevant-doc-ids set through each
pipeline mode and reports precision@5, recall@5, precision@10, recall@10, and
MRR@10, averaged across queries. This is what backs the IR-metrics comparison
table in the README.

Usage:
    python scripts/evaluate_search.py
    python scripts/evaluate_search.py --labeled-queries data/labeled_queries.json --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_search_engine.factory import build_pipeline  # noqa: E402
from semantic_search_engine.metrics import (  # noqa: E402
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from semantic_search_engine.pipeline import SearchMode, SearchPipeline  # noqa: E402

MODES_TO_LABELS = {
    SearchMode.LEXICAL: "Lexical only (BM25)",
    SearchMode.DENSE: "Dense only (embeddings)",
    SearchMode.HYBRID: "Hybrid (RRF fusion, no rerank)",
    SearchMode.HYBRID_RERANK: "Hybrid + cross-encoder rerank",
}


def load_labeled_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_mode(
    pipeline: SearchPipeline, mode: SearchMode, labeled_queries: list[dict], top_k: int = 10
) -> dict[str, float]:
    p5s, r5s, p10s, r10s, rrs = [], [], [], [], []
    for item in labeled_queries:
        relevant = set(item["relevant_doc_ids"])
        results = pipeline.search(item["query"], top_k=top_k, mode=mode)
        retrieved = [r.doc_id for r in results]
        p5s.append(precision_at_k(retrieved, relevant, k=5))
        r5s.append(recall_at_k(retrieved, relevant, k=5))
        p10s.append(precision_at_k(retrieved, relevant, k=10))
        r10s.append(recall_at_k(retrieved, relevant, k=10))
        rrs.append(reciprocal_rank(retrieved, relevant))
    return {
        "precision@5": mean(p5s),
        "recall@5": mean(r5s),
        "precision@10": mean(p10s),
        "recall@10": mean(r10s),
        "mrr@10": mean(rrs),
    }


def format_markdown_table(all_results: dict[str, dict[str, float]]) -> str:
    headers = ["Mode", "Precision@5", "Recall@5", "Precision@10", "Recall@10", "MRR@10"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for mode_label, scores in all_results.items():
        row = [mode_label] + [f"{scores[k]:.3f}" for k in [
            "precision@5", "recall@5", "precision@10", "recall@10", "mrr@10"
        ]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeled-queries",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled_queries.json",
        help="Path to the hand-labeled query -> relevant-doc-ids JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Results per query to evaluate.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to save results as JSON."
    )
    args = parser.parse_args()

    labeled_queries = load_labeled_queries(args.labeled_queries)
    print(f"Loaded {len(labeled_queries)} labeled queries from {args.labeled_queries}")

    print("Building pipeline (loading BM25 index, embedding model, cross-encoder)...")
    start = time.time()
    pipeline = build_pipeline()
    print(f"Pipeline ready in {time.time() - start:.1f}s")

    all_results: dict[str, dict[str, float]] = {}
    for mode, label in MODES_TO_LABELS.items():
        start = time.time()
        scores = evaluate_mode(pipeline, mode, labeled_queries, top_k=args.top_k)
        elapsed = time.time() - start
        all_results[label] = scores
        print(f"\n{label} (evaluated in {elapsed:.1f}s):")
        for metric_name, value in scores.items():
            print(f"  {metric_name:14s}: {value:.3f}")

    print("\nMarkdown summary table:\n")
    table = format_markdown_table(all_results)
    print(table)

    if args.output:
        args.output.write_text(
            json.dumps(all_results, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nSaved raw results to {args.output}")


if __name__ == "__main__":
    main()
