"""Quantitative retrieval evaluation: Hit@k, MRR, nDCG@k.

Measures how well retrieval finds chunks from the *correct source document* for a
labeled gold question set, and compares the bi-encoder alone vs. bi-encoder +
cross-encoder re-ranking.

    python -m src.evaluate            # run eval, print table, write results.md
    python -m src.evaluate --k 5      # evaluate at a different cut-off

NOTE on granularity: ground truth is document-level (a chunk is "relevant" if it
comes from a labeled relevant source). This is a proxy for true chunk-level
relevance — modest but honest, and enough to compare configurations fairly.
"""
from __future__ import annotations

import argparse
import json
import math

from src.config import EVAL_RESULTS_FILE, GOLD_QUESTIONS_FILE, settings
from src.retriever import Retriever
from src.vectorstore import VectorStore

# --------------------------------------------------------------------------
# Pure metric functions (unit-tested). Each takes a binary relevance list
# `rels` aligned to ranked results, e.g. [0, 1, 0, 1] means results 2 and 4
# were relevant.
# --------------------------------------------------------------------------

def hit_at_k(rels: list[int], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return 1.0 if any(rels[:k]) else 0.0


def reciprocal_rank(rels: list[int]) -> float:
    """1 / rank of the first relevant result (0 if none)."""
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def dcg_at_k(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))


def ndcg_at_k(rels: list[int], k: int) -> float:
    """Normalized DCG@k with binary relevance."""
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def load_gold() -> list[dict]:
    return json.loads(GOLD_QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]


def relevance_list(retriever: Retriever, question: str, relevant: set[str], k: int) -> list[int]:
    res = retriever.retrieve(question, top_k=k)
    return [1 if r.chunk["source_id"] in relevant else 0 for r in res.results]


def evaluate(retriever: Retriever, gold: list[dict], k: int) -> dict:
    hits, rrs, ndcgs = [], [], []
    for item in gold:
        rels = relevance_list(retriever, item["question"], set(item["relevant_sources"]), k)
        hits.append(hit_at_k(rels, k))
        rrs.append(reciprocal_rank(rels))
        ndcgs.append(ndcg_at_k(rels, k))
    n = len(gold)
    return {
        f"Hit@{k}": sum(hits) / n,
        "MRR": sum(rrs) / n,
        f"nDCG@{k}": sum(ndcgs) / n,
    }


def format_report(k: int, off: dict, on: dict, n: int) -> str:
    lines = [
        "# Retrieval Evaluation",
        "",
        f"Gold set: **{n} questions** · cut-off **k={k}** · document-level relevance.",
        f"Embedding: `{settings.embedding_model}` · Re-ranker: `{settings.rerank_model}`.",
        "",
        f"| Metric | Bi-encoder only | + Cross-encoder rerank | Δ |",
        f"|---|--:|--:|--:|",
    ]
    for metric in off:
        a, b = off[metric], on[metric]
        lines.append(f"| {metric} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    lines += [
        "",
        "**Hit@k** = fraction of questions with a correct-source chunk in the top-k.  ",
        "**MRR** = mean reciprocal rank of the first correct-source chunk.  ",
        "**nDCG@k** = ranking quality (rewards correct chunks appearing higher).",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (Hit@k, MRR, nDCG).")
    parser.add_argument("--k", type=int, default=5, help="evaluation cut-off (default 5)")
    args = parser.parse_args()

    gold = load_gold()
    store = VectorStore.load()  # load once, share across both configs

    print(f"Evaluating {len(gold)} gold questions at k={args.k} ...\n")
    off = evaluate(Retriever(store=store, use_reranker=False), gold, args.k)
    print("  bi-encoder only:          ", {m: round(v, 3) for m, v in off.items()})
    on = evaluate(Retriever(store=store, use_reranker=True), gold, args.k)
    print("  + cross-encoder rerank:   ", {m: round(v, 3) for m, v in on.items()})

    report = format_report(args.k, off, on, len(gold))
    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport written to {EVAL_RESULTS_FILE}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
