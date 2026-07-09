"""PoBot CLI — a Retrieval-Augmented chatbot for Hong Kong labour regulations.

Usage:
    python chatbot.py                       # interactive chat
    python chatbot.py "your question here"  # single question, then exit
    python chatbot.py --show-sources        # also print retrieved chunks + scores
    python chatbot.py --no-rerank           # disable the cross-encoder re-ranker

Requires a built index (`python -m src.ingest`) and an LLM key in .env.
"""
from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.rag import Answer, RagPipeline
from src.retriever import Retriever


def format_answer(ans: Answer, show_sources: bool) -> str:
    lines = [ans.text]

    if ans.confidence is not None and not ans.used_fallback:
        lines.append(f"\n(confidence: {ans.confidence:.0%})")

    if ans.citations:
        lines.append("\nSources:")
        seen = set()
        for c in ans.citations:
            key = (c.title, c.page)
            if key in seen:
                continue
            seen.add(key)
            loc = f", p.{c.page}" if c.page else ""
            lines.append(f"  [{c.marker}] {c.title}{loc} — {c.url}")

    if show_sources and ans.contexts:
        lines.append("\nRetrieved context (debug):")
        conf = f"confidence={ans.confidence:.3f}" if ans.confidence is not None else f"best cosine={ans.best_cosine:.3f}"
        lines.append(f"  reranked={ans.reranked}, {conf}, fallback={ans.used_fallback}")
        for i, r in enumerate(ans.contexts, start=1):
            rr = f", rerank={r.rerank_score:+.2f}" if r.rerank_score is not None else ""
            snippet = r.chunk["text"][:100].strip().replace("\n", " ")
            lines.append(f"  [{i}] cos={r.cosine_score:.3f}{rr}  {r.chunk['source_id']} :: {snippet}...")

    return "\n".join(lines)


def build_pipeline(use_reranker: bool | None, top_k: int | None) -> RagPipeline:
    retriever = Retriever(use_reranker=use_reranker)
    if top_k:
        # top_k is read from settings by default; override per-run if requested.
        import src.config as cfg
        object.__setattr__(cfg.settings, "top_k", top_k)  # settings is frozen
    return RagPipeline(retriever=retriever)


def run_once(pipe: RagPipeline, question: str, show_sources: bool) -> None:
    ans = pipe.answer(question)
    print("\n" + format_answer(ans, show_sources) + "\n")


def interactive(pipe: RagPipeline, show_sources: bool) -> None:
    print("PoBot — Hong Kong labour regulations assistant")
    print(f"(provider: {settings.llm_provider}/{settings.llm_model} | "
          f"reranker: {'on' if pipe.retriever.use_reranker else 'off'})")
    print("Ask a question in English or Tagalog. Type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            return
        try:
            ans = pipe.answer(q)
            print("\nPoBot: " + format_answer(ans, show_sources) + "\n")
        except Exception as exc:  # keep the session alive on transient errors
            print(f"\n[error] {exc}\n", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="PoBot RAG chatbot (Hong Kong labour regulations).")
    parser.add_argument("question", nargs="*", help="ask a single question, then exit")
    parser.add_argument("--show-sources", action="store_true", help="print retrieved chunks + scores")
    parser.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder re-ranker")
    parser.add_argument("--top-k", type=int, default=None, help="number of chunks to retrieve")
    args = parser.parse_args()

    use_reranker = False if args.no_rerank else None
    try:
        pipe = build_pipeline(use_reranker=use_reranker, top_k=args.top_k)
    except FileNotFoundError as exc:
        print(f"[setup] {exc}", file=sys.stderr)
        return 1

    if args.question:
        run_once(pipe, " ".join(args.question), args.show_sources)
    else:
        interactive(pipe, args.show_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
