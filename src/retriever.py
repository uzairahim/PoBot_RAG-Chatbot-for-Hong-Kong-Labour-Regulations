"""Retrieval orchestration: query -> ranked, cited chunks.

Pipeline:
    1. Embed the query (bi-encoder).
    2. Fetch a shortlist of candidates from FAISS by cosine similarity.
    3. (optional) Re-rank the shortlist with a cross-encoder and keep top_k.
    4. Report the best cosine score so the caller can decide the fallback.

This module knows nothing about the LLM — it only finds relevant context.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.embeddings import Embedder
from src.reranker import Reranker
from src.vectorstore import VectorStore


@dataclass
class Retrieved:
    """One retrieved chunk with its scores."""
    chunk: dict
    cosine_score: float
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    results: list[Retrieved]
    best_cosine: float          # best cosine over all candidates (backup gate)
    reranked: bool
    # Calibrated confidence in (0,1) from the top cross-encoder score — the
    # preferred confidence signal. None when the re-ranker is off (then callers
    # fall back to best_cosine). The sigmoid mapping lives in Reranker.
    confidence: float | None = None


class Retriever:
    def __init__(self, store: VectorStore | None = None, use_reranker: bool | None = None):
        self.store = store or VectorStore.load()
        self.embedder = Embedder(self.store.embedding_model or settings.embedding_model)
        self.use_reranker = settings.rerank_enabled if use_reranker is None else use_reranker
        self._reranker: Reranker | None = None  # loaded lazily on first use

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or settings.top_k

        # How many candidates to pull from FAISS before (optionally) re-ranking.
        n_candidates = max(settings.rerank_candidates, top_k) if self.use_reranker else top_k

        qv = self.embedder.encode_query(query)
        hits = self.store.search(qv, top_k=n_candidates)   # [(chunk, cosine), ...]
        if not hits:
            return RetrievalResult(results=[], best_cosine=0.0, reranked=False)

        best_cosine = max(score for _, score in hits)
        cosine_by_id = {c["chunk_id"]: score for c, score in hits}
        candidates = [c for c, _ in hits]

        if self.use_reranker:
            ranked = self.reranker.rerank(query, candidates, top_k=top_k)
            results = [
                Retrieved(chunk=c,
                          cosine_score=cosine_by_id.get(c["chunk_id"], 0.0),
                          rerank_score=rscore)
                for c, rscore in ranked
            ]
            top_rerank = results[0].rerank_score if results else None
            confidence = (self.reranker.score_to_confidence(top_rerank)
                          if top_rerank is not None else None)
            return RetrievalResult(results=results, best_cosine=best_cosine,
                                   reranked=True, confidence=confidence)

        results = [Retrieved(chunk=c, cosine_score=score) for c, score in hits[:top_k]]
        return RetrievalResult(results=results, best_cosine=best_cosine, reranked=False)
