"""Cross-encoder re-ranker.

A bi-encoder (our embedding model) scores a query and a chunk *independently*
and compares the two vectors — fast, but it can be fooled by surface similarity
(e.g. a table-of-contents chunk that merely repeats the query's keywords).

A cross-encoder instead reads the query and chunk *together* in one pass and
outputs a single relevance score. It's slower (so we only run it on the handful
of candidates the bi-encoder already shortlisted), but far more accurate at
ordering — which is exactly what pushes substantive chunks above boilerplate.

We use a multilingual cross-encoder so Tagalog queries benefit too.
"""
from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from src.config import settings


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model
        # max_length caps the query+passage token budget fed to the model.
        self.model = CrossEncoder(self.model_name, max_length=512)

    @staticmethod
    def score_to_confidence(logit: float) -> float:
        """Map a raw cross-encoder logit to a calibrated confidence in (0, 1).

        Cross-encoder scores are unbounded logits (not cosine!), so we squash
        them with a numerically-stable logistic/sigmoid. Irrelevant passages
        score strongly negative (-> ~0.03) and relevant ones positive (-> ~0.94),
        giving an interpretable "how sure are we" number. Monotonic, so it never
        changes the ranking. Callers compare this to settings.min_confidence.
        """
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        e = math.exp(logit)
        return e / (1.0 + e)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[tuple[dict, float]]:
        """Return the top_k (chunk, relevance_score) pairs, best first.

        `relevance_score` is the cross-encoder's raw logit (higher == more
        relevant). Scores are only meaningful *relative* to each other for the
        same query, so we use them for ordering, not as an absolute confidence.
        """
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, (float(s) for s in scores)),
                        key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
