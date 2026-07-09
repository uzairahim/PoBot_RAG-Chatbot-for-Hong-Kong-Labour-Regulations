"""Local, open-source embedding model wrapper.

Uses sentence-transformers with a multilingual E5 model so English AND Tagalog
queries embed into the same space (supports the multilingual bonus).

E5 models are trained with instruction prefixes: documents must be prefixed
with "passage: " and search queries with "query: ". We bake that in here so
callers never have to remember it.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._is_e5 = "e5" in self.model_name.lower()
        self.model = SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        # E5 requires "query:"/"passage:" prefixes; other models don't.
        if not self._is_e5:
            return texts
        tag = "query: " if kind == "query" else "passage: "
        return [tag + t for t in texts]

    def encode_passages(self, texts: list[str], batch_size: int = 32,
                        show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            self._prefix(texts, "passage"),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # unit vectors -> inner product == cosine
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            self._prefix([text], "query"),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")
