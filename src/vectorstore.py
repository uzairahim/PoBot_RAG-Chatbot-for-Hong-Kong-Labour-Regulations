"""FAISS vector store: build, persist, load, and search.

Embeddings are L2-normalized, so a FAISS inner-product index (IndexFlatIP)
returns cosine similarity in [-1, 1]. Chunk metadata is stored alongside the
index as JSONL so retrieved vectors can be mapped back to text + citations.
"""
from __future__ import annotations

import json

import faiss
import numpy as np

from src.config import INDEX_CHUNKS_FILE, INDEX_DIR, INDEX_FILE, INDEX_META_FILE


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[dict], embedding_model: str):
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model

    # --- build / persist ---
    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: list[dict], embedding_model: str) -> "VectorStore":
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(index, chunks, embedding_model)

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        # NOTE: faiss.write_index/read_index pass the path to C++ which can't
        # open non-ASCII paths on Windows (this repo's path contains an en-dash).
        # Serialize to bytes and let Python handle the (Unicode-safe) file I/O.
        INDEX_FILE.write_bytes(faiss.serialize_index(self.index).tobytes())
        with open(INDEX_CHUNKS_FILE, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        INDEX_META_FILE.write_text(
            json.dumps({"embedding_model": self.embedding_model,
                        "num_chunks": len(self.chunks),
                        "dim": self.index.d}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "VectorStore":
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"No FAISS index at {INDEX_FILE}. Run `python -m src.ingest` first."
            )
        data = np.frombuffer(INDEX_FILE.read_bytes(), dtype=np.uint8).copy()
        index = faiss.deserialize_index(data)
        chunks = [json.loads(ln) for ln in
                  INDEX_CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
        meta = json.loads(INDEX_META_FILE.read_text(encoding="utf-8"))
        return cls(index, chunks, meta.get("embedding_model", ""))

    # --- search ---
    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        """Return up to top_k (chunk, cosine_score) pairs, best first."""
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, idxs = self.index.search(query_vec, top_k)
        results: list[tuple[dict, float]] = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
