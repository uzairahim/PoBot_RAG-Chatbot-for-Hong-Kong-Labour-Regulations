"""End-to-end ingestion: raw documents -> chunks -> embeddings -> FAISS index.

Run once (re-run whenever sources or chunking settings change):

    python -m src.ingest

Outputs:
    data/processed/chunks.jsonl      all chunks + metadata (the processed corpus)
    data/processed/text/<id>.txt     human-readable cleaned text (if enabled)
    data/index/faiss.index           the vector index
    data/index/chunks.jsonl          chunk metadata aligned to index rows
    data/index/meta.json             embedding model + counts
"""
from __future__ import annotations

import json

from src.config import (
    PROCESSED_CHUNKS_FILE,
    PROCESSED_DIR,
    PROCESSED_TEXT_DIR,
    RAW_DIR,
    SOURCES_FILE,
    settings,
)
from src.embeddings import Embedder
from src.preprocessing import cleaned_full_text, process_source
from src.vectorstore import VectorStore


def load_sources() -> list[dict]:
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))["sources"]


def main() -> int:
    sources = load_sources()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Extract + clean + chunk every source.
    all_chunks: list[dict] = []
    if settings.write_processed_text:
        PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(sources)} sources (chunk_size={settings.chunk_size}, "
          f"overlap={settings.chunk_overlap}) ...")
    for s in sources:
        chunks = process_source(s, RAW_DIR, settings.chunk_size, settings.chunk_overlap)
        all_chunks.extend(c.to_dict() for c in chunks)
        print(f"  {s['id']:<28} -> {len(chunks):>4} chunks")
        if settings.write_processed_text:
            (PROCESSED_TEXT_DIR / f"{s['id']}.txt").write_text(
                cleaned_full_text(s, RAW_DIR), encoding="utf-8"
            )

    # Persist the processed corpus (this is what the pipeline consumes).
    with open(PROCESSED_CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks -> {PROCESSED_CHUNKS_FILE}")

    # 2) Embed all chunks with the local multilingual model.
    print(f"\nEmbedding {len(all_chunks)} chunks with '{settings.embedding_model}' ...")
    embedder = Embedder()
    vectors = embedder.encode_passages(
        [c["text"] for c in all_chunks], show_progress=True
    )
    print(f"  embeddings shape: {vectors.shape}")

    # 3) Build + save the FAISS index.
    store = VectorStore.build(vectors, all_chunks, embedder.model_name)
    store.save()
    print(f"\nFAISS index built and saved to data/index/ "
          f"({store.index.ntotal} vectors, dim={store.index.d}).")
    print("Ingestion complete. You can now run: python chatbot.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
