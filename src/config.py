"""Central configuration, loaded from environment / .env.

Every tunable knob lives here so the rest of the code never reads os.environ
directly. Import `settings` and use its attributes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from src/) if present.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# faiss and torch each ship their own OpenMP runtime; on Windows, loading a
# second large torch model (the hf_local LLM) on top of faiss can abort the
# process with a duplicate-OpenMP crash (exit 139). Allowing the duplicate is
# the accepted workaround. Set before torch/faiss import; setdefault so an
# explicit user value still wins. Harmless on platforms without this clash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- Filesystem layout (single source of truth for where everything lives) ---
# If the storage layout ever changes, change it HERE and nowhere else.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # original downloaded documents
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned text + chunk JSON
INDEX_DIR = DATA_DIR / "index"        # FAISS index + chunk metadata

# Individual files
SOURCES_FILE = DATA_DIR / "sources.json"            # source manifest (input)
PROCESSED_CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"  # processed corpus
PROCESSED_TEXT_DIR = PROCESSED_DIR / "text"         # human-readable cleaned text
INDEX_FILE = INDEX_DIR / "faiss.index"              # FAISS vectors
INDEX_CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"      # chunk metadata (aligned to rows)
INDEX_META_FILE = INDEX_DIR / "meta.json"           # embedding model + counts

EVAL_DIR = DATA_DIR / "eval"                        # evaluation harness
GOLD_QUESTIONS_FILE = EVAL_DIR / "gold_questions.json"  # labeled Q -> relevant source(s)
EVAL_RESULTS_FILE = EVAL_DIR / "results.md"         # generated metrics report


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # LLM provider (pluggable — see src/llm.py)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    # hf_local provider: run our fine-tuned model (base + LoRA adapter) in-process.
    hf_base_model: str = field(
        default_factory=lambda: os.getenv("HF_BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    )
    hf_adapter_dir: str = field(default_factory=lambda: os.getenv("HF_ADAPTER_DIR", ""))

    # Embeddings (local, open-source, multilingual)
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    )

    # Chunking
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))

    # Retrieval / confidence gate.
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 4))
    # Confidence gate = sigmoid(top cross-encoder score); fall back below this.
    # Used when the re-ranker is ON (the default) — it's a calibrated, well-
    # separated signal (irrelevant ~3%, relevant ~94%). 0.5 == logit 0.
    min_confidence: float = field(default_factory=lambda: _get_float("MIN_CONFIDENCE", 0.5))
    # Backup gate on bi-encoder cosine, used only when the re-ranker is OFF.
    # multilingual-e5 has a compressed cosine range (even unrelated text ~0.75),
    # so this is a weak signal — kept only for the no-reranker path.
    min_score: float = field(default_factory=lambda: _get_float("MIN_SCORE", 0.70))

    # Re-ranking (cross-encoder). Fetch `rerank_candidates` chunks by cosine,
    # then re-score them jointly with the query and keep the best `top_k`.
    rerank_enabled: bool = field(
        default_factory=lambda: os.getenv("RERANK_ENABLED", "true").lower() != "false"
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    )
    rerank_candidates: int = field(default_factory=lambda: _get_int("RERANK_CANDIDATES", 20))

    # Diagnostics: also write human-readable cleaned .txt per document during
    # ingest. Not consumed by the pipeline; useful for reviewing cleaning quality.
    write_processed_text: bool = field(
        default_factory=lambda: os.getenv("WRITE_PROCESSED_TEXT", "true").lower() != "false"
    )


settings = Settings()
