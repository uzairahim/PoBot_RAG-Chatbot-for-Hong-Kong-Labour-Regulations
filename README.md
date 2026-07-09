# PoBot — RAG Chatbot for Hong Kong Labour Regulations

A Retrieval-Augmented Generation (RAG) question-answering system that answers
questions about **Hong Kong labour & employment regulations** — grounded in
official HK Government sources, with inline citations, multilingual support
(English + Tagalog), and a confidence-based "I don't know" fallback.

Built as a take-home for a migrant-worker support use case (Filipino/Indonesian
domestic helpers and other workers navigating HK labour law).

---

## Features

| Requirement | Status |
|---|---|
| Collect ≥5 official regulation documents | ✅ **11 documents** (Labour Dept, Immigration Dept, MPFA) |
| Preprocess & chunk | ✅ extraction, cleaning, sentence-aware overlapping chunks (968 chunks) |
| Embed + vector store | ✅ local `multilingual-e5-small` → **FAISS** |
| RAG pipeline (LLM + grounded prompt) | ✅ swappable LLM, context-grounded answers |
| Chatbot interface (CLI) | ✅ interactive + one-shot |
| Evaluation | ✅ quantitative retrieval metrics + sample Q&As + failure cases |
| **Bonus:** source citations | ✅ inline `[n]` + source list with URLs |
| **Bonus:** multilingual (EN + Tagalog) | ✅ retrieval *and* generation |
| **Bonus:** confidence / fallback | ✅ calibrated cross-encoder confidence gate |
| **Bonus:** advanced retrieval | ✅ cross-encoder **re-ranking** |
| **Bonus:** fine-tuning | ✅ QLoRA notebook + dataset builder + `hf_local` provider (see `finetuning/`) |

---

## Architecture

```
                          INGESTION (one-time: python -m src.ingest)
  data/raw/*.pdf,*.html ─► extract+clean ─► chunk ─► embed (e5) ─► FAISS index
                                                                        │
                          QUERY TIME (python chatbot.py)                │
  question ─► embed ─► FAISS top-N ─► cross-encoder re-rank ─► top-K ───┤
                                              │                         │
                                     confidence gate                    │
                                  (sigmoid of rerank score)             │
                                      │            │                    │
                              low? → fallback   ok? → grounded prompt ─► LLM ─► cited answer
```

**Layering (each component has one job and one consumer):**
`Embedder` / `VectorStore` / `Reranker` → `Retriever` → `RagPipeline` → `chatbot.py`.

---

## Setup

Requires **Python 3.11+**.

```bash
# 1. Create an environment
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
#   or: conda create -n pobot-env python=3.11 && conda activate pobot-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
#   Add a FREE Groq API key (https://console.groq.com/keys) to .env:
#   GROQ_API_KEY=gsk_...
```

The LLM provider is swappable — set `LLM_PROVIDER` to `groq` (default), `openai`,
or `ollama` in `.env`. Embeddings and re-ranking run **locally** and need no key.

---

## Usage

```bash
# One-time: download sources and build the index
python -m src.fetch_sources        # -> data/raw/
python -m src.ingest               # -> data/processed/, data/index/

# Chat
python chatbot.py                          # interactive (English or Tagalog)
python chatbot.py "What are the rules for recruitment agencies?"   # one-shot
python chatbot.py --show-sources "..."     # also show retrieved chunks + scores
python chatbot.py --no-rerank "..."        # disable the cross-encoder re-ranker

# Evaluate & test
python -m src.evaluate --k 5       # Hit@k / MRR / nDCG, rerank on vs off
pytest -q                          # unit + integration tests
```

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` / `openai` / `ollama` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | model for the chosen provider |
| `GROQ_API_KEY` | — | free key from console.groq.com |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | local embeddings |
| `RERANK_ENABLED` | `true` | toggle cross-encoder re-ranking |
| `RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | multilingual re-ranker |
| `TOP_K` | `4` | chunks passed to the LLM |
| `MIN_CONFIDENCE` | `0.5` | fallback threshold (cross-encoder confidence) |
| `MIN_SCORE` | `0.70` | backup cosine gate (only when re-ranker off) |

All paths and settings are centralized in [`src/config.py`](src/config.py).

---

## Project structure

```
data/
  sources.json          # manifest of official sources (url, title, publisher)
  raw/                  # downloaded documents (collected)
  processed/            # chunks.jsonl + human-readable cleaned text
  index/                # FAISS index + chunk metadata
  eval/                 # gold questions + generated results.md
src/
  config.py             # single source of truth for paths + settings
  fetch_sources.py      # reproducible downloader
  preprocessing.py      # extract / clean / chunk (Chunk schema)
  embeddings.py         # local multilingual embedder
  vectorstore.py        # FAISS build/save/load/search
  reranker.py           # cross-encoder + score→confidence
  retriever.py          # embed → search → rerank → confidence
  llm.py                # pluggable LLM providers (groq/openai/ollama)
  rag.py                # prompt + grounding + citations + fallback gate
  evaluate.py           # Hit@k / MRR / nDCG harness
chatbot.py              # CLI
tests/                  # pytest: preprocessing, retrieval, rag, metrics, reranker
finetuning/             # (bonus) Colab-ready QLoRA notebook + dataset builder
EVALUATION.md           # metrics, sample Q&As, failure cases, approach summary
```

---

## How it works (key design decisions)

- **Local, open-source embeddings + re-ranking** (no per-query cost, no vendor
  lock-in) with a **swappable** LLM behind a one-method interface — change
  provider with one env var.
- **Two-stage retrieval:** cheap bi-encoder recall over all chunks, then an
  expensive cross-encoder only on the top-20 shortlist for precision.
- **Confidence via the cross-encoder, not cosine.** `multilingual-e5` cosines are
  compressed (even unrelated text scores ~0.75), so they can't gate a fallback.
  The cross-encoder score, squashed through a sigmoid, separates relevant (~0.94)
  from irrelevant (~0.03) cleanly. See [EVALUATION.md](EVALUATION.md#3-limitations-and-failure-cases).
- **Grounded prompting:** answers must use only retrieved context, cite `[n]`,
  match the question's language, and disclaim that this is general information.

---

## Limitations

Documented honestly in [EVALUATION.md](EVALUATION.md) — including "meta-answers"
on broad queries whose answer is fragmented across sources, no lexical/hybrid
search yet, and corpus recency. Answers are **general information, not legal
advice**.
