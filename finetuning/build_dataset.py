"""Build a fine-tuning dataset from our HK labour corpus (runs LOCALLY).

For each sampled chunk we:
  1. ask the strong teacher LLM (Groq 70B) to write a realistic question, then
  2. run our REAL RAG pipeline to produce a grounded, cited answer.

The saved example uses the exact production prompt (`build_prompt_messages`), so
the student model is trained on the same input distribution it will see at
inference. This is knowledge distillation: teacher = 70B, student = small model.

We also add out-of-domain examples whose target is the fallback message, so the
student learns to refuse instead of hallucinate.

    python finetuning/build_dataset.py            # uses config.yaml
    python finetuning/build_dataset.py --limit 30 # quick/cheap run

Output: finetuning/data/train.jsonl and val.jsonl (chat format).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml

FT_DIR = Path(__file__).resolve().parent
ROOT = FT_DIR.parent
sys.path.insert(0, str(ROOT))

from src.config import INDEX_CHUNKS_FILE  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.rag import FALLBACK_MESSAGE, RagPipeline, build_prompt_messages  # noqa: E402
from src.retriever import Retriever  # noqa: E402

# Out-of-domain questions -> teach the model to decline (target = fallback).
OUT_OF_DOMAIN = [
    "What is the capital of France?",
    "How do I bake chocolate chip cookies?",
    "What's the weather like in Manila today?",
    "Can you write me a poem about the ocean?",
    "Who won the last FIFA World Cup?",
    "What is the boiling point of water?",
]

QUESTION_PROMPT = (
    "You are helping build a training set about Hong Kong labour law. Given the "
    "passage below from an official HK labour document, write ONE clear, natural "
    "question that a migrant worker or employer would realistically ask and that "
    "is directly answered by the passage. Return ONLY the question, nothing else.\n\n"
    "PASSAGE:\n{passage}\n\nQuestion:"
)


def load_config() -> dict:
    return yaml.safe_load((FT_DIR / "config.yaml").read_text(encoding="utf-8"))


def load_chunks() -> list[dict]:
    return [json.loads(ln) for ln in
            INDEX_CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def is_good_seed(text: str) -> bool:
    """Skip table-of-contents / index / too-short chunks — they make poor seeds."""
    if len(text) < 200:
        return False
    if text.count("....") >= 2 or text.count(". . .") >= 2:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) > 0.6


def generate_question(llm, passage: str) -> str | None:
    try:
        q = llm.chat(
            [{"role": "user", "content": QUESTION_PROMPT.format(passage=passage[:1200])}],
            temperature=0.7, max_tokens=80,
        ).strip().strip('"')
        return q if q.endswith("?") and len(q) > 10 else None
    except Exception as exc:
        print(f"    [warn] question gen failed: {exc}", file=sys.stderr)
        time.sleep(5)  # back off on rate limit
        return None


def to_example(messages: list[dict], answer: str) -> dict:
    return {"messages": messages + [{"role": "assistant", "content": answer}]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the QLoRA fine-tuning dataset.")
    parser.add_argument("--limit", type=int, default=None, help="cap number of source chunks (cheap runs)")
    args = parser.parse_args()

    cfg = load_config()["dataset"]
    rng = random.Random(cfg["seed"])
    n_chunks = args.limit or cfg["num_source_chunks"]

    chunks = [c for c in load_chunks() if is_good_seed(c["text"])]
    rng.shuffle(chunks)
    seeds = chunks[:n_chunks]
    print(f"Generating from {len(seeds)} seed chunks (of {len(chunks)} usable)...")

    llm = get_llm()
    pipe = RagPipeline(retriever=Retriever())

    examples: list[dict] = []
    for i, chunk in enumerate(seeds, start=1):
        question = generate_question(llm, chunk["text"])
        if not question:
            continue
        ans = pipe.answer(question)            # real retrieval + teacher answer
        if ans.used_fallback or len(ans.text) < 40:
            continue
        messages = build_prompt_messages(question, ans.contexts)
        examples.append(to_example(messages, ans.text))
        if i % 10 == 0:
            print(f"  {i}/{len(seeds)}  ({len(examples)} kept)")
        time.sleep(1.5)                        # be gentle on the free-tier rate limit

    # Out-of-domain refusal examples (no teacher call — target is the fallback).
    for q in OUT_OF_DOMAIN:
        ans = pipe.answer(q)
        messages = build_prompt_messages(q, ans.contexts)
        examples.append(to_example(messages, FALLBACK_MESSAGE))

    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * cfg["val_fraction"]))
    val, train = examples[:n_val], examples[n_val:]

    for name, rows in [(cfg["train_file"], train), (cfg["val_file"], val)]:
        path = FT_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows):>4} examples -> {path}")

    print(f"\nDone: {len(train)} train + {len(val)} val "
          f"({sum(1 for e in examples if e['messages'][-1]['content'] == FALLBACK_MESSAGE)} refusal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
