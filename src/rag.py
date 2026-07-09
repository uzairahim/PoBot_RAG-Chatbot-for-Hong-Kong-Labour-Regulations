"""The RAG pipeline: question -> retrieve -> ground -> generate -> cited answer.

Ties retrieval, prompting, and the LLM together, and implements two of the
bonus features: inline source citations and a confidence-based fallback for
questions the corpus can't answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import settings
from src.llm import LLMProvider, get_llm
from src.retriever import Retrieved, Retriever

SYSTEM_PROMPT = """\
You are PoBot, an assistant that answers questions about Hong Kong labour and \
employment regulations for migrant workers (including foreign domestic helpers).

Rules:
- Answer ONLY using the numbered CONTEXT passages provided. Do not use outside knowledge.
- Cite the passages you rely on inline using their bracket numbers, e.g. [1], [2].
- If the context does not contain enough information to answer, say so plainly and \
suggest contacting the Hong Kong Labour Department. Do not guess.
- Reply in the SAME language as the user's question (e.g. English or Tagalog).
- Be clear, concise, and practical. Use plain language a worker can understand.
- End with a one-line reminder that this is general information, not legal advice.
"""

# Returned when retrieval confidence is below threshold — we never call the LLM,
# so it can never hallucinate an answer the corpus doesn't support.
FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information in my Hong Kong labour "
    "regulation sources to answer that confidently. For accurate guidance, "
    "please contact the Hong Kong Labour Department (labour.gov.hk) or call "
    "their hotline at 2717 1771.\n\n(This is general information, not legal advice.)"
)


@dataclass
class Citation:
    marker: int          # the [n] used in the answer/context
    title: str
    publisher: str
    url: str
    page: int | None


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    used_fallback: bool = False
    confidence: float | None = None   # calibrated (0,1); None if reranker off
    best_cosine: float = 0.0
    reranked: bool = False
    contexts: list[Retrieved] = field(default_factory=list)


def _build_context_block(results: list[Retrieved]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    blocks = []
    for i, r in enumerate(results, start=1):
        c = r.chunk
        loc = f", p.{c['page']}" if c.get("page") else ""
        blocks.append(f"[{i}] (Source: {c['title']}{loc})\n{c['text']}")
    return "\n\n".join(blocks)


def _citations_from(results: list[Retrieved]) -> list[Citation]:
    cites = []
    for i, r in enumerate(results, start=1):
        c = r.chunk
        cites.append(Citation(marker=i, title=c["title"], publisher=c["publisher"],
                              url=c["url"], page=c.get("page")))
    return cites


class RagPipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLMProvider | None = None):
        self.retriever = retriever or Retriever()
        # Defer LLM construction until first real query so retrieval-only use
        # (and the fallback path) works without an API key configured.
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @staticmethod
    def _is_low_confidence(retrieval: RetrievalResult) -> bool:
        """Decide whether to fall back instead of answering.

        Preferred signal is the re-ranker's calibrated confidence (well
        separated). Only when the re-ranker is off do we fall back to the weaker
        bi-encoder cosine gate.
        """
        if not retrieval.results:
            return True
        if retrieval.confidence is not None:
            return retrieval.confidence < settings.min_confidence
        return retrieval.best_cosine < settings.min_score

    def answer(self, question: str) -> Answer:
        retrieval = self.retriever.retrieve(question)

        # Confidence gate: if the best context is only weakly relevant, bail out
        # with the fallback instead of risking a hallucinated answer. We never
        # call the LLM here, so it cannot invent an unsupported answer.
        if self._is_low_confidence(retrieval):
            return Answer(text=FALLBACK_MESSAGE, used_fallback=True,
                          confidence=retrieval.confidence, best_cosine=retrieval.best_cosine,
                          reranked=retrieval.reranked, contexts=retrieval.results)

        context_block = _build_context_block(retrieval.results)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {question}\n\nAnswer:"},
        ]
        text = self.llm.chat(messages)
        return Answer(text=text, citations=_citations_from(retrieval.results),
                      used_fallback=False, confidence=retrieval.confidence,
                      best_cosine=retrieval.best_cosine, reranked=retrieval.reranked,
                      contexts=retrieval.results)
