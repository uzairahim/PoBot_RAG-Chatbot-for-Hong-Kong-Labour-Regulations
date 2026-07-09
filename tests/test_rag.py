"""Tests for the RAG pipeline's control flow, using fakes (no API key/network).

Focus: the confidence gate must short-circuit to the fallback WITHOUT calling
the LLM, and successful answers must carry citations.
"""
import pytest

from src.rag import FALLBACK_MESSAGE, RagPipeline
from src.retriever import Retrieved, RetrievalResult


def _chunk(cid="employment_ordinance::1", text="Employees are entitled to rest days.", page=5):
    return {
        "chunk_id": cid, "source_id": "employment_ordinance",
        "title": "A Concise Guide to the Employment Ordinance",
        "publisher": "Hong Kong Labour Department",
        "url": "https://example.gov.hk/eo.pdf", "category": "employment rights",
        "chunk_index": 1, "text": text, "page": page,
    }


class FakeRetriever:
    def __init__(self, result):
        self._result = result

    def retrieve(self, question, top_k=None):
        return self._result


class FakeLLM:
    def __init__(self):
        self.called = False

    def chat(self, messages, temperature=0.1, max_tokens=1024):
        self.called = True
        # Echo that it saw the context, so we can assert it was invoked.
        return "According to the context, employees get rest days [1]."


class ExplodingLLM:
    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called on the fallback path")


def test_fallback_on_low_confidence_even_with_high_cosine():
    # The whole point of switching signals: cosine is high (0.75, like the
    # "capital of France" case) but the calibrated confidence is low, so we must
    # still fall back — and never call the LLM.
    result = RetrievalResult(
        results=[Retrieved(chunk=_chunk(), cosine_score=0.75, rerank_score=-3.6)],
        best_cosine=0.75, reranked=True, confidence=0.03,
    )
    pipe = RagPipeline(retriever=FakeRetriever(result), llm=ExplodingLLM())
    ans = pipe.answer("What is the capital of France?")
    assert ans.used_fallback is True
    assert ans.text == FALLBACK_MESSAGE
    assert ans.citations == []


def test_no_fallback_when_confidence_high():
    result = RetrievalResult(
        results=[Retrieved(chunk=_chunk(), cosine_score=0.85, rerank_score=2.7)],
        best_cosine=0.85, reranked=True, confidence=0.94,
    )
    pipe = RagPipeline(retriever=FakeRetriever(result), llm=FakeLLM())
    ans = pipe.answer("How many rest days?")
    assert ans.used_fallback is False
    assert ans.confidence == 0.94


def test_fallback_when_no_results():
    result = RetrievalResult(results=[], best_cosine=0.0, reranked=False)
    pipe = RagPipeline(retriever=FakeRetriever(result), llm=ExplodingLLM())
    ans = pipe.answer("anything")
    assert ans.used_fallback is True


def test_successful_answer_calls_llm_and_has_citations():
    result = RetrievalResult(
        results=[
            Retrieved(chunk=_chunk("employment_ordinance::1"), cosine_score=0.82, rerank_score=3.1),
            Retrieved(chunk=_chunk("employment_ordinance::2", "Rest days are weekly.", 6),
                      cosine_score=0.79, rerank_score=2.4),
        ],
        best_cosine=0.82, reranked=True,
    )
    result.confidence = 0.93
    llm = FakeLLM()
    pipe = RagPipeline(retriever=FakeRetriever(result), llm=llm)
    ans = pipe.answer("Do employees get rest days?")
    assert llm.called is True
    assert ans.used_fallback is False
    assert len(ans.citations) == 2
    assert ans.citations[0].marker == 1
    assert ans.citations[0].page == 5
    assert ans.citations[1].title.startswith("A Concise Guide")


def test_confidence_gate_boundary(monkeypatch):
    # best_cosine exactly at threshold should NOT fall back (gate is strict <).
    # settings is a frozen dataclass, so patch the whole object with a copy.
    import dataclasses

    from src import rag
    monkeypatch.setattr(rag, "settings", dataclasses.replace(rag.settings, min_score=0.30))
    result = RetrievalResult(
        results=[Retrieved(chunk=_chunk(), cosine_score=0.30)],
        best_cosine=0.30, reranked=False,
    )
    pipe = RagPipeline(retriever=FakeRetriever(result), llm=FakeLLM())
    ans = pipe.answer("q")
    assert ans.used_fallback is False
