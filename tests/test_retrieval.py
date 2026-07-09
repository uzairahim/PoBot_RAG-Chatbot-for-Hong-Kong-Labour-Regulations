"""Integration tests for retrieval + re-ranking against the real FAISS index.

Skipped automatically if the index hasn't been built yet (run
`python -m src.ingest` first). These load real models, so they are slower.
"""
import pytest

from src.config import INDEX_FILE

pytestmark = pytest.mark.skipif(
    not INDEX_FILE.exists(),
    reason="FAISS index not built — run `python -m src.ingest` first",
)


@pytest.fixture(scope="module")
def store():
    from src.vectorstore import VectorStore
    return VectorStore.load()


def test_index_loads(store):
    assert store.index.ntotal > 0
    assert store.index.d == 384  # multilingual-e5-small dimension


def test_bi_encoder_retrieval_scores_sorted_and_bounded(store):
    from src.retriever import Retriever
    r = Retriever(store=store, use_reranker=False)
    res = r.retrieve("What are the rights of domestic workers in Hong Kong?", top_k=4)
    assert len(res.results) == 4
    scores = [x.cosine_score for x in res.results]
    assert scores == sorted(scores, reverse=True)          # descending
    assert all(-1.0 <= s <= 1.0 for s in scores)           # valid cosine range


def test_reranker_returns_scored_topk(store):
    from src.retriever import Retriever
    r = Retriever(store=store, use_reranker=True)
    res = r.retrieve("What are the rules for recruitment agencies?", top_k=4)
    assert res.reranked is True
    assert len(res.results) == 4
    assert all(x.rerank_score is not None for x in res.results)
    # rerank scores are what we sort by, so they must be descending
    rr = [x.rerank_score for x in res.results]
    assert rr == sorted(rr, reverse=True)


def test_multilingual_tagalog_query_retrieves(store):
    from src.retriever import Retriever
    r = Retriever(store=store, use_reranker=False)
    res = r.retrieve("Ano ang minimum wage para sa domestic helper?", top_k=4)
    assert res.results
    assert res.best_cosine > 0.5  # multilingual embeddings should match well
