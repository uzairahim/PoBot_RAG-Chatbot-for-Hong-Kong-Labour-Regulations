"""Unit tests for the reranker's score->confidence mapping (pure, no model)."""
from src.reranker import Reranker

s2c = Reranker.score_to_confidence


def test_zero_logit_is_half():
    assert s2c(0.0) == 0.5


def test_irrelevant_and_relevant_separate_cleanly():
    # Values observed empirically: ~-3.6 (out-of-domain) vs ~+2.7 (real answer).
    assert s2c(-3.6) < 0.10
    assert s2c(2.7) > 0.90


def test_monotonic():
    assert s2c(-2) < s2c(-1) < s2c(0) < s2c(1) < s2c(2)


def test_numerically_stable_at_extremes():
    # Must not overflow for large-magnitude logits.
    assert 0.0 <= s2c(-1000) < 1e-6
    assert 1.0 - 1e-6 < s2c(1000) <= 1.0