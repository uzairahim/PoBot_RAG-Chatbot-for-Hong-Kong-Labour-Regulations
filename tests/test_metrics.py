"""Unit tests for the pure retrieval-metric functions."""
import math

from src.evaluate import dcg_at_k, hit_at_k, ndcg_at_k, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k([0, 0, 1, 0], 4) == 1.0
    assert hit_at_k([0, 0, 1, 0], 2) == 0.0   # relevant is beyond k=2
    assert hit_at_k([0, 0, 0], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([1, 0, 0]) == 1.0
    assert reciprocal_rank([0, 1, 0]) == 0.5
    assert reciprocal_rank([0, 0, 1]) == 1 / 3
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_ndcg_perfect_and_zero():
    # A single relevant item already at the top is perfect ordering.
    assert ndcg_at_k([1, 0, 0, 0], 4) == 1.0
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_rewards_higher_rank():
    top = ndcg_at_k([1, 0, 0, 0], 4)
    lower = ndcg_at_k([0, 0, 0, 1], 4)
    assert top > lower
    # a relevant item at rank 4 contributes 1/log2(5)
    assert math.isclose(lower, (1 / math.log2(5)) / 1.0, rel_tol=1e-9)


def test_dcg_matches_formula():
    assert math.isclose(dcg_at_k([1, 1], 2), 1 / math.log2(2) + 1 / math.log2(3), rel_tol=1e-9)