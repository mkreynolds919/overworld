import math

import pytest

from overworld.evaluation.metrics import (
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RANKED = [10, 20, 30, 40, 50]
RELEVANT = {20, 50, 99}


def test_hit_rate():
    assert hit_rate_at_k(RANKED, RELEVANT, k=5) == 1.0
    assert hit_rate_at_k(RANKED, RELEVANT, k=1) == 0.0


def test_precision_divides_by_k():
    assert precision_at_k(RANKED, RELEVANT, k=5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, RELEVANT, k=10) == pytest.approx(2 / 10)


def test_recall():
    assert recall_at_k(RANKED, RELEVANT, k=5) == pytest.approx(2 / 3)


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, RELEVANT, k=5) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2, 3, 4], {1, 2}, k=4) == pytest.approx(1.0)


def test_ndcg_rewards_higher_ranks():
    assert ndcg_at_k([1, 9, 9, 9], {1}, k=4) > ndcg_at_k([9, 9, 9, 1], {1}, k=4)


def test_ndcg_ideal_capped_at_k():
    relevant = set(range(100))
    assert ndcg_at_k(list(range(10)), relevant, k=10) == pytest.approx(1.0)


def test_empty_relevant_raises():
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, set())


def test_coverage():
    lists = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
    assert catalog_coverage(lists, catalog_size=12, k=3) == pytest.approx(6 / 12)
