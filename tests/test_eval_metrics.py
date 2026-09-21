from __future__ import annotations

import pytest

from moviejev.eval.metrics import hit_rate_at_k, mrr, ndcg_at_k, percentile, precision_at_k


def test_ndcg_perfect_and_worst() -> None:
    assert ndcg_at_k([1, 2, 3, 4], {1, 2}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([3, 4, 1, 2], {1, 2}, 2) == 0.0
    assert ndcg_at_k([], {1}, 10) == 0.0
    assert ndcg_at_k([1], set(), 10) == 0.0


def test_ndcg_partial() -> None:
    # relevant at rank 2 only, one relevant item: DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k([9, 1, 8], {1}, 10) == pytest.approx(0.6309, abs=1e-3)


def test_hit_and_mrr() -> None:
    assert hit_rate_at_k([5, 6, 1], {1}, 3) == 1.0
    assert hit_rate_at_k([5, 6, 1], {1}, 2) == 0.0
    assert mrr([5, 6, 1], {1}) == pytest.approx(1 / 3)
    assert mrr([5, 6], {1}) == 0.0


def test_precision() -> None:
    assert precision_at_k([1, 2, 3, 4], {1, 3}, 4) == 0.5
    assert precision_at_k([1, 2], {1, 3}, 5) == pytest.approx(0.2)
    assert precision_at_k([1], {1}, 0) == 0.0


def test_percentile_nearest_rank() -> None:
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 95) == 5.0
    assert percentile([], 50) == 0.0
