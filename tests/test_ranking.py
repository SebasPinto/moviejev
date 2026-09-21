from __future__ import annotations

import pytest

from moviejev.models import Verdict
from moviejev.reranker.base import expected_value, rank
from tests.conftest import movie


@pytest.mark.parametrize(
    ("probs", "n", "expected"),
    [
        ({"0": 1.0}, 5, 0.0),
        ({"4": 1.0}, 5, 1.0),
        ({"2": 1.0}, 5, 0.5),
        ({"3": 0.5, "4": 0.5}, 5, 0.875),
        ({"1": 0.2, "3": 0.8}, 5, 0.65),
        ({}, 5, 0.0),
        ({"0": 1.0}, 1, 0.0),
    ],
)
def test_expected_value(probs: dict[str, float], n: int, expected: float) -> None:
    assert expected_value(probs, n) == pytest.approx(expected)


def test_expected_value_normalises_unnormalised_distribution() -> None:
    assert expected_value({"0": 2.0, "4": 2.0}, 5) == pytest.approx(0.5)


def test_violation_penalises_score() -> None:
    v = Verdict(movie=movie(1, "A"), expected_fit=0.9, fit_confidence=0.9, violation_prob=0.8)
    assert v.final_score == pytest.approx(0.18)


def test_rank_orders_by_score_then_confidence_then_title() -> None:
    a = Verdict(movie=movie(1, "A"), expected_fit=0.5, fit_confidence=0.2, violation_prob=0.0)
    b = Verdict(movie=movie(2, "B"), expected_fit=0.5, fit_confidence=0.9, violation_prob=0.0)
    c = Verdict(movie=movie(3, "C"), expected_fit=0.9, fit_confidence=0.1, violation_prob=0.0)
    d = Verdict(movie=movie(4, "D"), expected_fit=0.5, fit_confidence=0.9, violation_prob=0.0)
    assert [v.movie.title for v in rank([a, b, c, d])] == ["C", "B", "D", "A"]
