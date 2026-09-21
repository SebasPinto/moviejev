from __future__ import annotations

import random
from collections import Counter

from moviejev.eval.dataset import (
    MLMovie,
    MovieLens,
    Rating,
    build_prompt,
    sample_negatives,
    split_user,
)
from moviejev.pipeline import MAX_PROMPT_CHARS


def _ratings(user: int, specs: list[tuple[int, float]]) -> list[Rating]:
    return [Rating(user, mid, r, ts) for ts, (mid, r) in enumerate(specs, start=1)]


def test_split_holds_out_last_positives_and_trains_before_cutoff() -> None:
    rs = _ratings(1, [(1, 5), (2, 2), (3, 4), (4, 1), (5, 4), (6, 5), (7, 3), (8, 4)])
    split = split_user(rs, n_holdout=3, min_train=2)
    assert split is not None
    assert [r.movie_id for r in split.holdout] == [5, 6, 8]
    assert [r.movie_id for r in split.train] == [1, 2, 3, 4]  # nothing at/after ts of movie 5


def test_split_rejects_users_with_too_little_data() -> None:
    assert split_user(_ratings(1, [(1, 5), (2, 5)]), n_holdout=3, min_train=1) is None
    assert split_user(_ratings(1, [(1, 5), (2, 5), (3, 5)]), n_holdout=2, min_train=5) is None


def _ml() -> MovieLens:
    movies = {
        i: MLMovie(i, f"M{i} (2000)", ("Drama",), 100 + i if i != 3 else None) for i in range(1, 8)
    }
    counts = Counter({1: 50, 2: 40, 3: 30, 4: 20, 5: 10, 6: 2, 7: 9})
    return MovieLens(movies, {}, counts)


def test_negatives_exclude_rated_unlinked_and_rare() -> None:
    ml = _ml()
    out = sample_negatives(ml, exclude={1}, n=3, rng=random.Random(0), min_ratings=5)  # noqa: S311
    assert len(out) == len(set(out)) == 3
    assert 1 not in out  # rated
    assert 3 not in out  # no tmdb id
    assert 6 not in out  # too few ratings


def test_negatives_are_deterministic_for_a_seed() -> None:
    ml = _ml()
    a = sample_negatives(ml, set(), 3, random.Random(7))  # noqa: S311
    b = sample_negatives(ml, set(), 3, random.Random(7))  # noqa: S311
    assert a == b


def test_prompt_mentions_liked_and_disliked_and_is_bounded() -> None:
    movies = {i: MLMovie(i, f"Movie {i}", (), i) for i in range(1, 40)}
    rs = _ratings(1, [(i, 5.0 if i % 2 else 1.0) for i in range(1, 30)] + [(30, 5.0), (31, 5.0)])
    split = split_user(rs, n_holdout=2, min_train=5)
    assert split is not None
    p = build_prompt(split, movies)
    assert "I loved:" in p and "I did not enjoy:" in p
    assert "Movie 30" not in p and "Movie 31" not in p  # holdout never leaks into the prompt
    assert len(p) <= MAX_PROMPT_CHARS
