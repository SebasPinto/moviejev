from __future__ import annotations

import csv
import io
import logging
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

from moviejev.pipeline import MAX_PROMPT_CHARS

log = logging.getLogger(__name__)

ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True)
class MLMovie:
    movie_id: int
    title: str
    genres: tuple[str, ...]
    tmdb_id: int | None


@dataclass
class UserSplit:
    user_id: int
    train: list[Rating]
    holdout: list[Rating]  # positives (rating >= threshold), chronologically last


@dataclass
class MovieLens:
    movies: dict[int, MLMovie]
    ratings_by_user: dict[int, list[Rating]]
    rating_counts: Counter[int]

    @classmethod
    def load(cls, root: Path) -> MovieLens:
        links: dict[int, int | None] = {}
        with (root / "links.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = (row.get("tmdbId") or "").strip()
                links[int(row["movieId"])] = int(raw) if raw.isdigit() else None
        movies: dict[int, MLMovie] = {}
        with (root / "movies.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mid = int(row["movieId"])
                genres = tuple(
                    g for g in row["genres"].split("|") if g and g != "(no genres listed)"
                )
                movies[mid] = MLMovie(mid, row["title"].strip(), genres, links.get(mid))
        by_user: dict[int, list[Rating]] = defaultdict(list)
        counts: Counter[int] = Counter()
        with (root / "ratings.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = Rating(
                    int(row["userId"]),
                    int(row["movieId"]),
                    float(row["rating"]),
                    int(row["timestamp"]),
                )
                by_user[r.user_id].append(r)
                counts[r.movie_id] += 1
        for rs in by_user.values():
            rs.sort(key=lambda r: (r.timestamp, r.movie_id))
        return cls(movies, dict(by_user), counts)


def ensure_dataset(data_dir: Path, url: str = ML_SMALL_URL, timeout_s: float = 60.0) -> Path:
    """Download and extract MovieLens once. Returns the folder holding the csv files."""
    target = data_dir / "ml-latest-small"
    if (target / "ratings.csv").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    r = httpx.get(url, timeout=timeout_s, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        wanted = {"ratings.csv", "movies.csv", "links.csv"}
        for member in z.namelist():
            name = Path(member).name
            if name in wanted and not member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                (target / name).write_bytes(z.read(member))
    if not (target / "ratings.csv").exists():
        raise RuntimeError("archive did not contain ratings.csv")
    return target


def split_user(
    ratings: list[Rating], n_holdout: int, min_train: int, positive: float = 4.0
) -> UserSplit | None:
    """Hold out the user's last `n_holdout` positive ratings; train on everything before them."""
    positives = [r for r in ratings if r.rating >= positive]
    if len(positives) < n_holdout:
        return None
    holdout = positives[-n_holdout:]
    cutoff = holdout[0].timestamp
    train = [r for r in ratings if r.timestamp < cutoff]
    if len(train) < min_train:
        return None
    return UserSplit(ratings[0].user_id, train, holdout)


def sample_negatives(
    ml: MovieLens, exclude: set[int], n: int, rng: random.Random, min_ratings: int = 5
) -> list[int]:
    """Popular movies the user never rated, weighted by rating count so they are plausible."""
    pool = [
        m
        for m, c in ml.rating_counts.items()
        if c >= min_ratings and m not in exclude and ml.movies[m].tmdb_id is not None
    ]
    if len(pool) < n:
        raise ValueError("not enough movies for negative sampling")
    weights = [float(ml.rating_counts[m]) for m in pool]
    chosen: list[int] = []
    while len(chosen) < n:
        for m in rng.choices(pool, weights=weights, k=n):
            if m not in chosen:
                chosen.append(m)
                if len(chosen) == n:
                    break
    return chosen


def build_prompt(
    split: UserSplit, movies: dict[int, MLMovie], n_liked: int = 8, n_disliked: int = 4
) -> str:
    """Free-text request in the shape a real user would type, from training ratings only."""
    liked = sorted(
        (r for r in split.train if r.rating >= 4.0), key=lambda r: (-r.rating, -r.timestamp)
    )[:n_liked]
    disliked = sorted((r for r in split.train if r.rating <= 2.0), key=lambda r: r.rating)[
        :n_disliked
    ]
    parts = ["I loved: " + ", ".join(movies[r.movie_id].title for r in liked) + "."]
    if disliked:
        parts.append(
            "I did not enjoy: " + ", ".join(movies[r.movie_id].title for r in disliked) + "."
        )
    parts.append("Recommend movies I would rate highly.")
    return " ".join(parts)[:MAX_PROMPT_CHARS]
