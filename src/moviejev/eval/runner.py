from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moviejev.catalog.tmdb import TMDBCatalog
from moviejev.eval.dataset import (
    MovieLens,
    build_prompt,
    ensure_dataset,
    sample_negatives,
    split_user,
)
from moviejev.eval.metrics import hit_rate_at_k, mean, mrr, ndcg_at_k, percentile, precision_at_k
from moviejev.llm.base import LLM, LLMError, TokenUsage
from moviejev.models import Movie, TasteProfile
from moviejev.pipeline import Pipeline
from moviejev.reranker.base import Reranker, rank
from moviejev.reranker.jev import JevError
from moviejev.reranker.passthrough import PassthroughReranker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    users: int
    holdout: int = 5
    negatives: int = 15
    k: int = 10
    seed: int = 42
    min_train: int = 15
    rerankers: tuple[str, ...] = ("none", "llm_judge", "jev")
    data_dir: Path = Path("data")
    llm_price_in: float = 3.0  # USD per 1M tokens
    llm_price_out: float = 15.0
    jev_price_in: float = 0.042
    jev_price_out: float = 0.0


@dataclass
class RerankerStats:
    name: str
    ndcg: list[float] = field(default_factory=list)
    hit: list[float] = field(default_factory=list)
    hit1: list[float] = field(default_factory=list)
    prec5: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    latency_s: list[float] = field(default_factory=list)
    low_confidence: int = 0
    verdicts: int = 0
    errors: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)

    def summary(self, k: int, price_in: float, price_out: float) -> dict[str, Any]:
        cost = (self.usage.input_tokens * price_in + self.usage.output_tokens * price_out) / 1e6
        return {
            "reranker": self.name,
            "users": len(self.ndcg),
            f"ndcg@{k}": round(mean(self.ndcg), 4),
            f"hit_rate@{k}": round(mean(self.hit), 4),
            "hit@1": round(mean(self.hit1), 4),
            "precision@5": round(mean(self.prec5), 4),
            "mrr": round(mean(self.mrr), 4),
            "low_confidence_rate": round(self.low_confidence / self.verdicts, 4)
            if self.verdicts
            else 0.0,
            "errors": self.errors,
            "calls": self.usage.calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cost_usd": round(cost, 4),
            "latency_p50_s": round(percentile(self.latency_s, 50), 3),
            "latency_p95_s": round(percentile(self.latency_s, 95), 3),
        }


class MovieCards:
    """TMDB movie cards by id, cached on disk. The eval needs overviews MovieLens lacks."""

    def __init__(self, path: Path, catalog: TMDBCatalog) -> None:
        self._path = path
        self._catalog = catalog
        self._cards: dict[int, Movie | None] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                self._cards[int(k)] = Movie.model_validate(v) if v is not None else None

    async def get(self, tmdb_id: int) -> Movie | None:
        if tmdb_id not in self._cards:
            self._cards[tmdb_id] = await self._catalog.fetch(tmdb_id)
            self._flush()
        return self._cards[tmdb_id]

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(k): (v.model_dump() if v else None) for k, v in self._cards.items()}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._path)


def _usage_delta(before: TokenUsage, after: TokenUsage) -> TokenUsage:
    return TokenUsage(
        calls=after.calls - before.calls,
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
    )


def _snapshot(u: TokenUsage) -> TokenUsage:
    return TokenUsage(u.calls, u.input_tokens, u.output_tokens)


async def run_eval(
    cfg: EvalConfig,
    llm: LLM,
    catalog: TMDBCatalog,
    make_reranker: Callable[[str], Reranker],
) -> dict[str, Any]:
    """Per user: hold out last positives, mix with popular negatives, let each reranker
    order the same candidate set against the same profile. Returns a JSON-serialisable report."""
    root = ensure_dataset(cfg.data_dir)
    ml = MovieLens.load(root)
    cards = MovieCards(cfg.data_dir / "eval" / "tmdb_cards.json", catalog)
    profiler = Pipeline(llm, catalog, PassthroughReranker(), n_candidates=5, top_k=1)
    rerankers = {name: make_reranker(name) for name in cfg.rerankers}
    stats = {name: RerankerStats(name) for name in cfg.rerankers}
    rng = random.Random(cfg.seed)  # noqa: S311 - reproducible sampling, not security

    user_ids = sorted(ml.ratings_by_user)
    rng.shuffle(user_ids)
    per_user: list[dict[str, Any]] = []
    profile_before = _snapshot(llm.usage)

    for uid in user_ids:
        if len(per_user) >= cfg.users:
            break
        split = split_user(ml.ratings_by_user[uid], cfg.holdout, cfg.min_train)
        if split is None:
            continue
        rated = {r.movie_id for r in ml.ratings_by_user[uid]}
        neg_ids = sample_negatives(ml, rated, cfg.negatives, rng)
        pos_ids = [r.movie_id for r in split.holdout if ml.movies[r.movie_id].tmdb_id is not None]

        movies: list[Movie] = []
        relevant: set[int] = set()
        for mid in pos_ids + neg_ids:
            tmdb_id = ml.movies[mid].tmdb_id
            assert tmdb_id is not None
            card = await cards.get(tmdb_id)
            if card is None:
                continue
            movies.append(card)
            if mid in pos_ids:
                relevant.add(card.tmdb_id)
        if not relevant or len(movies) < cfg.holdout + cfg.negatives // 2:
            log.warning("user %d: too few resolvable candidates, skipping", uid)
            continue
        rng.shuffle(movies)

        prompt = build_prompt(split, ml.movies)
        try:
            profile: TasteProfile = await profiler.profile(prompt)
        except LLMError as e:
            log.warning("user %d: profile failed: %s", uid, e)
            continue

        row: dict[str, Any] = {"user_id": uid, "relevant": sorted(relevant), "prompt": prompt}
        for name, rr in rerankers.items():
            st = stats[name]
            before = _snapshot(llm.usage)
            t0 = time.perf_counter()
            try:
                verdicts = rank(await rr.judge(profile, movies))
            except (JevError, LLMError) as e:
                st.errors += 1
                log.warning("user %d: %s failed: %s", uid, name, e)
                verdicts = []
            elapsed = time.perf_counter() - t0
            if name == "llm_judge":
                d = _usage_delta(before, llm.usage)
                st.usage.calls += d.calls
                st.usage.input_tokens += d.input_tokens
                st.usage.output_tokens += d.output_tokens
            if not verdicts:
                continue
            st.latency_s.append(elapsed)
            ranked = [v.movie.tmdb_id for v in verdicts]
            st.ndcg.append(ndcg_at_k(ranked, relevant, cfg.k))
            st.hit.append(hit_rate_at_k(ranked, relevant, cfg.k))
            st.hit1.append(hit_rate_at_k(ranked, relevant, 1))
            st.prec5.append(precision_at_k(ranked, relevant, 5))
            st.mrr.append(mrr(ranked, relevant))
            st.low_confidence += sum(v.low_confidence for v in verdicts)
            st.verdicts += len(verdicts)
            row[name] = ranked[: cfg.k]
        per_user.append(row)
        log.info("user %d done (%d/%d)", uid, len(per_user), cfg.users)

    jev = rerankers.get("jev")
    if jev is not None and hasattr(jev, "usage"):
        stats["jev"].usage = jev.usage
    profile_usage = _usage_delta(profile_before, llm.usage)
    # llm_judge tokens are counted separately above; profile usage is what remains.
    judge_calls = stats["llm_judge"].usage.calls if "llm_judge" in stats else 0
    judge_in = stats["llm_judge"].usage.input_tokens if "llm_judge" in stats else 0
    judge_out = stats["llm_judge"].usage.output_tokens if "llm_judge" in stats else 0
    profile_only = TokenUsage(
        profile_usage.calls - judge_calls,
        profile_usage.input_tokens - judge_in,
        profile_usage.output_tokens - judge_out,
    )

    summaries = []
    for name, st in stats.items():
        if name == "jev":
            summaries.append(st.summary(cfg.k, cfg.jev_price_in, cfg.jev_price_out))
        else:
            summaries.append(st.summary(cfg.k, cfg.llm_price_in, cfg.llm_price_out))
    return {
        "config": {**cfg.__dict__, "data_dir": str(cfg.data_dir)},
        "users_evaluated": len(per_user),
        "profile_llm": {
            "calls": profile_only.calls,
            "input_tokens": profile_only.input_tokens,
            "output_tokens": profile_only.output_tokens,
            "cost_usd": round(
                (
                    profile_only.input_tokens * cfg.llm_price_in
                    + profile_only.output_tokens * cfg.llm_price_out
                )
                / 1e6,
                4,
            ),
        },
        "rerankers": summaries,
        "per_user": per_user,
    }


def markdown_table(report: dict[str, Any]) -> str:
    rows = report["rerankers"]
    if not rows:
        return "(no results)"
    cols = [c for c in rows[0] if c != "reranker"]
    head = "| reranker | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = [
        "| " + str(r["reranker"]) + " | " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows
    ]
    return "\n".join([head, sep, *body])
