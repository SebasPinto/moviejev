from __future__ import annotations

from moviejev.models import Movie, TasteProfile, Verdict
from moviejev.reranker.base import Reranker


class PassthroughReranker(Reranker):
    """Baseline: keep the LLM's own order. Needed to measure what the reranker adds."""

    name = "none"

    async def judge(self, profile: TasteProfile, movies: list[Movie]) -> list[Verdict]:
        n = max(len(movies), 1)
        return [
            Verdict(
                movie=m,
                expected_fit=1.0 - i / n,
                fit_confidence=1.0,
                violation_prob=0.0,
                reason="llm_order",
            )
            for i, m in enumerate(movies)
        ]
