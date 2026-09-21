from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from moviejev.llm.base import LLM
from moviejev.models import Movie, TasteProfile, Verdict
from moviejev.reranker.base import FIT_LEVELS, REASON_CHOICES, VIOLATION_INSTRUCTION, Reranker

_SYSTEM = (
    "You are a strict movie-recommendation judge. Reply with a single JSON object and nothing else."
)


class _Judgement(BaseModel):
    fit_level: int = Field(ge=0, le=len(FIT_LEVELS) - 1)
    reason: str
    violates: bool


class LLMJudgeReranker(Reranker):
    """Same rubric as Jev, answered by a text LLM, so the two can be compared on equal terms."""

    name = "llm_judge"

    def __init__(self, llm: LLM, concurrency: int = 4) -> None:
        self._llm = llm
        self._sem = asyncio.Semaphore(concurrency)

    async def _judge_one(self, profile: TasteProfile, movie: Movie) -> Verdict:
        levels = "\n".join(f"{i}: {t}" for i, t in enumerate(FIT_LEVELS))
        reasons = "\n".join(f"{k}: {v}" for k, v in REASON_CHOICES.items())
        user = (
            f"## Viewer profile\n{profile.as_state()}\n\n## Candidate movie\n{movie.as_state()}\n\n"
            f"Rate fit on this scale:\n{levels}\n\nMain match dimension (one key):\n{reasons}\n\n"
            f"violates: true if '{VIOLATION_INSTRUCTION}'.\n\n"
            'Return: {"fit_level": int, "reason": string, "violates": bool}'
        )
        async with self._sem:
            j = await self._llm.complete_json(_SYSTEM, user, _Judgement, max_tokens=200)
        reason = j.reason if j.reason in REASON_CHOICES else "themes"
        return Verdict(
            movie=movie,
            expected_fit=j.fit_level / (len(FIT_LEVELS) - 1),
            fit_confidence=1.0,  # a text LLM gives no calibrated confidence; this is the point
            violation_prob=1.0 if j.violates else 0.0,
            reason=reason,
        )

    async def judge(self, profile: TasteProfile, movies: list[Movie]) -> list[Verdict]:
        return list(await asyncio.gather(*(self._judge_one(profile, m) for m in movies)))
