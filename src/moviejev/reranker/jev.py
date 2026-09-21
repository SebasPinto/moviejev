from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from moviejev.models import Movie, TasteProfile, Verdict
from moviejev.reranker.base import (
    FIT_LEVELS,
    REASON_CHOICES,
    VIOLATION_INSTRUCTION,
    Reranker,
    expected_value,
)

log = logging.getLogger(__name__)


class _ScoreAnswer(BaseModel):
    score: float
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class _ChoiceAnswer(BaseModel):
    choice: str
    confidence: float = Field(ge=0, le=1)


class _NoulAnswer(BaseModel):
    noul: float = Field(ge=0, le=1)


class _JevResponse(BaseModel):
    model: str = ""
    answers: dict[str, dict[str, Any]]


class JevError(RuntimeError):
    pass


class JevReranker(Reranker):
    """One System One call per candidate, in parallel. State = profile + movie card only."""

    name = "jev"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float,
        min_confidence: float,
        concurrency: int = 8,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout_s,
        )
        self._model = model
        self._min_conf = min_confidence
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def build_state(profile: TasteProfile, movie: Movie) -> str:
        return f"## Viewer profile\n{profile.as_state()}\n\n## Candidate movie\n{movie.as_state()}"

    @staticmethod
    def questions() -> dict[str, Any]:
        return {
            "fit": {
                "type": "score",
                "instructions": "How well the candidate movie matches the viewer profile",
                "criteria": FIT_LEVELS,
            },
            "reason": {
                "type": "choice",
                "instructions": "The main dimension on which the candidate matches the profile",
                "criteria": REASON_CHOICES,
            },
            "violates": {"type": "noul", "instructions": VIOLATION_INSTRUCTION},
        }

    async def _judge_one(self, profile: TasteProfile, movie: Movie) -> Verdict:
        body = {
            "state": self.build_state(profile, movie),
            "model": self._model,
            "questions": self.questions(),
        }
        async with self._sem:
            r = await self._client.post("/v1/systemone", json=body)
        if r.status_code != 200:
            raise JevError(f"jev {r.status_code}: {r.text[:200]}")
        try:
            parsed = _JevResponse.model_validate(r.json())
            fit = _ScoreAnswer.model_validate(parsed.answers["fit"])
            reason = _ChoiceAnswer.model_validate(parsed.answers["reason"])
            viol = _NoulAnswer.model_validate(parsed.answers["violates"])
        except (ValidationError, KeyError, ValueError) as e:
            raise JevError(f"unexpected jev payload: {e}") from e
        return Verdict(
            movie=movie,
            expected_fit=expected_value(fit.probabilities, len(FIT_LEVELS)),
            fit_confidence=fit.confidence,
            violation_prob=viol.noul,
            reason=reason.choice,
            low_confidence=fit.confidence < self._min_conf,
        )

    async def judge(self, profile: TasteProfile, movies: list[Movie]) -> list[Verdict]:
        results = await asyncio.gather(
            *(self._judge_one(profile, m) for m in movies), return_exceptions=True
        )
        verdicts: list[Verdict] = []
        for m, res in zip(movies, results, strict=True):
            if isinstance(res, BaseException):
                log.warning("jev failed for %r, dropping: %s", m.title, res)
                continue
            verdicts.append(res)
        if movies and not verdicts:
            raise JevError("all jev calls failed")
        return verdicts
