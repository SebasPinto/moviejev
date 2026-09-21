from __future__ import annotations

import json

import pytest

from moviejev.llm.base import LLM, LLMError
from moviejev.models import TasteProfile
from moviejev.reranker.llm_judge import LLMJudgeReranker
from tests.conftest import movie


class _FlakyLLM(LLM):
    """Fails for one title, answers the rubric for the rest."""

    def __init__(self, bad_title: str) -> None:
        super().__init__()
        self._bad = bad_title

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if f"Title: {self._bad}" in user:
            return "not json at all"
        return json.dumps({"fit_level": 4, "reason": "tone", "violates": False})


async def test_per_candidate_failure_drops_only_that_movie(profile: TasteProfile) -> None:
    rr = LLMJudgeReranker(_FlakyLLM("Bad"))
    out = await rr.judge(profile, [movie(1, "Good"), movie(2, "Bad")])
    assert [v.movie.title for v in out] == ["Good"]
    assert out[0].expected_fit == 1.0 and out[0].reason == "tone"


async def test_all_failures_raise(profile: TasteProfile) -> None:
    rr = LLMJudgeReranker(_FlakyLLM("Only"))
    with pytest.raises(LLMError):
        await rr.judge(profile, [movie(1, "Only")])
