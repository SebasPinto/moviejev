from __future__ import annotations

import pytest

from moviejev.pipeline import MAX_PROMPT_CHARS, Pipeline
from moviejev.reranker import LLMJudgeReranker, PassthroughReranker
from tests.conftest import FakeCatalog, FakeLLM, movie


def _pipeline(reranker: object | None = None, n: int = 5, k: int = 3) -> tuple[Pipeline, FakeLLM]:
    llm = FakeLLM(
        [
            {"title": "Contact", "year": 1997},
            {"title": "contact", "year": 1997},  # duplicate, different case
            {"title": "Made Up Movie", "year": 2010},  # not in catalog
            {"title": "Moon", "year": 2009},
            {"title": "Primer", "year": 2004},
        ]
    )
    catalog = FakeCatalog(
        {"contact": movie(1, "Contact"), "moon": movie(2, "Moon"), "primer": movie(3, "Primer")}
    )
    rr = reranker if reranker is not None else PassthroughReranker()
    return Pipeline(llm, catalog, rr, n_candidates=n, top_k=k), llm  # type: ignore[arg-type]


async def test_end_to_end_drops_hallucinated_and_dedups() -> None:
    p, _ = _pipeline()
    res = await p.run("something like Arrival, no horror")
    assert res.dropped_unverified == ["Made Up Movie"]
    assert [r.movie.title for r in res.recommendations] == ["Contact", "Moon", "Primer"]
    assert all(r.explanation for r in res.recommendations)
    assert res.reranker == "none"


async def test_llm_judge_reranker_uses_shared_rubric() -> None:
    llm = FakeLLM([{"title": "Contact"}])
    p = Pipeline(llm, FakeCatalog({"contact": movie(1, "Contact")}), LLMJudgeReranker(llm), 5, 3)
    res = await p.run("sci-fi")
    assert res.reranker == "llm_judge"
    assert res.recommendations[0].score == pytest.approx(0.75)  # level 3 of 0..4


async def test_prompt_validation() -> None:
    p, _ = _pipeline()
    with pytest.raises(ValueError, match="empty"):
        await p.run("   ")
    with pytest.raises(ValueError, match="too long"):
        await p.run("x" * (MAX_PROMPT_CHARS + 1))


async def test_top_k_and_candidate_cap() -> None:
    p, _ = _pipeline(n=2, k=1)
    res = await p.run("sci-fi")
    assert len(res.recommendations) == 1
