from __future__ import annotations

import httpx
import pytest
import respx

from moviejev.models import TasteProfile
from moviejev.reranker.jev import JevError, JevReranker
from tests.conftest import movie

BASE = "https://jev.test"


def _answers(
    fit_probs: dict[str, float], conf: float, viol: float, reason: str = "themes"
) -> dict[str, object]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "fit": {"type": "score", "score": 3.0, "confidence": conf, "probabilities": fit_probs},
            "reason": {"type": "choice", "choice": reason, "confidence": 0.8, "probabilities": {}},
            "violates": {"type": "noul", "noul": viol},
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


@pytest.fixture
def jev() -> JevReranker:
    return JevReranker("k", BASE, "jev-latest", timeout_s=5, min_confidence=0.35)


def test_state_is_minimal_and_contains_both_sides(profile: TasteProfile) -> None:
    s = JevReranker.build_state(profile, movie(1, "Contact", ["Drama"]))
    assert "Liked: Arrival" in s and "Must avoid: horror" in s
    assert "Title: Contact (2000)" in s and "Genres: Drama" in s


def test_questions_have_the_three_primitives() -> None:
    q = JevReranker.questions()
    assert {q["fit"]["type"], q["reason"]["type"], q["violates"]["type"]} == {
        "score",
        "choice",
        "noul",
    }
    assert len(q["fit"]["criteria"]) == 5


@respx.mock
async def test_judge_parses_and_flags_low_confidence(
    jev: JevReranker, profile: TasteProfile
) -> None:
    route = respx.post(f"{BASE}/v1/systemone").mock(
        side_effect=[
            httpx.Response(200, json=_answers({"3": 0.5, "4": 0.5}, conf=0.9, viol=0.0)),
            httpx.Response(200, json=_answers({"1": 1.0}, conf=0.2, viol=0.9, reason="genre")),
        ]
    )
    out = await jev.judge(profile, [movie(1, "Contact"), movie(2, "Saw")])
    assert route.call_count == 2
    assert out[0].expected_fit == pytest.approx(0.875) and not out[0].low_confidence
    assert out[1].expected_fit == pytest.approx(0.25) and out[1].low_confidence
    assert out[1].violation_prob == 0.9 and out[1].reason == "genre"
    assert out[1].final_score == pytest.approx(0.025)
    assert jev.usage.calls == 2 and jev.usage.input_tokens == 200


@respx.mock
async def test_partial_failures_are_dropped_not_fatal(
    jev: JevReranker, profile: TasteProfile
) -> None:
    respx.post(f"{BASE}/v1/systemone").mock(
        side_effect=[
            httpx.Response(200, json=_answers({"4": 1.0}, conf=1.0, viol=0.0)),
            httpx.Response(500, text="boom"),
        ]
    )
    out = await jev.judge(profile, [movie(1, "A"), movie(2, "B")])
    assert [v.movie.title for v in out] == ["A"]


@respx.mock
async def test_all_failures_raise(jev: JevReranker, profile: TasteProfile) -> None:
    respx.post(f"{BASE}/v1/systemone").mock(return_value=httpx.Response(401, text="nope"))
    with pytest.raises(JevError):
        await jev.judge(profile, [movie(1, "A")])


@respx.mock
async def test_malformed_payload_raises(jev: JevReranker, profile: TasteProfile) -> None:
    respx.post(f"{BASE}/v1/systemone").mock(return_value=httpx.Response(200, json={"answers": {}}))
    with pytest.raises(JevError):
        await jev.judge(profile, [movie(1, "A")])


def test_request_body_shape(jev: JevReranker, profile: TasteProfile) -> None:
    body = {
        "state": JevReranker.build_state(profile, movie(1, "A")),
        "model": "jev-latest",
        "questions": JevReranker.questions(),
    }
    assert set(body) == {"state", "model", "questions"}
