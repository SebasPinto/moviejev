from __future__ import annotations

from pathlib import Path

from moviejev.eval.cache import CachingLLM, JsonCache
from tests.conftest import FakeLLM


async def test_cache_hits_skip_inner_and_persist(tmp_path: Path) -> None:
    inner = FakeLLM([{"title": "Contact"}])
    llm = CachingLLM(inner, JsonCache(tmp_path / "c.json"))
    system = "You are a strict movie-recommendation judge."
    a = await llm.complete(system, "u", 10)
    b = await llm.complete(system, "u", 10)
    assert a == b and len(inner.calls) == 1
    assert (llm.hits, llm.misses) == (1, 1)

    again = CachingLLM(FakeLLM([]), JsonCache(tmp_path / "c.json"))
    assert await again.complete(system, "u", 10) == a
    assert again.hits == 1


def test_cache_key_depends_on_all_inputs() -> None:
    k = CachingLLM.key
    assert k("s", "u", 1) != k("s", "u", 2) != k("s", "x", 2) != k("t", "x", 2)
