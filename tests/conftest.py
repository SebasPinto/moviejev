from __future__ import annotations

import json

import pytest

from moviejev.catalog.base import Catalog
from moviejev.llm.base import LLM
from moviejev.models import Candidate, Movie, TasteProfile


class FakeLLM(LLM):
    """Scripted LLM: returns canned JSON keyed by which system prompt is used."""

    def __init__(self, candidates: list[dict[str, object]]) -> None:
        super().__init__()
        self.calls: list[str] = []
        self._cands = candidates

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.calls.append(system)
        if "taste profile" in system:
            return json.dumps(
                {"liked_titles": ["Arrival"], "genres": ["Science Fiction"], "themes": ["language"],
                 "tone": "quiet", "exclusions": ["horror"], "era": ""}
            )  # fmt: skip
        if "propose real" in system:
            return "```json\n" + json.dumps({"candidates": self._cands}) + "\n```"
        if "explanations" in system:
            titles = [
                line.split(" | ")[0]
                for line in user.split("Films (title | main match dimension):\n")[1]
                .split("\n\n")[0]
                .split("\n")
            ]
            return json.dumps({"explanations": {t: f"because {t}" for t in titles}})
        if "judge" in system:
            return json.dumps({"fit_level": 3, "reason": "themes", "violates": False})
        raise AssertionError(f"unexpected system prompt: {system}")


class FakeCatalog(Catalog):
    def __init__(self, known: dict[str, Movie]) -> None:
        self._known = known

    async def resolve(self, candidate: Candidate) -> Movie | None:
        return self._known.get(candidate.title.casefold())


def movie(tmdb_id: int, title: str, genres: list[str] | None = None) -> Movie:
    return Movie(tmdb_id=tmdb_id, title=title, year=2000, overview="x", genres=genres or [])


@pytest.fixture
def profile() -> TasteProfile:
    return TasteProfile(liked_titles=["Arrival"], genres=["Science Fiction"], exclusions=["horror"])
