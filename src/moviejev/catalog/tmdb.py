from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel, Field, ValidationError

from moviejev.catalog.base import Catalog
from moviejev.models import Candidate, Movie

log = logging.getLogger(__name__)

# TMDB genre ids -> names (stable; avoids an extra request per call).
_GENRES: dict[int, str] = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}  # fmt: skip


class _TMDBResult(BaseModel):
    id: int
    title: str = Field(max_length=300)
    release_date: str = ""
    overview: str = ""
    genre_ids: list[int] = Field(default_factory=list)
    popularity: float = 0.0


class _TMDBSearch(BaseModel):
    results: list[_TMDBResult] = Field(default_factory=list)


class _TMDBGenre(BaseModel):
    id: int


class _TMDBDetail(BaseModel):
    id: int
    title: str = Field(max_length=300)
    release_date: str = ""
    overview: str = ""
    genres: list[_TMDBGenre] = Field(default_factory=list)
    popularity: float = 0.0


def _year(release_date: str) -> int | None:
    head = release_date[:4]
    return int(head) if len(head) == 4 and head.isdigit() else None


class TMDBCatalog(Catalog):
    def __init__(self, api_key: str, base_url: str, timeout_s: float, concurrency: int = 6) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=timeout_s,
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def resolve(self, candidate: Candidate) -> Movie | None:
        params: dict[str, str | int] = {"query": candidate.title, "include_adult": "false"}
        if candidate.year:
            params["year"] = candidate.year
        async with self._sem:
            try:
                r = await self._client.get("/search/movie", params=params)
                r.raise_for_status()
                parsed = _TMDBSearch.model_validate(r.json())
            except (httpx.HTTPError, ValidationError, ValueError) as e:
                log.warning("tmdb lookup failed for %r: %s", candidate.title, e)
                return None
        if not parsed.results:
            return None
        best = parsed.results[0]
        return Movie(
            tmdb_id=best.id,
            title=best.title,
            year=_year(best.release_date),
            overview=best.overview[:1200],
            genres=[_GENRES[g] for g in best.genre_ids if g in _GENRES],
            popularity=best.popularity,
        )

    async def fetch(self, tmdb_id: int) -> Movie | None:
        """Movie card by TMDB id. Used by the offline eval, where ids come from MovieLens."""
        async with self._sem:
            try:
                r = await self._client.get(f"/movie/{tmdb_id}")
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                d = _TMDBDetail.model_validate(r.json())
            except (httpx.HTTPError, ValidationError, ValueError) as e:
                log.warning("tmdb fetch failed for %d: %s", tmdb_id, e)
                return None
        return Movie(
            tmdb_id=d.id,
            title=d.title,
            year=_year(d.release_date),
            overview=d.overview[:1200],
            genres=[_GENRES[g.id] for g in d.genres if g.id in _GENRES],
            popularity=d.popularity,
        )
