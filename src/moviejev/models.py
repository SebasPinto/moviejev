from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TasteProfile(BaseModel):
    """Structured interpretation of the user's request. Produced by the LLM, validated here."""

    liked_titles: list[str] = Field(default_factory=list, max_length=10)
    genres: list[str] = Field(default_factory=list, max_length=8)
    themes: list[str] = Field(default_factory=list, max_length=8)
    tone: str = Field(default="", max_length=120)
    exclusions: list[str] = Field(default_factory=list, max_length=10)
    era: str = Field(default="", max_length=60)

    @field_validator("liked_titles", "genres", "themes", "exclusions", "tone", "era", mode="before")
    @classmethod
    def _clip(cls, v: object, info: object) -> object:
        # LLMs overrun length hints; clipping keeps the bound without losing the request.
        limits = {
            "liked_titles": 10,
            "genres": 8,
            "themes": 8,
            "exclusions": 10,
            "tone": 120,
            "era": 60,
        }
        name = getattr(info, "field_name", "")
        return v[: limits[name]] if isinstance(v, (str, list)) and name in limits else v

    def as_state(self) -> str:
        parts = []
        if self.liked_titles:
            parts.append(f"Liked: {', '.join(self.liked_titles)}")
        if self.genres:
            parts.append(f"Genres: {', '.join(self.genres)}")
        if self.themes:
            parts.append(f"Themes: {', '.join(self.themes)}")
        if self.tone:
            parts.append(f"Tone: {self.tone}")
        if self.era:
            parts.append(f"Era: {self.era}")
        if self.exclusions:
            parts.append(f"Must avoid: {', '.join(self.exclusions)}")
        return "\n".join(parts)


class Candidate(BaseModel):
    """A title proposed by the LLM, before catalog validation."""

    title: str = Field(min_length=1, max_length=200)
    year: int | None = Field(default=None, ge=1880, le=2100)


class Movie(BaseModel):
    """A title validated against the catalog."""

    tmdb_id: int
    title: str
    year: int | None
    overview: str = Field(default="", max_length=1200)
    genres: list[str] = Field(default_factory=list)
    popularity: float = 0.0

    def as_state(self) -> str:
        y = f" ({self.year})" if self.year else ""
        g = f"\nGenres: {', '.join(self.genres)}" if self.genres else ""
        return f"Title: {self.title}{y}{g}\nOverview: {self.overview}"


class Verdict(BaseModel):
    """Reranker output for one movie. Provider-agnostic."""

    movie: Movie
    expected_fit: float = Field(ge=0.0, le=1.0)
    fit_confidence: float = Field(ge=0.0, le=1.0)
    violation_prob: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    low_confidence: bool = False

    @property
    def final_score(self) -> float:
        return self.expected_fit * (1.0 - self.violation_prob)


class Recommendation(BaseModel):
    movie: Movie
    score: float
    reason: str
    explanation: str
    low_confidence: bool


class RecommendResponse(BaseModel):
    profile: TasteProfile
    recommendations: list[Recommendation]
    dropped_unverified: list[str]
    reranker: str
