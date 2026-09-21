from __future__ import annotations

from abc import ABC, abstractmethod

from moviejev.models import Movie, TasteProfile, Verdict


class Reranker(ABC):
    name: str

    @abstractmethod
    async def judge(self, profile: TasteProfile, movies: list[Movie]) -> list[Verdict]: ...


def rank(verdicts: list[Verdict]) -> list[Verdict]:
    """Deterministic ordering: final score desc, then confidence desc, then title (stable)."""
    return sorted(verdicts, key=lambda v: (-v.final_score, -v.fit_confidence, v.movie.title))


# Shared rubric so Jev and the LLM judge answer the *same* question. Keep in sync.
FIT_LEVELS: list[str] = [
    "Clearly a poor match for this profile",
    "Weak match; shares little with the profile",
    "Partial match; overlaps on some genres or themes",
    "Strong match on tone, themes and genres",
    "Near-perfect match; exactly what this person is asking for",
]

REASON_CHOICES: dict[str, str] = {
    "tone": "Matches mainly on tone, mood or pacing",
    "themes": "Matches mainly on themes or subject matter",
    "genre": "Matches mainly on genre conventions",
    "style": "Matches mainly on directorial or visual style",
    "era": "Matches mainly on period or era",
}

VIOLATION_INSTRUCTION = (
    "This movie violates one of the items listed under 'Must avoid' in the profile"
)


def expected_value(probabilities: dict[str, float], n_levels: int) -> float:
    """E[level]/(n-1) over an ordered scale. Uses the full distribution, not the argmax."""
    if n_levels < 2:
        return 0.0
    total = sum(probabilities.values())
    if total <= 0:
        return 0.0
    ev = sum(int(k) * p for k, p in probabilities.items()) / total
    return max(0.0, min(1.0, ev / (n_levels - 1)))
