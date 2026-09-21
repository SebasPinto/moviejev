from __future__ import annotations

from abc import ABC, abstractmethod

from moviejev.models import Candidate, Movie


class Catalog(ABC):
    """Source of truth for what exists. Anything not found here never reaches the user."""

    @abstractmethod
    async def resolve(self, candidate: Candidate) -> Movie | None: ...
