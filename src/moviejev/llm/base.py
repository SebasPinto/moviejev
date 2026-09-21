from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    pass


class LLM(ABC):
    """Minimal provider-agnostic text LLM. Only what the pipeline needs."""

    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...

    async def complete_json(
        self, system: str, user: str, schema: type[T], max_tokens: int = 1024
    ) -> T:
        """Ask for JSON and validate it. LLM output is untrusted: never trust, always parse."""
        raw = await self.complete(system, user, max_tokens)
        text = _FENCE.sub("", raw.strip())
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned non-JSON: {text[:200]!r}") from e
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMError(f"LLM JSON failed schema {schema.__name__}: {e}") from e
