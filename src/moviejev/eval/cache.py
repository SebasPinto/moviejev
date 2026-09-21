from __future__ import annotations

import hashlib
import json
from pathlib import Path

from moviejev.llm.base import LLM


class JsonCache:
    """Tiny persistent dict. Written on every set so an interrupted run keeps its progress."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = {}
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data = {str(k): str(v) for k, v in loaded.items()}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def __len__(self) -> int:
        return len(self._data)


class CachingLLM(LLM):
    """Memoises `complete` on disk so reruns of the eval only pay for the reranker under test."""

    def __init__(self, inner: LLM, cache: JsonCache) -> None:
        super().__init__()
        self._inner = inner
        self._cache = cache
        self.usage = inner.usage  # share the counter: cache hits cost nothing
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(system: str, user: str, max_tokens: int) -> str:
        h = hashlib.sha256()
        h.update(system.encode())
        h.update(b"\x00")
        h.update(user.encode())
        h.update(b"\x00")
        h.update(str(max_tokens).encode())
        return h.hexdigest()

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        k = self.key(system, user, max_tokens)
        hit = self._cache.get(k)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        out = await self._inner.complete(system, user, max_tokens)
        self._cache.set(k, out)
        return out
