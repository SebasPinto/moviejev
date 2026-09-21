from __future__ import annotations

from openai import AsyncOpenAI

from moviejev.llm.base import LLM, LLMError


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=2)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content
        if not text:
            raise LLMError("empty completion")
        return text
