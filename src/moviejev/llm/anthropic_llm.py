from __future__ import annotations

from anthropic import AsyncAnthropic

from moviejev.llm.base import LLM, LLMError


class AnthropicLLM(LLM):
    def __init__(
        self, api_key: str, model: str, timeout_s: float, workspace_id: str | None = None
    ) -> None:
        super().__init__()
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        self._client = AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=2, default_headers=headers
        )
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.usage.add(msg.usage.input_tokens, msg.usage.output_tokens)
        parts = [b.text for b in msg.content if b.type == "text"]
        if not parts:
            raise LLMError("empty completion")
        return "".join(parts)
