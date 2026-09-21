from __future__ import annotations

from moviejev.config import Settings
from moviejev.llm.base import LLM


def build_llm(s: Settings) -> LLM:
    if s.llm_provider == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        from moviejev.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            s.anthropic_api_key.get_secret_value(),
            s.anthropic_model,
            s.request_timeout_s,
            workspace_id=s.anthropic_workspace_id,
        )
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
    from moviejev.llm.openai_llm import OpenAILLM

    return OpenAILLM(s.openai_api_key.get_secret_value(), s.openai_model, s.request_timeout_s)
