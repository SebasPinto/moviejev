from __future__ import annotations

from moviejev.llm.anthropic_llm import AnthropicLLM


def test_workspace_header_sent_when_configured() -> None:
    llm = AnthropicLLM("k", "m", timeout_s=1, workspace_id="wrkspc_123")
    assert llm._client.default_headers["anthropic-workspace-id"] == "wrkspc_123"


def test_no_workspace_header_by_default() -> None:
    llm = AnthropicLLM("k", "m", timeout_s=1)
    assert "anthropic-workspace-id" not in llm._client.default_headers
