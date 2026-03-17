"""Cookbook LLM provider configuration.

Centralizes API keys, base URLs, and model IDs so any cookbook example
can swap providers by changing a single import line:

    from _providers import cerebras as llm   # <- change this
    from _providers import groq as llm       # <- to this
    from _providers import claude_code as llm  # uses Claude Code CLI

Then use ``llm.api_key``, ``llm.base_url``, ``llm.large``, ``llm.small``
throughout the file.

For ``claude_code``, ``api_key`` is empty (auth is handled by the CLI).
Use ``provider="claude_code"`` in ``Tract.open()`` or pass
``llm_client=llm.client()`` for direct client access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


@dataclass(frozen=True)
class Provider:
    """Frozen bag of credentials + model aliases for one provider."""

    api_key: str
    base_url: str
    large: str
    small: str
    xlarge: str = ""
    _provider_type: str = ""

    @property
    def available(self) -> bool:
        """Check if this provider is ready to use.

        For claude_code, checks that ``claude`` CLI is on PATH.
        For others, checks that ``api_key`` is non-empty.
        """
        if self._provider_type == "claude_code":
            import shutil
            return shutil.which("claude") is not None
        return bool(self.api_key)

    def client(self, model: str | None = None, **kwargs: Any) -> Any:
        """Create an LLM client for this provider.

        For claude_code, returns a ClaudeCodeClient (subprocess-based).
        For others, returns an OpenAIClient or AnthropicClient.
        """
        if self._provider_type == "claude_code":
            from tract.llm.claude_code import ClaudeCodeClient
            return ClaudeCodeClient(
                model=model or self.large,
                **kwargs,
            )
        elif self._provider_type == "anthropic":
            from tract.llm.anthropic_client import AnthropicClient
            return AnthropicClient(
                api_key=self.api_key,
                base_url=self.base_url or None,
                default_model=model or self.large,
                **kwargs,
            )
        else:
            from tract.llm.client import OpenAIClient
            return OpenAIClient(
                api_key=self.api_key,
                base_url=self.base_url or None,
                default_model=model or self.large,
                **kwargs,
            )

    def tract_kwargs(self, model: str | None = None) -> dict[str, Any]:
        """Return kwargs suitable for ``Tract.open(**llm.tract_kwargs())``.

        For claude_code, returns ``llm_client=ClaudeCodeClient(...)``
        (since there's no api_key to pass).

        For others, returns ``api_key=..., base_url=..., model=...``.
        """
        if self._provider_type == "claude_code":
            return {
                "llm_client": self.client(model=model),
                "model": model or self.large,
            }
        return {
            "api_key": self.api_key,
            "base_url": self.base_url or None,
            "model": model or self.large,
        }


cerebras = Provider(
    api_key=os.environ.get("CEREBRAS_API_KEY", ""),
    base_url=os.environ.get("CEREBRAS_BASE_URL", ""),
    xlarge="gpt-oss-120b",
    large="gpt-oss-120b",
    small="llama3.1-8b",
)

claude_code = Provider(
    api_key="",  # auth handled by Claude Code CLI
    base_url="",
    xlarge="opus",
    large="sonnet",
    small="haiku",
    _provider_type="claude_code",
)

groq = Provider(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url=os.environ.get("GROQ_BASE_URL", ""),
    xlarge="moonshotai/kimi-k2-instruct-0905",
    large="meta-llama/llama-4-scout-17b-16e-instruct",
    small="qwen/qwen3-32b",
)
