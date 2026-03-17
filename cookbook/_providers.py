"""Cookbook LLM provider configuration.

Centralizes API keys, base URLs, and model IDs so any cookbook example
can swap providers by changing a single import line:

    from _providers import cerebras as llm   # <- change this
    from _providers import groq as llm       # <- to this

Then use ``llm.api_key``, ``llm.base_url``, ``llm.large``, ``llm.small``
throughout the file.
"""

import os
from dataclasses import dataclass
from pathlib import Path

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


cerebras = Provider(
    api_key=os.environ.get("CEREBRAS_API_KEY", ""),
    base_url=os.environ.get("CEREBRAS_BASE_URL", ""),
    xlarge="gpt-oss-120b",
    large="gpt-oss-120b",
    small="llama3.1-8b",
)


def _load_claude_code_token() -> str:
    """Read Claude Code OAuth token, return empty string on failure."""
    try:
        import json
        creds = Path.home() / ".claude" / ".credentials.json"
        if not creds.is_file():
            return ""
        data = json.loads(creds.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {})
        token = oauth.get("accessToken", "")
        # Check expiry
        import time
        expires_at = oauth.get("expiresAt", 0)
        if expires_at and time.time() * 1000 >= expires_at:
            return ""
        return token
    except Exception:
        return ""


claude_code = Provider(
    api_key=_load_claude_code_token(),
    base_url="",  # uses default Anthropic API
    xlarge="claude-opus-4-6",
    large="claude-sonnet-4-6",
    small="claude-haiku-4-5-20251001",
)

groq = Provider(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url=os.environ.get("GROQ_BASE_URL", ""),
    xlarge="moonshotai/kimi-k2-instruct-0905",
    large="meta-llama/llama-4-scout-17b-16e-instruct",
    small="qwen/qwen3-32b",
)
