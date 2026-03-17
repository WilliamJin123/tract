"""Claude Code credential helper.

Reads OAuth credentials from Claude Code's local credential store
(``~/.claude/.credentials.json``) and provides a factory function to
create an :class:`AnthropicClient` using those credentials.

The OAuth access token is passed as ``api_key=`` (via the ``X-Api-Key``
header) because Anthropic's API does not yet support Bearer-based OAuth
(the ``auth_token=`` path returns 401).

Token expiry is checked before each client creation.  If the token has
expired, an :class:`LLMAuthError` is raised with instructions to refresh
the Claude Code session.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from tract.llm.errors import LLMAuthError, LLMConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "load_claude_code_credentials",
    "create_claude_code_client",
]

_DEFAULT_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"


def load_claude_code_credentials(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load Claude Code OAuth credentials from disk.

    Args:
        path: Override path to the credentials file.
            Defaults to ``~/.claude/.credentials.json``.

    Returns:
        The ``claudeAiOauth`` dict with keys ``accessToken``,
        ``refreshToken``, ``expiresAt``, ``scopes``, etc.

    Raises:
        LLMConfigError: If the credentials file is missing or malformed.
        LLMAuthError: If the access token has expired.
    """
    cred_path = Path(path) if path else _DEFAULT_CREDENTIALS_PATH

    if not cred_path.is_file():
        raise LLMConfigError(
            f"Claude Code credentials not found at {cred_path}. "
            f"Ensure Claude Code is installed and authenticated "
            f"(run 'claude' in your terminal to sign in)."
        )

    try:
        raw = json.loads(cred_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise LLMConfigError(
            f"Failed to read Claude Code credentials at {cred_path}: {exc}"
        ) from exc

    oauth = raw.get("claudeAiOauth")
    if not oauth or not isinstance(oauth, dict):
        raise LLMConfigError(
            f"No 'claudeAiOauth' key in {cred_path}. "
            f"File may be from an older Claude Code version."
        )

    access_token = oauth.get("accessToken", "")
    if not access_token:
        raise LLMConfigError(
            "Claude Code credentials file has no accessToken. "
            "Try re-authenticating: run 'claude' in your terminal."
        )

    # Check expiry (expiresAt is in milliseconds)
    expires_at_ms = oauth.get("expiresAt", 0)
    now_ms = time.time() * 1000
    if expires_at_ms and now_ms >= expires_at_ms:
        raise LLMAuthError(
            "Claude Code OAuth token has expired. "
            "Run 'claude' in your terminal to refresh your session."
        )

    remaining_min = (expires_at_ms - now_ms) / 60_000 if expires_at_ms else None
    if remaining_min is not None:
        logger.info(
            "Claude Code token valid for ~%.0f minutes", remaining_min
        )
        if remaining_min < 10:
            logger.warning(
                "Claude Code token expires in < 10 minutes — "
                "consider refreshing your session."
            )

    return oauth


def create_claude_code_client(
    *,
    model: str = "claude-sonnet-4-6",
    credentials_path: str | Path | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    default_max_tokens: int = 8192,
    **kwargs: Any,
) -> Any:
    """Create an AnthropicClient using Claude Code's OAuth token.

    This is a convenience factory that reads the local credential store
    and passes the access token to ``AnthropicClient(api_key=...)``.

    Args:
        model: Default model for LLM calls.
        credentials_path: Override path to credentials file.
        timeout: Request timeout in seconds.
        max_retries: Max retries for transient errors.
        default_max_tokens: Default max_tokens for completions.
        **kwargs: Extra keyword arguments forwarded to AnthropicClient.

    Returns:
        A configured AnthropicClient instance.

    Raises:
        LLMConfigError: If credentials are missing.
        LLMAuthError: If the token is expired.
    """
    from tract.llm.anthropic_client import AnthropicClient

    oauth = load_claude_code_credentials(credentials_path)
    access_token = oauth["accessToken"]

    return AnthropicClient(
        api_key=access_token,
        default_model=model,
        timeout=timeout,
        max_retries=max_retries,
        default_max_tokens=default_max_tokens,
        **kwargs,
    )
