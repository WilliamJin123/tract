"""Claude Code LLM client.

Uses the ``claude`` CLI (``claude -p``) as an LLM backend, piggybacking on
the user's existing Claude Code subscription.  This avoids needing a
separate Anthropic API key — Claude Code handles OAuth, token refresh,
and rate limiting internally.

The client implements the :class:`~tract.llm.protocols.LLMClient` protocol
so it can be used anywhere tract expects an LLM client::

    from tract.llm.claude_code import ClaudeCodeClient

    client = ClaudeCodeClient(model="sonnet")
    t.configure_llm(client=client)

Model aliases (``sonnet``, ``opus``, ``haiku``) are supported alongside
full model IDs (``claude-sonnet-4-6``).

Limitations:
- No streaming support (returns full responses only).
- Tool/function calling is not supported through this client.
- Each ``chat()`` call spawns a subprocess (~2-4s overhead).
- Async ``achat()`` uses ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from typing import Any

from tract.llm.errors import LLMAuthError, LLMConfigError, LLMResponseError

logger = logging.getLogger(__name__)

__all__ = [
    "ClaudeCodeClient",
    "load_claude_code_credentials",
    "create_claude_code_client",
]

# ---------------------------------------------------------------------------
# Credential helpers (kept for backward compat / introspection)
# ---------------------------------------------------------------------------

_DEFAULT_CREDENTIALS_PATH = "~/.claude/.credentials.json"


def load_claude_code_credentials(
    path: str | None = None,
) -> dict[str, Any]:
    """Load Claude Code OAuth credentials from disk.

    Useful for checking token expiry or subscription type.
    Not needed for :class:`ClaudeCodeClient` (which delegates auth
    to the ``claude`` CLI).

    Returns:
        The ``claudeAiOauth`` dict with keys ``accessToken``,
        ``refreshToken``, ``expiresAt``, ``scopes``, etc.
    """
    from pathlib import Path

    cred_path = Path(path).expanduser() if path else Path(_DEFAULT_CREDENTIALS_PATH).expanduser()

    if not cred_path.is_file():
        raise LLMConfigError(
            f"Claude Code credentials not found at {cred_path}. "
            f"Run 'claude' in your terminal to sign in."
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

    return oauth


# ---------------------------------------------------------------------------
# ClaudeCodeClient
# ---------------------------------------------------------------------------


class ClaudeCodeClient:
    """LLM client that delegates to the ``claude`` CLI.

    Implements the :class:`~tract.llm.protocols.LLMClient` protocol by
    shelling out to ``claude -p --output-format json``.  All authentication
    is handled by Claude Code's own OAuth flow.

    Args:
        model: Default model alias or full ID (e.g. ``"sonnet"``,
            ``"opus"``, ``"claude-sonnet-4-6"``).
        timeout: Subprocess timeout in seconds.
        claude_bin: Path to the ``claude`` binary.  Auto-detected if None.
        system_prompt: Optional system prompt appended to every request.
        no_session_persistence: If True, pass ``--no-session-persistence``
            to avoid saving cookbook test runs as Claude Code sessions.
    """

    def __init__(
        self,
        model: str = "sonnet",
        timeout: float = 120.0,
        claude_bin: str | None = None,
        system_prompt: str | None = None,
        no_session_persistence: bool = True,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._system_prompt = system_prompt
        self._no_session_persistence = no_session_persistence

        # Locate claude binary
        if claude_bin:
            self._claude_bin = claude_bin
        else:
            found = shutil.which("claude")
            if not found:
                raise LLMConfigError(
                    "Could not find 'claude' on PATH. "
                    "Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
                )
            self._claude_bin = found

        # Stats
        self.calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # LLMClient protocol
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send messages via ``claude -p``, return OpenAI-format response."""
        prompt, system = self._format_messages(messages)
        cmd = self._build_command(
            prompt, model=model, system=system,
            max_tokens=max_tokens, **kwargs,
        )

        logger.debug("ClaudeCodeClient running: %s", " ".join(cmd[:6]) + "...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMResponseError(
                f"Claude CLI timed out after {self._timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise LLMConfigError(
                f"Claude binary not found at {self._claude_bin}"
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise LLMResponseError(
                f"Claude CLI exited with code {result.returncode}: {stderr}"
            )

        return self._parse_response(result.stdout)

    def close(self) -> None:
        """No-op (no persistent resources)."""

    def extract_content(self, response: dict) -> str:
        """Extract text content from the response."""
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return response.get("_raw_result", "")

    def extract_usage(self, response: dict) -> dict | None:
        """Extract usage dict from the response."""
        return response.get("usage")

    # ------------------------------------------------------------------
    # Async support
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Async chat via ``claude -p`` subprocess."""
        prompt, system = self._format_messages(messages)
        cmd = self._build_command(
            prompt, model=model, system=system,
            max_tokens=max_tokens, **kwargs,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise LLMResponseError(
                f"Claude CLI timed out after {self._timeout}s"
            ) from exc

        if proc.returncode != 0:
            err = stderr.decode().strip() if stderr else "unknown error"
            raise LLMResponseError(
                f"Claude CLI exited with code {proc.returncode}: {err}"
            )

        return self._parse_response(stdout.decode())

    async def aclose(self) -> None:
        """No-op."""

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ClaudeCodeClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> ClaudeCodeClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Cumulative usage statistics."""
        return {
            "calls": self.calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _format_messages(
        self, messages: list[dict[str, str]],
    ) -> tuple[str, str | None]:
        """Convert OpenAI-format messages to a prompt string + system prompt.

        Returns:
            (prompt_text, system_prompt_or_None)
        """
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                conversation_parts.append(f"[Assistant]: {content}")
            elif role == "tool":
                # Tool results — include as context
                tool_id = msg.get("tool_call_id", "")
                conversation_parts.append(
                    f"[Tool Result {tool_id}]: {content}"
                )
            else:
                # user or other
                conversation_parts.append(content)

        # Combine system prompt from messages + instance default
        all_system = []
        if self._system_prompt:
            all_system.append(self._system_prompt)
        all_system.extend(system_parts)
        system = "\n\n".join(all_system) if all_system else None

        prompt = "\n\n".join(conversation_parts) if conversation_parts else ""
        return prompt, system

    def _build_command(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Build the ``claude`` CLI command."""
        cmd = [
            self._claude_bin,
            "-p",
            "--output-format", "json",
            "--model", model or self._model,
        ]

        if self._no_session_persistence:
            cmd.append("--no-session-persistence")

        if system:
            cmd.extend(["--append-system-prompt", system])

        if max_tokens is not None:
            # Claude Code doesn't have --max-tokens but we can include
            # it as part of the system prompt instruction
            cmd.extend([
                "--append-system-prompt",
                f"Keep your response under {max_tokens} tokens.",
            ])

        cmd.append(prompt)
        return cmd

    def _parse_response(self, stdout: str) -> dict:
        """Parse ``claude -p --output-format json`` output to OpenAI format."""
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Failed to parse Claude CLI JSON output: {exc}\n"
                f"Raw output: {stdout[:500]}"
            ) from exc

        if data.get("is_error"):
            raise LLMResponseError(
                f"Claude CLI returned error: {data.get('result', 'unknown')}"
            )

        result_text = data.get("result", "")
        stop_reason = data.get("stop_reason", "end_turn")
        usage_raw = data.get("usage", {})
        model_usage = data.get("modelUsage", {})

        # Extract usage
        input_tokens = usage_raw.get("input_tokens", 0)
        output_tokens = usage_raw.get("output_tokens", 0)
        cache_creation = usage_raw.get("cache_creation_input_tokens", 0)
        cache_read = usage_raw.get("cache_read_input_tokens", 0)

        # Update cumulative stats
        self.calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += data.get("total_cost_usd", 0.0)

        # Determine model name from modelUsage keys
        model_name = ""
        if model_usage:
            model_name = next(iter(model_usage), "")

        # Map stop_reason to OpenAI finish_reason
        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }

        usage: dict[str, Any] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if cache_creation:
            usage["cache_creation_input_tokens"] = cache_creation
        if cache_read:
            usage["cache_read_input_tokens"] = cache_read

        return {
            "id": data.get("session_id", ""),
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result_text,
                },
                "finish_reason": finish_reason_map.get(stop_reason, "stop"),
            }],
            "usage": usage,
            "_raw_result": result_text,
            "_cost_usd": data.get("total_cost_usd", 0.0),
            "_duration_ms": data.get("duration_ms", 0),
        }


# ---------------------------------------------------------------------------
# Convenience factory (backward-compatible name)
# ---------------------------------------------------------------------------


def create_claude_code_client(
    *,
    model: str = "sonnet",
    timeout: float = 120.0,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> ClaudeCodeClient:
    """Create a :class:`ClaudeCodeClient` instance.

    This is the recommended way to use Claude models via your Claude Code
    subscription for cookbook testing and development.

    Example::

        from tract.llm.claude_code import create_claude_code_client

        client = create_claude_code_client(model="sonnet")
        t.configure_llm(client=client)

        # Now all LLM operations (compress, generate, etc.) go through
        # your Claude Code subscription — no API key needed.
    """
    return ClaudeCodeClient(
        model=model,
        timeout=timeout,
        system_prompt=system_prompt,
        **kwargs,
    )
