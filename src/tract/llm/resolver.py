"""Built-in OpenAI-powered conflict resolver.

Uses an OpenAI-compatible LLM to resolve merge conflicts, rebase warnings,
and cherry-pick issues. Implements the ResolverCallable protocol.
"""

from __future__ import annotations

from typing import Any

from tract.llm.protocols import LLMClient, Resolution


class OpenAIResolver:
    """Built-in conflict resolver using an OpenAI-compatible LLM.

    Implements the ResolverCallable protocol. Sends conflict info
    to the LLM and returns a Resolution.

    Usage::

        from tract.llm import OpenAIClient, OpenAIResolver

        client = OpenAIClient(api_key="sk-...")
        resolver = OpenAIResolver(client, model="gpt-4o-mini")
        resolution = resolver(conflict_info)
    """

    def __init__(
        self,
        client: LLMClient | Any,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the resolver.

        Args:
            client: An LLM client conforming to the LLMClient protocol.
            model: Model to use for resolution. Falls back to client default.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens for the resolution response.
            system_prompt: Custom system prompt. Falls back to default.
        """
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or self._default_system_prompt()

    @staticmethod
    def _default_system_prompt() -> str:
        """Return the default system prompt for conflict resolution."""
        return (
            "You are a context merge resolver for an LLM context management system. "
            "You receive conflicting context from two branches and must produce a single, "
            "coherent resolution. Respond with ONLY the resolved content text. "
            "Do not add explanations or metadata unless asked."
        )

    def __call__(self, issue: object) -> Resolution:
        """Resolve a conflict/issue using the LLM.

        Args:
            issue: A conflict info object. Duck-typed access to attributes:
                conflict_type, content_a_text, content_b_text,
                ancestor_content_text, compiled_context, etc.

        Returns:
            Resolution with action="resolved" and the LLM's response.
        """
        user_prompt = self._format_issue(issue)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self._client.chat(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            content_text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            from tract.llm.errors import LLMResponseError

            raise LLMResponseError(
                f"Cannot extract content from LLM response: {exc}. "
                f"Response: {response}"
            ) from exc
        usage = response.get("usage")
        gen_config: dict[str, Any] = {
            "model": response.get("model", self._model),
            "temperature": self._temperature,
            "source": "infrastructure:merge",
        }
        if usage:
            gen_config["usage"] = usage
        return Resolution(
            action="resolved",
            content_text=content_text,
            reasoning=f"LLM-resolved using model {gen_config.get('model')}",
            generation_config=gen_config,
        )

    def _format_issue(self, issue: object) -> str:
        """Format a conflict/issue object into a user prompt string.

        Uses duck-typed attribute access so this works with any issue type
        (ConflictInfo, RebaseWarning, ImportIssue) without importing them.
        """
        conflict_type = getattr(issue, "conflict_type", "unknown")
        parts: list[str] = [f"Conflict type: {conflict_type}"]

        content_a = getattr(issue, "content_a_text", None)
        content_b = getattr(issue, "content_b_text", None)
        if content_a:
            parts.append(f"Branch A content:\n{content_a}")
        if content_b:
            parts.append(f"Branch B content:\n{content_b}")

        ancestor_text = getattr(issue, "ancestor_content_text", None)
        if ancestor_text:
            parts.append(f"Common ancestor content:\n{ancestor_text}")

        context = getattr(issue, "compiled_context", None)
        if context and hasattr(context, "messages"):
            msgs = context.messages[:5]
            ctx_lines: list[str] = []
            for m in msgs:
                role = getattr(m, "role", "unknown")
                content = getattr(m, "content", "")
                ctx_lines.append(f"[{role}] {content[:200]}")
            ctx_text = "\n".join(ctx_lines)
            parts.append(f"Surrounding context (truncated):\n{ctx_text}")

        return "\n\n".join(parts)
