"""PendingToolResult -- hook object for tool result operations.

Allows handlers to inspect, edit, summarize, or reject a tool result
before it enters the commit chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.pending import Pending, PendingStatus

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.tract import Tract


@dataclass(repr=False)
class PendingToolResult(Pending):
    """A tool result that has been received but not yet committed.

    Handlers can inspect, edit, summarize, or reject the result
    before it enters the commit chain.
    """

    tool_call_id: str = ""
    tool_name: str = ""
    content: str = ""
    token_count: int = 0

    # Set by edit_result() or summarize() -- preserves original content
    original_content: str | None = None

    # Whether this tool result represents an error
    is_error: bool = False

    # Stores the CommitInfo result after approve()
    _result: Any = field(default=None, repr=False)

    _public_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "approve", "reject", "edit_result", "summarize",
        }),
        repr=False,
    )

    def __repr__(self):
        status = self.status.value if hasattr(self.status, 'value') else str(self.status)
        return f"<PendingToolResult: {self.tool_name}, {self.token_count} tokens, {status}>"

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = "tool_result"

    def approve(self) -> CommitInfo:
        """Approve and commit the tool result.

        Returns:
            :class:`CommitInfo` for the committed tool result.

        Raises:
            RuntimeError: If status is not "pending" or no execute function is set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                "Cannot approve: no execute function set. "
                "This PendingToolResult was not created by Tract.tool_result()."
            )
        self.status = PendingStatus.APPROVED
        self._result = self._execute_fn(self)
        return self._result

    def edit_result(self, new_content: str) -> None:
        """Replace the result content before commit.

        The original content is preserved in ``original_content`` for
        provenance. Can be called multiple times (original_content is
        only set on the first call).

        Args:
            new_content: The replacement content text.

        Raises:
            RuntimeError: If this pending has already been resolved.
        """
        self._require_pending()
        if self.original_content is None:
            self.original_content = self.content
        self.content = new_content

    def summarize(
        self,
        *,
        instructions: str | None = None,
        target_tokens: int | None = None,
        include_context: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        """Summarize the result content via LLM.

        Uses ``TOOL_SUMMARIZE_SYSTEM`` as the system prompt by default.
        When ``include_context=True``, the current conversation is compiled
        and included in the prompt so the LLM can filter intelligently.
        The original content is preserved in ``original_content``.

        Requires an LLM client configured on the Tract instance
        (via ``api_key=`` on ``Tract.open()`` or ``configure_operations()``).

        Args:
            instructions: Extra guidance for the summarization LLM.
            target_tokens: Target token count for the summary.
            include_context: If True, compile the current conversation and
                pass it as context to the summarization prompt.
            system_prompt: Override the default system prompt entirely.
                When ``include_context=True`` and no explicit system_prompt
                is given, ``TOOL_CONTEXT_SUMMARIZE_SYSTEM`` is used instead
                of the default ``TOOL_SUMMARIZE_SYSTEM``.

        Raises:
            RuntimeError: If this pending has already been resolved.
            AttributeError: If no LLM client is configured.
        """
        self._require_pending()
        if self.original_content is None:
            self.original_content = self.content

        from tract.prompts.summarize import (
            TOOL_CONTEXT_SUMMARIZE_SYSTEM,
            TOOL_SUMMARIZE_SYSTEM,
            build_summarize_prompt,
        )

        llm = self.tract._resolve_llm_client("compress")

        # Build context text if requested
        context_text: str | None = None
        if include_context:
            ctx = self.tract.compile()
            lines = []
            for msg in ctx.messages:
                lines.append(f"{msg.role}: {msg.content}")
            context_text = "\n".join(lines)

        prompt = build_summarize_prompt(
            f"[tool:{self.tool_name}]: {self.content}",
            target_tokens=target_tokens,
            instructions=instructions,
            context_text=context_text,
        )

        # Determine system prompt: explicit > context-aware > default
        if system_prompt is not None:
            sys_prompt = system_prompt
        elif include_context:
            sys_prompt = TOOL_CONTEXT_SUMMARIZE_SYSTEM
        else:
            sys_prompt = TOOL_SUMMARIZE_SYSTEM

        response = llm.chat([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ])
        self.content = response["choices"][0]["message"]["content"]
