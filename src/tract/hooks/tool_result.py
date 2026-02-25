"""PendingToolResult -- hook object for tool result operations.

Allows handlers to inspect, edit, summarize, or reject a tool result
before it enters the commit chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.pending import Pending

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.tract import Tract


@dataclass
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

    # Stores the CommitInfo result after approve()
    _result: Any = field(default=None, repr=False)

    _public_actions: set[str] = field(
        default_factory=lambda: {
            "approve", "reject", "edit_result", "summarize",
        },
        repr=False,
    )

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
        self.status = "approved"
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
    ) -> None:
        """Summarize the result content via LLM.

        Uses ``TOOL_SUMMARIZE_SYSTEM`` as the system prompt. The original
        content is preserved in ``original_content``.

        Requires an LLM client configured on the Tract instance
        (via ``api_key=`` on ``Tract.open()`` or ``configure_operations()``).

        Args:
            instructions: Extra guidance for the summarization LLM.
            target_tokens: Target token count for the summary.

        Raises:
            RuntimeError: If this pending has already been resolved.
            AttributeError: If no LLM client is configured.
        """
        self._require_pending()
        if self.original_content is None:
            self.original_content = self.content

        from tract.prompts.summarize import TOOL_SUMMARIZE_SYSTEM, build_summarize_prompt

        llm = self.tract._resolve_llm_client("compress")
        prompt = build_summarize_prompt(
            f"[tool:{self.tool_name}]: {self.content}",
            target_tokens=target_tokens,
            instructions=instructions,
        )
        response = llm.chat([
            {"role": "system", "content": TOOL_SUMMARIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        self.content = response["choices"][0]["message"]["content"]
