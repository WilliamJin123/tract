"""PendingCompress -- hook object for compression operations.

Replaces PendingCompression from models/compression.py with a richer
interface: approve/reject/edit_summary/retry/validate/edit_interactive.

Internal fields mirror the old PendingCompression for finalization
compatibility with compress_range() / _finalize_compression().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.guidance import GuidanceMixin
from tract.hooks.pending import Pending
from tract.hooks.validation import ValidationResult

if TYPE_CHECKING:
    from tract.models.compression import CompressResult
    from tract.storage.schema import CommitRow
    from tract.tract import Tract


@dataclass
class PendingCompress(GuidanceMixin, Pending):
    """A compression operation that has been planned but not yet committed.

    Mutable: users/handlers can edit summaries, retry individual groups,
    or reject the entire compression before it commits.

    Public fields expose what the handler needs to make decisions.
    Internal fields (prefixed with _) carry state needed by the
    finalization pipeline.
    """

    # -- Public fields --------------------------------------------------

    summaries: list[str] = field(default_factory=list)
    """LLM-generated summary texts, one per compression group."""

    source_commits: list[str] = field(default_factory=list)
    """Hashes of commits being compressed (consumed by the compression)."""

    preserved_commits: list[str] = field(default_factory=list)
    """Hashes of pinned commits that pass through uncompressed."""

    original_tokens: int = 0
    """Total tokens in the source commits before compression."""

    estimated_tokens: int = 0
    """Estimated tokens after compression (based on summaries)."""

    guidance: str | None = None
    """Guidance text for the compression (from user instructions or LLM)."""

    guidance_source: str | None = None
    """Where guidance came from: None, "user", "llm", or "user+llm"."""

    # -- Internal state for finalization --------------------------------
    # These mirror the old PendingCompression fields and are set by
    # compress_range(), read by _finalize_compression().

    _range_commits: list[CommitRow] | None = field(default=None, repr=False)
    _pinned_commits: list[CommitRow] | None = field(default=None, repr=False)
    _normal_commits: list[CommitRow] | None = field(default=None, repr=False)
    _pinned_hashes: set[str] | None = field(default=None, repr=False)
    _skip_hashes: set[str] | None = field(default=None, repr=False)
    _groups: list[list[CommitRow]] | None = field(default=None, repr=False)
    _branch_name: str | None = field(default=None, repr=False)
    _target_tokens: int | None = field(default=None, repr=False)
    _instructions: str | None = field(default=None, repr=False)
    _system_prompt: str | None = field(default=None, repr=False)
    _head_hash: str | None = field(default=None, repr=False)
    _generation_config: dict | None = field(default=None, repr=False)

    # -- Whitelist for agent dispatch -----------------------------------

    _public_actions: set[str] = field(
        default_factory=lambda: {
            "approve",
            "reject",
            "edit_summary",
            "edit_guidance",
        },
        repr=False,
    )

    def __post_init__(self) -> None:
        # Ensure operation is set correctly
        if not self.operation:
            self.operation = "compress"

    # -- Core methods ---------------------------------------------------

    def approve(self) -> CompressResult:
        """Finalize and commit all summaries, completing the compression.

        Returns:
            CompressResult with compression_id, summary_commits, new_head, etc.

        Raises:
            RuntimeError: If status is not "pending" or no execute function is set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                "Cannot approve: no execute function set. "
                "This PendingCompress was not created by Tract.compress()."
            )
        self.status = "approved"
        return self._execute_fn(self)

    def reject(self, reason: str = "") -> None:
        """Reject the compression, discarding all planned changes.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = "rejected"
        self.rejection_reason = reason

    # -- Editing methods ------------------------------------------------

    def edit_summary(self, index: int, new_text: str) -> None:
        """Replace the summary text at the given index.

        Args:
            index: Index into the summaries list.
            new_text: Replacement summary text.

        Raises:
            IndexError: If index is out of range.
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.summaries[index] = new_text

    # -- Retry and validation stubs -------------------------------------

    def retry(self, index: int, *, guidance: str = "", **llm_overrides: Any) -> None:
        """Re-run LLM generation for one summary group.

        Re-generates the summary at the given index, injecting the previous
        output and guidance into the prompt for steering.

        Args:
            index: Index of the summary to regenerate.
            guidance: Feedback text to inject into the retry prompt.
            **llm_overrides: Override LLM parameters for this retry
                (e.g. model, temperature).

        Raises:
            NotImplementedError: Until Phase 1 wiring is complete.
        """
        raise NotImplementedError(
            "retry() is not yet implemented. Use edit_summary() for manual corrections."
        )

    def validate(self) -> ValidationResult:
        """Validate the current summaries against quality criteria.

        Returns:
            ValidationResult indicating whether all summaries pass.

        Raises:
            NotImplementedError: Until Phase 1 wiring is complete.
        """
        raise NotImplementedError(
            "validate() is not yet implemented."
        )

    def edit_interactive(self) -> None:
        """Launch an interactive editing session for summaries.

        Opens a rich TUI for reviewing and editing each summary
        before approving or rejecting.

        Raises:
            NotImplementedError: Until CLI integration is complete.
        """
        raise NotImplementedError(
            "edit_interactive() is not yet implemented. Use edit_summary() for manual edits."
        )

    # -- Display --------------------------------------------------------
    # Inherits Rich-based pprint() from Pending base class.
