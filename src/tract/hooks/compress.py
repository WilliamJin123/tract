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
from tract.hooks.pending import Pending, PendingStatus
from tract.hooks.validation import ValidationResult

if TYPE_CHECKING:
    from tract.models.compression import CompressResult
    from tract.storage.schema import CommitRow
    from tract.tract import Tract


@dataclass(repr=False)
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
    _two_stage: bool = field(default=False, repr=False)

    # -- Whitelist for agent dispatch -----------------------------------

    _public_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "approve",
            "reject",
            "edit_summary",
            "edit_guidance",
            "retry",
            "validate",
        }),
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
        self.status = PendingStatus.APPROVED
        self._result = self._execute_fn(self)
        return self._result

    def reject(self, reason: str = "") -> None:
        """Reject the compression, discarding all planned changes.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = PendingStatus.REJECTED
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

    # -- Retry and validation -------------------------------------------

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
            RuntimeError: If status is not "pending".
            IndexError: If index is out of range.
            RuntimeError: If no groups are available for retry.
        """
        self._require_pending()

        if index < 0 or index >= len(self.summaries):
            raise IndexError(
                f"Summary index {index} is out of range. "
                f"Valid indices: 0..{len(self.summaries) - 1}"
            )

        if not self._groups or index >= len(self._groups):
            raise RuntimeError(
                "Cannot retry: no compression groups available. "
                "This PendingCompress may have been created with content= (manual mode)."
            )

        # Deferred imports to avoid circular dependencies
        from tract.operations.compression import _build_messages_text, _summarize_group

        group = self._groups[index]
        messages_text = _build_messages_text(group, self.tract._blob_repo)

        # Combine instructions: self._instructions base + guidance overlay
        combined = self._instructions or ""
        if self.guidance:
            combined = (self.guidance + "\n" + combined) if combined else self.guidance
        if guidance:
            combined = (combined + "\n" + guidance) if combined else guidance

        new_summary = _summarize_group(
            messages_text,
            self.tract._resolve_llm_client("compress"),
            self.tract._token_counter,
            target_tokens=self._target_tokens,
            instructions=combined or None,
            system_prompt=self._system_prompt,
            llm_kwargs=llm_overrides or None,
        )

        self.summaries[index] = new_summary

        # Recalculate estimated tokens
        self.estimated_tokens = sum(
            self.tract._token_counter.count_text(s) for s in self.summaries
        )

    def validate(self) -> ValidationResult:
        """Validate the current summaries against quality criteria.

        Checks each summary for:
        1. Non-empty (no blank summaries)
        2. Not trivially short (< 10 chars = suspiciously truncated)
        3. Token ratio: if _target_tokens is set, individual summary
           should not exceed _target_tokens * 1.5

        Returns:
            ValidationResult indicating whether all summaries pass.
        """
        for i, summary in enumerate(self.summaries):
            # Check non-empty
            if not summary or not summary.strip():
                return ValidationResult(
                    passed=False,
                    diagnosis=f"Summary at index {i} is empty.",
                    index=i,
                )

            # Check not trivially short
            if len(summary.strip()) < 10:
                return ValidationResult(
                    passed=False,
                    diagnosis=(
                        f"Summary at index {i} is suspiciously short "
                        f"({len(summary.strip())} chars). "
                        f"Content: {summary.strip()!r}"
                    ),
                    index=i,
                )

            # Check token ratio if target set
            if self._target_tokens is not None:
                token_count = self.tract._token_counter.count_text(summary)
                max_tokens = int(self._target_tokens * 1.5)
                if token_count > max_tokens:
                    return ValidationResult(
                        passed=False,
                        diagnosis=(
                            f"Summary at index {i} exceeds token budget: "
                            f"{token_count} tokens > {max_tokens} "
                            f"(target={self._target_tokens} * 1.5)."
                        ),
                        index=i,
                    )

        return ValidationResult(passed=True)

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

    def __repr__(self):
        status = self.status.value if hasattr(self.status, 'value') else str(self.status)
        pct = ""
        if self.original_tokens > 0:
            pct = f", {int((1 - self.estimated_tokens / self.original_tokens) * 100)}% reduction"
        return f"<PendingCompress: {len(self.summaries)} summaries, {self.original_tokens}->{self.estimated_tokens} tokens{pct}, {status}>"

    def _compact_detail(self) -> str:
        pct = ""
        if self.original_tokens > 0:
            pct = f" ({int((1 - self.estimated_tokens / self.original_tokens) * 100)}%)"
        return f"{self.original_tokens}->{self.estimated_tokens} tokens{pct}"

    def _pprint_details(self, console, *, verbose: bool = False) -> None:
        """Show compression-specific details: token ratio, summaries, guidance."""
        from rich.panel import Panel

        # Token ratio summary
        if self.original_tokens > 0:
            pct = int((1 - self.estimated_tokens / self.original_tokens) * 100)
            console.print(
                f"  Compression: {self.original_tokens} -> {self.estimated_tokens} "
                f"tokens ({pct}% reduction)"
            )
        else:
            console.print(
                f"  Compression: {self.original_tokens} -> {self.estimated_tokens} tokens"
            )

        # Verbose: show summary previews
        if verbose and self.summaries:
            console.print("  [bold]Summary previews:[/bold]")
            for i, summary in enumerate(self.summaries):
                preview = summary[:120]
                if len(summary) > 120:
                    preview += "..."
                console.print(f"    [{i}] {preview}")

        # Guidance panel
        if self.guidance:
            console.print(Panel(self.guidance, title="Guidance", style="cyan"))
