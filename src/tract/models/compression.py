"""Domain models for the compression subsystem.

Provides data classes for compression results, pending compressions,
garbage collection results, and reorder warnings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tract.storage.schema import CommitRow


@dataclass(frozen=True)
class CompressResult:
    """Result of a completed compression operation.

    Tracks which commits were compressed, what summaries were produced,
    and the token savings achieved.
    """

    compression_id: str
    original_tokens: int
    compressed_tokens: int
    source_commits: tuple[str, ...]
    summary_commits: tuple[str, ...]
    preserved_commits: tuple[str, ...]
    compression_ratio: float
    """Ratio of compressed_tokens / original_tokens. Values < 1.0 indicate
    effective compression; values > 1.0 indicate the summary expanded tokens."""
    new_head: str


@dataclass
class PendingCompression:
    """A compression that has been planned but not yet committed.

    .. deprecated::
        Use :class:`tract.hooks.compress.PendingCompress` instead.
        This class is kept for backward compatibility with existing tests
        and will be removed in a future version.

    Mutable: users can edit summaries before approving.
    The _commit_fn is set internally by Tract and should not be
    set by users directly.
    """

    summaries: list[str]
    source_commits: list[str]
    preserved_commits: list[str]
    original_tokens: int
    estimated_tokens: int
    _commit_fn: Callable[[PendingCompression], CompressResult] | None = field(
        default=None, repr=False
    )

    # Internal state for finalization (set by compress_range, read by _finalize_compression)
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

    def edit_summary(self, index: int, new_text: str) -> None:
        """Replace the summary text at the given index.

        Args:
            index: Index into the summaries list.
            new_text: Replacement summary text.

        Raises:
            IndexError: If index is out of range.
        """
        self.summaries[index] = new_text

    def approve(self) -> CompressResult:
        """Finalize the compression by calling the commit function.

        Returns:
            CompressResult with the committed compression details.

        Raises:
            CompressionError: If no commit function has been set.
        """
        from tract.exceptions import CompressionError

        if self._commit_fn is None:
            raise CompressionError(
                "Cannot approve: no commit function set. "
                "This PendingCompression was not created by Tract.compress()."
            )
        return self._commit_fn(self)


@dataclass(frozen=True)
class GCResult:
    """Result of a garbage collection operation.

    Tracks what was removed and how much space was freed.
    """

    commits_removed: int
    blobs_removed: int
    tokens_freed: int
    source_commits_removed: int
    duration_seconds: float


@dataclass(frozen=True)
class ReorderWarning:
    """Warning about potential issues when reordering commits during compression.

    Severity levels:
    - "structural": Affects commit graph integrity (e.g., edit before target)
    - "semantic": May affect meaning (e.g., response chain break)
    """

    warning_type: Literal["edit_before_target", "response_chain_break"]
    commit_hash: str
    description: str
    severity: Literal["structural", "semantic"]
