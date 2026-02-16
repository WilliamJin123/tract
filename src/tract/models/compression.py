"""Domain models for the compression subsystem.

Provides data classes for compression results, pending compressions,
garbage collection results, and reorder warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompressResult:
    """Result of a completed compression operation.

    Tracks which commits were compressed, what summaries were produced,
    and the token savings achieved.
    """

    compression_id: str
    original_tokens: int
    compressed_tokens: int
    source_commits: list[str]
    summary_commits: list[str]
    preserved_commits: list[str]
    compression_ratio: float
    new_head: str


@dataclass
class PendingCompression:
    """A compression that has been planned but not yet committed.

    Mutable: users can edit summaries before approving.
    The _commit_fn is set internally by Tract and should not be
    set by users directly.
    """

    summaries: list[str]
    source_commits: list[str]
    preserved_commits: list[str]
    original_tokens: int
    estimated_tokens: int
    _commit_fn: object | None = field(default=None, repr=False)

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
        return self._commit_fn(self)  # type: ignore[operator]


@dataclass(frozen=True)
class GCResult:
    """Result of a garbage collection operation.

    Tracks what was removed and how much space was freed.
    """

    commits_removed: int
    blobs_removed: int
    tokens_freed: int
    archives_removed: int
    duration_seconds: float


@dataclass(frozen=True)
class ReorderWarning:
    """Warning about potential issues when reordering commits during compression.

    Severity levels:
    - "structural": Affects commit graph integrity (e.g., edit before target)
    - "semantic": May affect meaning (e.g., response chain break)
    """

    warning_type: str
    commit_hash: str
    description: str
    severity: str
