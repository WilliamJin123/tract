"""Domain models for the compression subsystem.

Provides data classes for compression results,
garbage collection results, and reorder warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


@dataclass(frozen=True)
class ToolCompactResult:
    """Result of a tool-call compaction operation.

    Unlike :class:`CompressResult` (which collapses commits into summaries),
    tool compaction uses EDIT commits to shorten each tool result in-place,
    preserving commit structure, roles, and metadata.
    """

    edit_commits: tuple[str, ...]
    """Hashes of the new EDIT commits (one per compacted result)."""

    source_commits: tuple[str, ...]
    """Hashes of the original tool result commits that were compacted."""

    original_tokens: int
    """Total tokens of the original tool results before compaction."""

    compacted_tokens: int
    """Total tokens of the compacted tool results."""

    tool_names: tuple[str, ...]
    """Unique tool names involved in the compacted turns."""

    turn_count: int
    """Number of tool turns that were compacted."""


@dataclass(frozen=True)
class ToolDropResult:
    """Result of dropping failed tool turns from context.

    Returned by :meth:`Tract.drop_failed_tool_turns`.  Tracks how many
    turns were dropped and how many tokens were freed.
    """

    turns_dropped: int
    """Number of tool turns that were marked SKIP."""

    commits_skipped: int
    """Total commits annotated with SKIP (calls + results)."""

    tokens_freed: int
    """Approximate tokens freed by skipping the dropped turns."""

    tool_names: tuple[str, ...]
    """Unique tool names involved in the dropped turns."""


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
