"""Merge and rebase domain models for Trace.

Defines data models for merge, rebase, and import operations:
conflict information, merge results, rebase warnings, import issues,
and the review/commit flow.
"""

from __future__ import annotations

import enum
from typing import Literal, Optional

from pydantic import BaseModel

from tract.models.commit import CommitInfo


class ConflictType(str, enum.Enum):
    """Types of structural merge conflicts."""

    BOTH_EDIT = "both_edit"
    SKIP_VS_EDIT = "skip_vs_edit"
    EDIT_PLUS_APPEND = "edit_plus_append"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class ConflictInfo(BaseModel):
    """Rich context for a single merge conflict, passed to the resolver.

    Contains all information needed for either human review or LLM-mediated
    conflict resolution: both versions, the common ancestor, and surrounding
    branch history.
    """

    conflict_type: ConflictType
    commit_a: CommitInfo  # From current (target) branch
    commit_b: CommitInfo  # From source branch
    content_a_text: str = ""  # Pre-loaded content text from commit_a's blob
    content_b_text: str = ""  # Pre-loaded content text from commit_b's blob
    ancestor: Optional[CommitInfo] = None  # Common ancestor commit if applicable
    ancestor_content_text: Optional[str] = None  # Pre-loaded ancestor content text
    target_hash: Optional[str] = None  # The commit being conflicted over
    branch_a_commits: list[CommitInfo] = []
    branch_b_commits: list[CommitInfo] = []

    model_config = {"arbitrary_types_allowed": True}

    def to_marker_text(self) -> str:
        """Generate editable git-style conflict markers.

        Returns a string with ``<<<<<<<`` / ``=======`` / ``>>>>>>>`` markers
        wrapping both versions of the conflict.  If ``ancestor_content_text``
        is available, a ``|||||||`` section is included for 3-way display.
        """
        a_label = (self.commit_a.commit_hash or "ours")[:8]
        b_label = (self.commit_b.commit_hash or "theirs")[:8]

        lines: list[str] = []
        lines.append(f"<<<<<<< {a_label}")
        lines.append(self.content_a_text)
        if self.ancestor_content_text is not None:
            lines.append("||||||| ancestor")
            lines.append(self.ancestor_content_text)
        lines.append("=======")
        lines.append(self.content_b_text)
        lines.append(f">>>>>>> {b_label}")
        return "\n".join(lines)

    @staticmethod
    def parse_conflict_markers(text: str) -> str | None:
        """Parse text that may contain conflict markers.

        If both ``<<<<<<<`` and ``>>>>>>>`` markers remain, the conflict
        is considered unresolved and ``None`` is returned.  Otherwise
        the (stripped) text is treated as the user's resolution.
        """
        if "<<<<<<< " in text and ">>>>>>> " in text:
            return None
        return text.strip()

    def __repr__(self) -> str:
        th = (self.target_hash or "")[:8]
        return f"ConflictInfo({self.conflict_type.value} target={th})"

    def __str__(self) -> str:
        th = (self.target_hash or "")[:8]
        return f"{self.conflict_type.value} conflict at {th}"

    def pprint(self) -> None:
        """Pretty-print this conflict as a git-style diff."""
        from tract.formatting import pprint_conflict_info

        pprint_conflict_info(self)


class MergeResult(BaseModel):
    """Result of a merge operation. Returned for review before commit.

    For fast-forward and clean merges, ``committed`` is True immediately.
    For conflict merges, the caller reviews conflicts, optionally edits
    resolutions, and finalizes with ``Tract.commit_merge(result)``.
    """

    merge_type: Literal["fast_forward", "clean", "conflict", "semantic"]
    source_branch: str
    target_branch: str
    merge_base_hash: Optional[str] = None
    conflicts: list[ConflictInfo] = []
    resolutions: dict[str, str] = {}  # target_hash -> resolved content text
    resolution_reasoning: dict[str, str] = {}  # target_hash -> LLM reasoning
    auto_merged_content: list[CommitInfo] = []  # Commits that auto-merged cleanly
    generation_configs: dict[str, dict] = {}  # target_hash -> gen config from resolver
    committed: bool = False  # Set True after commit_merge()
    merge_commit_hash: Optional[str] = None  # Set after commit_merge()

    # Parent hashes captured at merge time (for commit_merge)
    source_tip_hash: Optional[str] = None
    target_tip_hash: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def edit_resolution(self, target_hash: str, new_content: str) -> None:
        """Edit a conflict resolution before committing.

        Args:
            target_hash: The commit hash of the conflicted target.
            new_content: The new resolved content text.
        """
        self.resolutions[target_hash] = new_content

    def __repr__(self) -> str:
        n = len(self.conflicts)
        status = "committed" if self.committed else "pending"
        return (
            f"MergeResult({self.merge_type} "
            f"{self.source_branch}->{self.target_branch} "
            f"conflicts={n} {status})"
        )

    def __str__(self) -> str:
        n = len(self.conflicts)
        status = "committed" if self.committed else "pending"
        return (
            f"{self.merge_type} merge "
            f"{self.source_branch}->{self.target_branch} "
            f"({n} conflicts, {status})"
        )

    def pprint(self) -> None:
        """Pretty-print this merge result summary."""
        from tract.formatting import pprint_merge_result

        pprint_merge_result(self)


# ---------------------------------------------------------------------------
# Rebase and import models
# ---------------------------------------------------------------------------


class RebaseWarning(BaseModel):
    """Semantic safety issue detected during rebase."""

    warning_type: Literal["reorder_changes_meaning", "edit_target_missing"]
    commit: CommitInfo  # The commit being replayed
    new_base: Optional[CommitInfo] = None  # The new parent context
    original_base: Optional[CommitInfo] = None
    context_before: list[CommitInfo] = []  # Commits before in original order
    context_after: list[CommitInfo] = []  # Commits after in new order
    description: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def __repr__(self) -> str:
        return f"RebaseWarning({self.warning_type} commit={self.commit!s})"

    def __str__(self) -> str:
        if self.description:
            return f"{self.warning_type}: {self.description}"
        return self.warning_type


class ImportIssue(BaseModel):
    """Issue detected during import-commit."""

    issue_type: Literal["edit_target_missing", "context_dependency"]
    commit: CommitInfo  # The commit being imported
    target_branch_head: Optional[CommitInfo] = None
    missing_target: Optional[str] = None  # The edit_target hash that doesn't exist on target
    description: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def __repr__(self) -> str:
        return f"ImportIssue({self.issue_type} commit={self.commit!s})"

    def __str__(self) -> str:
        if self.description:
            return f"{self.issue_type}: {self.description}"
        return self.issue_type


class RebaseResult(BaseModel):
    """Result of a rebase operation."""

    replayed_commits: list[CommitInfo] = []  # New commits created during replay
    original_commits: list[CommitInfo] = []  # Original commits that were replayed
    warnings: list[RebaseWarning] = []
    resolutions: dict[str, str] = {}  # commit_hash -> resolved content
    new_head: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def __repr__(self) -> str:
        n = len(self.replayed_commits)
        w = len(self.warnings)
        return f"RebaseResult(replayed={n} warnings={w})"

    def __str__(self) -> str:
        n = len(self.replayed_commits)
        head = (self.new_head or "")[:8]
        return f"rebase: {n} commits replayed, head={head}"


class ImportResult(BaseModel):
    """Result of an import-commit operation."""

    original_commit: Optional[CommitInfo] = None
    new_commit: Optional[CommitInfo] = None
    issues: list[ImportIssue] = []
    resolutions: dict[str, str] = {}

    model_config = {"arbitrary_types_allowed": True}

    def __repr__(self) -> str:
        orig = str(self.original_commit) if self.original_commit else "None"
        n = len(self.issues)
        return f"ImportResult(original={orig} issues={n})"

    def __str__(self) -> str:
        orig = str(self.original_commit) if self.original_commit else "?"
        new = str(self.new_commit) if self.new_commit else "?"
        return f"import {orig} -> {new}"
