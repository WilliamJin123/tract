"""Merge domain models for Trace.

Defines data models for merge operations: conflict information,
merge results, and the review/commit flow.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from tract.models.commit import CommitInfo


class ConflictInfo(BaseModel):
    """Rich context for a single merge conflict, passed to the resolver.

    Contains all information needed for either human review or LLM-mediated
    conflict resolution: both versions, the common ancestor, and surrounding
    branch history.
    """

    conflict_type: Literal["both_edit", "skip_vs_edit", "edit_plus_append"]
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

    # Internal: parent hashes for the merge commit
    _source_tip_hash: Optional[str] = None
    _target_tip_hash: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def edit_resolution(self, target_hash: str, new_content: str) -> None:
        """Edit a conflict resolution before committing.

        Args:
            target_hash: The commit hash of the conflicted target.
            new_content: The new resolved content text.
        """
        self.resolutions[target_hash] = new_content
