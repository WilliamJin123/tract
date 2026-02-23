"""History operations: status computation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo


@dataclass(frozen=True)
class StatusInfo:
    """Current tract status information returned by Tract.status().

    Attributes:
        head_hash: Current HEAD commit hash, or None if no commits.
        branch_name: Current branch name, or None if detached.
        is_detached: Whether HEAD is in detached state.
        commit_count: Total commits in compiled chain from HEAD.
        token_count: Compiled token count (from compile()).
        token_budget_max: Maximum token budget, or None if no budget configured.
        token_source: Token source identifier (e.g. "tiktoken:cl100k_base").
        recent_commits: Last 3 commits in reverse chronological order.
    """

    head_hash: str | None
    branch_name: str | None  # None if detached
    is_detached: bool
    commit_count: int  # total commits in chain from HEAD
    token_count: int  # compiled token count (from compile())
    token_budget_max: int | None  # None if no budget configured
    token_source: str
    recent_commits: list[CommitInfo] = field(default_factory=list)  # last 3 commits

    def __str__(self) -> str:
        head = self.head_hash[:8] if self.head_hash else "None"
        branch = self.branch_name or "detached"
        budget_str = ""
        if self.token_budget_max:
            pct = self.token_count / self.token_budget_max * 100
            budget_str = f"/{self.token_budget_max} ({pct:.0f}%)"
        return f"{branch} @ {head} | {self.commit_count} commits | {self.token_count}{budget_str} tokens"

    def pprint(self, *, max_chars: int | None = None) -> None:
        """Pretty-print this status using rich formatting.

        Args:
            max_chars: Max display characters before truncation.
                None (default) means no limit.
        """
        from tract.formatting import pprint_status_info

        pprint_status_info(self, max_chars=max_chars)
