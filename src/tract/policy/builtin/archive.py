"""ArchivePolicy -- detect stale branches and propose archiving.

Fires on ``compile`` trigger.  Checks whether the current branch has
few commits and has been inactive for a configurable number of days.
If so, proposes archiving the branch by renaming it to an archive prefix.

This helps keep the branch list clean by archiving inactive side branches
that are no longer being actively developed.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from tract.models.policy import PolicyAction
from tract.policy.protocols import Policy

if TYPE_CHECKING:
    from tract.hooks.policy import PendingPolicy
    from tract.tract import Tract


class ArchivePolicy(Policy):
    """Detect stale branches and propose archiving to archive/ prefix.

    Constructor Args:
        stale_days: Number of days of inactivity before a branch is stale (default 7).
        min_commits: Maximum number of commits for a branch to be considered
            "small enough to archive" (default 3).
        archive_prefix: Prefix for archived branches (default "archive/").
    """

    def __init__(
        self,
        stale_days: int = 7,
        min_commits: int = 3,
        archive_prefix: str = "archive/",
    ) -> None:
        self._stale_days = stale_days
        self._min_commits = min_commits
        self._archive_prefix = archive_prefix

    @property
    def name(self) -> str:
        return "auto-archive"

    @property
    def priority(self) -> int:
        return 500

    @property
    def trigger(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        """Check if current branch is stale and small enough to archive."""
        current_branch = tract.current_branch
        if current_branch is None or current_branch == "main":
            return None

        # Already archived
        if current_branch.startswith(self._archive_prefix):
            return None

        # Get recent commits
        commits = tract.log(limit=self._min_commits + 1)
        if not commits:
            return None

        # Check staleness: age of most recent commit
        most_recent = commits[0].created_at
        now = datetime.now()
        # Handle timezone-aware datetimes
        if most_recent.tzinfo is not None:
            most_recent = most_recent.replace(tzinfo=None)
        age_days = (now - most_recent).days

        # Branch qualifies if it has few commits AND is stale
        if len(commits) <= self._min_commits and age_days >= self._stale_days:
            archive_name = f"{self._archive_prefix}{current_branch}"
            return PolicyAction(
                action_type="archive",
                params={
                    "archive_name": archive_name,
                    "source": current_branch,
                },
                reason=(
                    f"Branch '{current_branch}' has {len(commits)} commits "
                    f"and has been inactive for {age_days} days "
                    f"(threshold: {self._stale_days} days)"
                ),
                autonomy="collaborative",
            )

        return None

    # ------------------------------------------------------------------
    # Hook integration
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingPolicy) -> None:
        """Auto-approve archive proposals."""
        pending.approve()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Serialize policy configuration to a dict."""
        return {
            "name": self.name,
            "stale_days": self._stale_days,
            "min_commits": self._min_commits,
            "archive_prefix": self._archive_prefix,
            "enabled": True,
        }

    @classmethod
    def from_config(cls, config: dict) -> ArchivePolicy:
        """Deserialize an ArchivePolicy from a config dict."""
        return cls(
            stale_days=config.get("stale_days", 7),
            min_commits=config.get("min_commits", 3),
            archive_prefix=config.get("archive_prefix", "archive/"),
        )
