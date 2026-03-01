"""MergeTrigger -- evaluate branch completion criteria and propose merge.

Fires on ``commit`` trigger.  Checks whether the current branch has
accumulated enough commits and has been idle long enough to be
considered "complete".  When both conditions are met, proposes a merge
into the target branch in collaborative mode.

Skips evaluation when on the target branch itself or in detached HEAD
state, since merge proposals only make sense for feature/topic branches.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tract.models.trigger import TriggerAction
from tract.triggers.protocols import Trigger

if TYPE_CHECKING:
    from tract.hooks.trigger import PendingTrigger
    from tract.hooks.validation import HookRejection
    from tract.tract import Tract

logger = logging.getLogger(__name__)


class MergeTrigger(Trigger):
    """Evaluate branch completion criteria and propose merge.

    Constructor Args:
        target_branch: Branch to merge INTO (default "main").
        completion_commits: Minimum number of commits on the branch
            before considering it for merge (default 5).
        idle_seconds: Seconds since last commit before the branch is
            considered "done" (default 300 = 5 minutes).
    """

    def __init__(
        self,
        target_branch: str = "main",
        completion_commits: int = 5,
        idle_seconds: int = 300,
    ) -> None:
        self._target_branch = target_branch
        self._completion_commits = completion_commits
        self._idle_seconds = idle_seconds

    @property
    def name(self) -> str:
        return "auto-merge"

    @property
    def priority(self) -> int:
        return 350

    @property
    def fires_on(self) -> str:
        return "commit"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        """Check if branch has enough commits and has been idle."""
        # Skip in detached HEAD state
        current_branch = tract.current_branch
        if current_branch is None:
            return None

        # Skip if on the target branch itself
        if current_branch == self._target_branch:
            return None

        # Get recent commits on this branch
        commits = tract.log(limit=self._completion_commits + 1)
        if len(commits) < self._completion_commits:
            return None

        # Check idle time: age of most recent commit
        if not commits:
            return None

        most_recent = commits[0].created_at
        now = datetime.now(timezone.utc)
        # Normalize to UTC-aware for comparison
        if most_recent.tzinfo is None:
            most_recent = most_recent.replace(tzinfo=timezone.utc)
        idle_seconds = (now - most_recent).total_seconds()

        if idle_seconds >= self._idle_seconds:
            return TriggerAction(
                action_type="merge",
                params={
                    "source": current_branch,
                    "target": self._target_branch,
                },
                reason=(
                    f"Branch '{current_branch}' has {len(commits)} commits "
                    f"and has been idle for {idle_seconds:.0f}s "
                    f"(threshold: {self._idle_seconds}s)"
                ),
                autonomy="collaborative",
            )

        return None

    # ------------------------------------------------------------------
    # Hook integration
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingTrigger) -> None:
        """Auto-approve merge proposals by default."""
        pending.approve()

    def on_rejection(self, rejection: HookRejection) -> None:
        """Log rejection."""
        logger.info("MergeTrigger action rejected: %s", rejection.reason)

    def on_success(self, result: object) -> None:
        """Log successful merge."""
        logger.debug("MergeTrigger action succeeded: %s", result)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Serialize trigger configuration to a dict."""
        return {
            "name": self.name,
            "target_branch": self._target_branch,
            "completion_commits": self._completion_commits,
            "idle_seconds": self._idle_seconds,
            "enabled": True,
        }

    @classmethod
    def from_config(cls, config: dict) -> MergeTrigger:
        """Deserialize a MergeTrigger from a config dict."""
        return cls(
            target_branch=config.get("target_branch", "main"),
            completion_commits=config.get("completion_commits", 5),
            idle_seconds=config.get("idle_seconds", 300),
        )
