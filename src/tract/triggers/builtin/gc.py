"""GCTrigger -- evaluate dead commit count and propose garbage collection.

Fires on ``commit`` trigger.  Counts unreachable commits (those not
reachable from any branch tip) and proposes GC when the count exceeds
a configurable threshold.  Optionally also checks the database file
size against a storage threshold.

This helps keep the storage footprint manageable by periodically
cleaning up orphaned commits left by compression and rebase operations.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from tract.models.trigger import TriggerAction
from tract.triggers.protocols import Trigger

if TYPE_CHECKING:
    from tract.hooks.trigger import PendingTrigger
    from tract.hooks.validation import HookRejection
    from tract.tract import Tract

logger = logging.getLogger(__name__)


class GCTrigger(Trigger):
    """Evaluate dead commit count or storage size and propose GC.

    Constructor Args:
        max_dead_commits: Maximum number of unreachable commits before
            triggering GC (default 50).
        max_storage_mb: Optional storage size threshold in MB.  If set,
            GC is also triggered when the database file exceeds this size.
    """

    def __init__(
        self,
        max_dead_commits: int = 50,
        max_storage_mb: float | None = None,
    ) -> None:
        self._max_dead_commits = max_dead_commits
        self._max_storage_mb = max_storage_mb

    @property
    def name(self) -> str:
        return "auto-gc"

    @property
    def priority(self) -> int:
        return 450

    @property
    def fires_on(self) -> str:
        return "commit"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        """Check if dead commit count or storage size exceeds thresholds."""
        # Count unreachable commits using the same logic as plan_gc
        dead_count = self._count_dead_commits(tract)

        if dead_count >= self._max_dead_commits:
            return TriggerAction(
                action_type="gc",
                params={"retention": "default"},
                reason=(
                    f"Dead commit count {dead_count} exceeds threshold "
                    f"{self._max_dead_commits}"
                ),
                autonomy="collaborative",
            )

        # Check storage size threshold (optional)
        if self._max_storage_mb is not None:
            storage_mb = self._get_storage_mb(tract)
            if storage_mb is not None and storage_mb >= self._max_storage_mb:
                return TriggerAction(
                    action_type="gc",
                    params={"retention": "default"},
                    reason=(
                        f"Storage size {storage_mb:.1f} MB exceeds threshold "
                        f"{self._max_storage_mb:.1f} MB"
                    ),
                    autonomy="collaborative",
                )

        return None

    def _count_dead_commits(self, tract: Tract) -> int:
        """Count commits not reachable from any branch tip.

        Uses the same reachability logic as plan_gc() in the
        compression operations module.
        """
        from tract.operations.dag import get_all_ancestors

        # Get all branch tips
        branches = tract.list_branches()
        reachable: set[str] = set()

        for branch_info in branches:
            tip = branch_info.commit_hash
            if tip:
                ancestors = get_all_ancestors(
                    tip, tract._commit_repo, tract._parent_repo,
                    stop_at=reachable,
                )
                reachable.update(ancestors)

        # Also include detached HEAD if applicable
        if tract.is_detached and tract.head:
            ancestors = get_all_ancestors(
                tract.head, tract._commit_repo, tract._parent_repo,
                stop_at=reachable,
            )
            reachable.update(ancestors)

        # Count all commits and subtract reachable
        all_commits = tract._commit_repo.get_all(tract.tract_id)
        dead_count = sum(1 for c in all_commits if c.commit_hash not in reachable)
        return dead_count

    def _get_storage_mb(self, tract: Tract) -> float | None:
        """Get the database file size in MB, or None if not file-backed."""
        try:
            # Try to get the database URL from the session/engine
            engine = tract._session.get_bind()
            url_str = str(engine.url)
            # SQLite file-backed DBs have paths like sqlite:///path/to/db.sqlite
            if "sqlite" in url_str and ":memory:" not in url_str:
                # Extract path from URL
                path = url_str.replace("sqlite:///", "").replace("sqlite://", "")
                if path and os.path.exists(path):
                    size_bytes = os.path.getsize(path)
                    return size_bytes / (1024 * 1024)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Hook integration
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingTrigger) -> None:
        """Auto-approve GC proposals by default."""
        pending.approve()

    def on_rejection(self, rejection: HookRejection) -> None:
        """Log rejection."""
        logger.info("GCTrigger action rejected: %s", rejection.reason)

    def on_success(self, result: object) -> None:
        """Log successful GC."""
        logger.debug("GCTrigger action succeeded: %s", result)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Serialize trigger configuration to a dict."""
        cfg: dict = {
            "name": self.name,
            "max_dead_commits": self._max_dead_commits,
            "enabled": True,
        }
        if self._max_storage_mb is not None:
            cfg["max_storage_mb"] = self._max_storage_mb
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> GCTrigger:
        """Deserialize a GCTrigger from a config dict."""
        return cls(
            max_dead_commits=config.get("max_dead_commits", 50),
            max_storage_mb=config.get("max_storage_mb"),
        )
