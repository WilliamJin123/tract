"""RebaseTrigger -- evaluate branch divergence and propose rebase.

Fires on ``commit`` trigger.  Checks whether the current branch has
diverged from a target branch by more than a configurable number of
commits (or tokens).  When the threshold is exceeded, proposes a
rebase in collaborative mode so the user can approve or reject.

Skips evaluation when on the target branch itself or in detached HEAD
state, since rebase is only meaningful for feature/topic branches.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tract.models.trigger import TriggerAction
from tract.triggers.protocols import Trigger

if TYPE_CHECKING:
    from tract.hooks.trigger import PendingTrigger
    from tract.hooks.validation import HookRejection
    from tract.tract import Tract

logger = logging.getLogger(__name__)


class RebaseTrigger(Trigger):
    """Evaluate branch divergence from a target branch and propose rebase.

    Constructor Args:
        target_branch: Branch to compare against (default "main").
        divergence_commits: Max commits behind target before triggering
            (default 20).
        divergence_tokens: Optional token-based threshold.  If set, the
            trigger also fires when the token count of commits on the
            target branch (since the merge base) exceeds this value.
    """

    def __init__(
        self,
        target_branch: str = "main",
        divergence_commits: int = 20,
        divergence_tokens: int | None = None,
    ) -> None:
        self._target_branch = target_branch
        self._divergence_commits = divergence_commits
        self._divergence_tokens = divergence_tokens

    @property
    def name(self) -> str:
        return "auto-rebase"

    @property
    def priority(self) -> int:
        return 400

    @property
    def fires_on(self) -> str:
        return "commit"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        """Check if the current branch has diverged too far from the target."""
        # Skip in detached HEAD state
        current_branch = tract.current_branch
        if current_branch is None:
            return None

        # Skip if on the target branch itself
        if current_branch == self._target_branch:
            return None

        # Check that the target branch exists
        branches = tract.list_branches()
        target_exists = any(b.name == self._target_branch for b in branches)
        if not target_exists:
            return None

        # Compute divergence using the DAG utilities
        # We need the target branch tip and the current HEAD
        current_head = tract.head
        if current_head is None:
            return None

        # Get the target branch tip hash
        target_tip = None
        for b in branches:
            if b.name == self._target_branch:
                target_tip = b.commit_hash
                break

        if target_tip is None:
            return None

        # Find merge base between current HEAD and target tip
        from tract.operations.dag import find_merge_base, get_branch_commits

        merge_base = find_merge_base(
            tract._commit_repo,
            tract._parent_repo,
            current_head,
            target_tip,
        )

        # If merge base is the target tip, we are already up-to-date
        if merge_base == target_tip:
            return None

        # Count how many commits the target branch has since the merge base
        # (this is the "behind" count -- commits on target we don't have)
        if merge_base is not None:
            target_commits = get_branch_commits(
                tract._commit_repo, tract._parent_repo, target_tip, merge_base
            )
        else:
            # No common ancestor -- all target commits count as divergence
            target_commits = list(tract._commit_repo.get_ancestors(target_tip))

        behind_count = len(target_commits)

        # Check commit divergence threshold
        if behind_count >= self._divergence_commits:
            return TriggerAction(
                action_type="rebase",
                params={"target": self._target_branch},
                reason=(
                    f"Branch '{current_branch}' is {behind_count} commits "
                    f"behind '{self._target_branch}' "
                    f"(threshold: {self._divergence_commits})"
                ),
                autonomy="collaborative",
            )

        # Check token divergence threshold (optional)
        if self._divergence_tokens is not None and target_commits:
            token_divergence = sum(c.token_count for c in target_commits)
            if token_divergence >= self._divergence_tokens:
                return TriggerAction(
                    action_type="rebase",
                    params={"target": self._target_branch},
                    reason=(
                        f"Branch '{current_branch}' is {token_divergence} tokens "
                        f"behind '{self._target_branch}' "
                        f"(threshold: {self._divergence_tokens})"
                    ),
                    autonomy="collaborative",
                )

        return None

    # ------------------------------------------------------------------
    # Hook integration
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingTrigger) -> None:
        """Auto-approve rebase proposals by default."""
        pending.approve()

    def on_rejection(self, rejection: HookRejection) -> None:
        """Log rejection."""
        logger.info("RebaseTrigger action rejected: %s", rejection.reason)

    def on_success(self, result: object) -> None:
        """Log successful rebase."""
        logger.debug("RebaseTrigger action succeeded: %s", result)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Serialize trigger configuration to a dict."""
        cfg: dict = {
            "name": self.name,
            "target_branch": self._target_branch,
            "divergence_commits": self._divergence_commits,
            "enabled": True,
        }
        if self._divergence_tokens is not None:
            cfg["divergence_tokens"] = self._divergence_tokens
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> RebaseTrigger:
        """Deserialize a RebaseTrigger from a config dict."""
        return cls(
            target_branch=config.get("target_branch", "main"),
            divergence_commits=config.get("divergence_commits", 20),
            divergence_tokens=config.get("divergence_tokens"),
        )
