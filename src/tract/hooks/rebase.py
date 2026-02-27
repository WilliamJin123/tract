"""PendingRebase -- hook object for rebase operations.

Wired to Tract.rebase() in Phase 2. Plan phase determines commits
to replay and checks safety, execute phase does the actual rebase.
Handlers can exclude specific commits from the replay plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.pending import Pending, PendingStatus

if TYPE_CHECKING:
    from tract.storage.schema import CommitRow
    from tract.tract import Tract


@dataclass(repr=False)
class PendingRebase(Pending):
    """A rebase operation that has been planned but not yet executed.

    Mutable: handlers can exclude specific commits from the replay plan
    before approving or rejecting.

    Fields:
        replay_plan: Ordered list of commit hashes to replay.
        target_base: Hash of the commit to rebase onto.
        warnings: List of warnings about potential issues with the rebase.
    """

    replay_plan: list[str] = field(default_factory=list)
    """Ordered list of commit hashes to replay onto target_base."""

    target_base: str = ""
    """Hash of the commit to rebase onto."""

    warnings: list[Any] = field(default_factory=list)
    """Warnings about potential issues (e.g. RebaseWarning instances)."""

    # -- Internal state for execute phase --------------------------------

    _commit_rows: list[CommitRow] | None = field(default=None, repr=False)
    """Full CommitRow objects for the execute phase (set by Tract.rebase())."""

    _current_branch: str | None = field(default=None, repr=False)
    """Current branch name (set by Tract.rebase())."""

    _current_tip: str | None = field(default=None, repr=False)
    """Original tip of current branch for rollback (set by Tract.rebase())."""

    # -- Whitelist for agent dispatch -----------------------------------

    _public_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({"approve", "reject", "exclude"}),
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = "rebase"

    # -- Core methods ---------------------------------------------------

    def approve(self) -> Any:
        """Execute the rebase, replaying commits onto the target base.

        Returns:
            RebaseResult with new commit hashes and updated HEAD.

        Raises:
            RuntimeError: If status is not "pending" or no execute function is set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                "Cannot approve: no execute function set. "
                "This PendingRebase was not created by Tract.rebase()."
            )
        self.status = PendingStatus.APPROVED
        self._result = self._execute_fn(self)
        return self._result

    def reject(self, reason: str = "") -> None:
        """Reject the rebase, leaving history unchanged.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = PendingStatus.REJECTED
        self.rejection_reason = reason

    # -- Editing methods ------------------------------------------------

    def exclude(self, commit_hash: str) -> None:
        """Exclude a commit from the replay plan, skipping it during rebase.

        Args:
            commit_hash: Hash of the commit to exclude from replay.

        Raises:
            ValueError: If commit_hash is not in replay_plan.
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        if commit_hash not in self.replay_plan:
            raise ValueError(
                f"Commit {commit_hash!r} is not in the replay plan."
            )
        self.replay_plan.remove(commit_hash)

    # -- Display --------------------------------------------------------

    def __repr__(self):
        status = self.status.value if hasattr(self.status, 'value') else str(self.status)
        base_short = self.target_base[:8] if self.target_base else "???"
        return f"<PendingRebase: {len(self.replay_plan)} commits onto {base_short}..., {status}>"
