"""Domain models for the policy engine subsystem.

Provides data classes for policy actions, proposals, evaluation results,
and log entries used by the policy evaluation pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PolicyAction:
    """An action that a policy wants to perform.

    Immutable: once a policy determines an action, it doesn't change.
    The autonomy level determines whether it executes immediately or
    requires approval.
    """

    action_type: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    autonomy: str = "collaborative"  # "autonomous", "collaborative", "supervised"


@dataclass
class PolicyProposal:
    """A proposed policy action awaiting approval.

    Mutable: status changes as the proposal is approved/rejected.
    Follows the PendingCompression pattern -- _execute_fn is set
    internally and should not be set by users directly.
    """

    proposal_id: str
    policy_name: str
    action: PolicyAction
    created_at: datetime
    status: str = "pending"  # "pending", "approved", "rejected", "expired", "executed"
    _execute_fn: Callable[[PolicyProposal], object] | None = field(
        default=None, repr=False
    )

    def approve(self) -> object:
        """Approve and execute the proposed action.

        Returns:
            Result of executing the action.

        Raises:
            PolicyExecutionError: If no execute function has been set.
        """
        from tract.exceptions import PolicyExecutionError

        if self._execute_fn is None:
            raise PolicyExecutionError(
                "Cannot approve: no execute function set. "
                "This PolicyProposal was not created by the policy engine."
            )
        self.status = "approved"
        return self._execute_fn(self)

    def reject(self, reason: str = "") -> None:
        """Reject the proposed action.

        Args:
            reason: Optional explanation for why the proposal was rejected.
        """
        self.status = "rejected"


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a single policy against current state.

    Immutable: once evaluation completes, the result is final.
    """

    policy_name: str
    triggered: bool
    action: PolicyAction | None = None
    outcome: str = "skipped"  # "executed", "proposed", "skipped", "error"
    error: str | None = None
    commit_hash: str | None = None


@dataclass(frozen=True)
class PolicyLogEntry:
    """A log entry recording a policy evaluation for audit purposes.

    Maps 1:1 with PolicyLogRow in the database.
    """

    id: int
    tract_id: str
    policy_name: str
    trigger: str  # "compile" or "commit"
    action_type: str | None
    reason: str | None
    outcome: str  # "executed", "proposed", "skipped", "error"
    commit_hash: str | None
    error_message: str | None
    created_at: datetime
