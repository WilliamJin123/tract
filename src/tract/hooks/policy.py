"""PendingPolicy -- hook object for policy-triggered actions.

Stub for Phase 1. Full wiring to PolicyEvaluator happens in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.pending import Pending

if TYPE_CHECKING:
    from tract.tract import Tract


@dataclass
class PendingPolicy(Pending):
    """A policy-triggered action that requires approval.

    Mutable: handlers can modify action parameters before approving
    or rejecting.

    Fields:
        policy_name: Name of the policy that triggered this action.
        action_type: Type of action being proposed (e.g. "compress", "branch").
        action_params: Parameters for the action (mutable by handler).
        reason: Human-readable explanation of why the policy triggered.
    """

    policy_name: str = ""
    """Name of the policy that triggered this action."""

    action_type: str = ""
    """Type of action being proposed (e.g. "compress", "branch", "pin")."""

    action_params: dict = field(default_factory=dict)
    """Parameters for the proposed action. Mutable by handler."""

    reason: str = ""
    """Why the policy triggered this action."""

    # -- Whitelist for agent dispatch -----------------------------------

    _public_actions: set[str] = field(
        default_factory=lambda: {"approve", "reject", "modify_params"},
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = "policy"

    # -- Core methods ---------------------------------------------------

    def approve(self) -> Any:
        """Approve and execute the policy-proposed action with current params.

        Returns:
            Result of executing the proposed action.

        Raises:
            RuntimeError: If status is not "pending" or no execute function is set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                "Cannot approve: no execute function set. "
                "This PendingPolicy was not created by the policy engine."
            )
        self.status = "approved"
        return self._execute_fn(self)

    def reject(self, reason: str = "") -> None:
        """Reject the policy-proposed action.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = "rejected"
        self.rejection_reason = reason

    # -- Editing methods ------------------------------------------------

    def modify_params(self, params: dict) -> None:
        """Merge parameter updates into action_params before approval.

        Args:
            params: Dict of parameter updates to merge in.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.action_params.update(params)

    # -- Display --------------------------------------------------------
    # Inherits Rich-based pprint() from Pending base class.
