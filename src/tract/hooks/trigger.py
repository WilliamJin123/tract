"""PendingTrigger -- hook object for trigger-fired actions.

Stub for Phase 1. Full wiring to TriggerEvaluator happens in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.pending import Pending, PendingStatus

if TYPE_CHECKING:
    from tract.tract import Tract


@dataclass(repr=False)
class PendingTrigger(Pending):
    """A trigger-fired action that requires approval.

    Mutable: handlers can modify action parameters before approving
    or rejecting.

    Fields:
        trigger_name: Name of the trigger that triggered this action.
        action_type: Type of action being proposed (e.g. "compress", "branch").
        action_params: Parameters for the action (mutable by handler).
        reason: Human-readable explanation of why the trigger triggered.
    """

    trigger_name: str = ""
    """Name of the trigger that triggered this action."""

    action_type: str = ""
    """Type of action being proposed (e.g. "compress", "branch", "pin")."""

    action_params: dict = field(default_factory=dict)
    """Parameters for the proposed action. Mutable by handler."""

    reason: str = ""
    """Why the trigger triggered this action."""

    # -- Whitelist for agent dispatch -----------------------------------

    _public_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({"approve", "reject", "modify_params"}),
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = "trigger"

    # -- Core methods ---------------------------------------------------

    def approve(self) -> Any:
        """Approve and execute the trigger-proposed action with current params.

        Returns:
            Result of executing the proposed action.

        Raises:
            RuntimeError: If status is not "pending" or no execute function is set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                "Cannot approve: no execute function set. "
                "This PendingTrigger was not created by the trigger engine."
            )
        self.status = PendingStatus.APPROVED
        self._result = self._execute_fn(self)
        return self._result

    def reject(self, reason: str = "") -> None:
        """Reject the trigger-proposed action.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = PendingStatus.REJECTED
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

    def __repr__(self):
        status = self.status.value if hasattr(self.status, 'value') else str(self.status)
        return f"<PendingTrigger: {self.trigger_name}, {self.action_type}, {status}>"

    def _compact_detail(self) -> str:
        return f"{self.trigger_name} -> {self.action_type}"

    def _pprint_details(self, console, *, verbose: bool = False) -> None:
        """Show trigger-specific details: trigger name, action, reason, params."""
        console.print(
            f"  Trigger: [bold]{self.trigger_name}[/bold] -> "
            f"[bold]{self.action_type}[/bold]"
        )
        if self.reason:
            console.print(f"  Reason: {self.reason}")
        if verbose and self.action_params:
            console.print("  [bold]Action params:[/bold]")
            for k, v in self.action_params.items():
                console.print(f"    {k}: {v!r}")
