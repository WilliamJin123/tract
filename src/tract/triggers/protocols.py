"""Trigger ABC -- base class for all triggers.

Users implement this to create custom triggers (e.g., auto-compress
when tokens exceed threshold, pin important commits, archive stale branches).

Built-in CompressTrigger, PinTrigger, etc. also implement this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tract.models.trigger import TriggerAction

if TYPE_CHECKING:
    from tract.hooks.trigger import PendingTrigger
    from tract.hooks.validation import HookRejection
    from tract.tract import Tract


class Trigger(ABC):
    """Abstract base class for all triggers.

    Users implement this to create custom triggers.
    Built-in CompressTrigger, PinTrigger, etc. also implement this.

    Example::

        class AutoCompress(Trigger):
            @property
            def name(self) -> str:
                return "auto-compress"

            def evaluate(self, tract: Tract) -> TriggerAction | None:
                compiled = tract.compile()
                if compiled.token_count > 8000:
                    return TriggerAction(
                        action_type="compress",
                        params={"target_tokens": 4000},
                        reason="Token count exceeded 8000",
                        autonomy="collaborative",
                    )
                return None
    """

    @abstractmethod
    def evaluate(self, tract: Tract) -> TriggerAction | None:
        """Evaluate whether this trigger should fire.

        Must be FAST -- check thresholds only, no LLM calls.
        Returns TriggerAction if the trigger wants to fire, None if conditions not met.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this trigger (e.g., 'auto-compress')."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority. Lower runs first. Default 100."""
        return 100

    @property
    def fires_on(self) -> str:
        """When this trigger evaluates: 'compile' or 'commit'. Default 'compile'."""
        return "compile"

    # ------------------------------------------------------------------
    # Hook integration (Phase 2)
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingTrigger) -> None:
        """Override for trigger-specific review logic.

        Called when a collaborative trigger action is routed through the
        hook system and no user hook is registered for "trigger". This
        provides a trigger-specific default behavior (e.g., auto-approve
        with specific conditions, add cooldown logic).

        Overridden by user hooks (``t.on("trigger", handler)``).

        Args:
            pending: The PendingTrigger to approve, reject, or modify.
        """
        pending.approve()

    def on_rejection(self, rejection: HookRejection) -> None:
        """Adapt behavior after a hook rejection.

        Called when the hook handler (or default_handler) rejects the
        trigger action. Triggers can use this for cooldown, parameter
        adjustment, or other adaptive behavior.

        Args:
            rejection: Structured rejection information.
        """
        pass

    def on_success(self, result: object) -> None:
        """Learn from successful execution.

        Called when the trigger action is approved and executed
        successfully. Triggers can use this for logging, statistics,
        or adjusting future behavior.

        Args:
            result: The result of executing the action.
        """
        pass
