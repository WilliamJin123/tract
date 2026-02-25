"""Policy ABC -- base class for all policies.

Users implement this to create custom policies (e.g., auto-compress
when tokens exceed threshold, pin important commits, archive stale branches).

Built-in CompressPolicy, PinPolicy, etc. also implement this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tract.models.policy import PolicyAction

if TYPE_CHECKING:
    from tract.hooks.policy import PendingPolicy
    from tract.hooks.validation import HookRejection
    from tract.tract import Tract


class Policy(ABC):
    """Abstract base class for all policies.

    Users implement this to create custom policies.
    Built-in CompressPolicy, PinPolicy, etc. also implement this.

    Example::

        class AutoCompress(Policy):
            @property
            def name(self) -> str:
                return "auto-compress"

            def evaluate(self, tract: Tract) -> PolicyAction | None:
                compiled = tract.compile()
                if compiled.token_count > 8000:
                    return PolicyAction(
                        action_type="compress",
                        params={"target_tokens": 4000},
                        reason="Token count exceeded 8000",
                        autonomy="collaborative",
                    )
                return None
    """

    @abstractmethod
    def evaluate(self, tract: Tract) -> PolicyAction | None:
        """Evaluate whether this policy should fire.

        Must be FAST -- check thresholds only, no LLM calls.
        Returns PolicyAction if the policy wants to fire, None if conditions not met.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this policy (e.g., 'auto-compress')."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority. Lower runs first. Default 100."""
        return 100

    @property
    def trigger(self) -> str:
        """When this policy evaluates: 'compile' or 'commit'. Default 'compile'."""
        return "compile"

    # ------------------------------------------------------------------
    # Hook integration (Phase 2)
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingPolicy) -> None:
        """Override for policy-specific review logic.

        Called when a collaborative policy action is routed through the
        hook system and no user hook is registered for "policy". This
        provides a policy-specific default behavior (e.g., auto-approve
        with specific conditions, add cooldown logic).

        Overridden by user hooks (``t.on("policy", handler)``).

        Args:
            pending: The PendingPolicy to approve, reject, or modify.
        """
        pending.approve()

    def on_rejection(self, rejection: HookRejection) -> None:
        """Adapt behavior after a hook rejection.

        Called when the hook handler (or default_handler) rejects the
        policy action. Policies can use this for cooldown, parameter
        adjustment, or other adaptive behavior.

        Args:
            rejection: Structured rejection information.
        """
        pass

    def on_success(self, result: object) -> None:
        """Learn from successful execution.

        Called when the policy action is approved and executed
        successfully. Policies can use this for logging, statistics,
        or adjusting future behavior.

        Args:
            result: The result of executing the action.
        """
        pass
