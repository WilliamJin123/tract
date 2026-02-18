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
