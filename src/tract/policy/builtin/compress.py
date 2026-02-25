"""CompressPolicy -- auto-compress when token usage exceeds threshold.

Fires on ``compile`` trigger when the current token count exceeds a
configurable percentage of the token budget.  Returns a collaborative
PolicyAction so the user can approve or reject compression.

Preserves pinned commits automatically (handled by the underlying
``compress()`` implementation).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tract.models.policy import PolicyAction
from tract.policy.protocols import Policy

if TYPE_CHECKING:
    from tract.hooks.policy import PendingPolicy
    from tract.hooks.validation import HookRejection
    from tract.tract import Tract

logger = logging.getLogger(__name__)


class CompressPolicy(Policy):
    """Auto-compress when token usage exceeds threshold percentage of budget.

    Constructor Args:
        threshold: Fraction of budget that triggers compression (default 0.9 = 90%).
        summary_content: If set, passed as ``content`` to ``compress()`` so
            the summary is deterministic (avoids LLM dependency in tests).
    """

    def __init__(
        self,
        threshold: float = 0.9,
        summary_content: str | None = None,
    ) -> None:
        self._threshold = threshold
        self._summary_content = summary_content

    @property
    def name(self) -> str:
        return "auto-compress"

    @property
    def priority(self) -> int:
        return 200

    @property
    def trigger(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        """Check if token usage exceeds threshold and propose compression."""
        # No budget configured -- nothing to check
        budget = tract.config.token_budget
        if budget is None or budget.max_tokens is None:
            return None

        max_tokens = budget.max_tokens
        status = tract.status()
        token_count = status.token_count

        if token_count >= max_tokens * self._threshold:
            params: dict = {}
            if self._summary_content is not None:
                params["content"] = self._summary_content
            return PolicyAction(
                action_type="compress",
                params=params,
                reason=(
                    f"Token usage {token_count}/{max_tokens} "
                    f"exceeds {self._threshold:.0%} threshold"
                ),
                autonomy="collaborative",
            )

        return None

    # ------------------------------------------------------------------
    # Hook integration
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingPolicy) -> None:
        """Auto-approve compression proposals by default."""
        pending.approve()

    def on_rejection(self, rejection: HookRejection) -> None:
        """Log rejection and increase threshold slightly to avoid rapid re-firing."""
        logger.info(
            "CompressPolicy action rejected: %s", rejection.reason
        )

    def on_success(self, result: object) -> None:
        """Log successful compression."""
        logger.debug("CompressPolicy action succeeded: %s", result)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Serialize policy configuration to a dict."""
        cfg: dict = {
            "name": self.name,
            "threshold": self._threshold,
            "enabled": True,
        }
        if self._summary_content is not None:
            cfg["summary_content"] = self._summary_content
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> CompressPolicy:
        """Deserialize a CompressPolicy from a config dict."""
        return cls(
            threshold=config.get("threshold", 0.9),
            summary_content=config.get("summary_content"),
        )
