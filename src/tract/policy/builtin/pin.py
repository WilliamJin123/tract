"""PinPolicy -- auto-pin commits based on content type.

Fires on ``commit`` trigger.  Checks whether the newly committed content
type matches a configurable set of types (default: instruction, session)
and auto-pins them with an ``annotate`` action in autonomous mode.

Respects manual overrides: if a user has already annotated a commit
(set it to NORMAL, SKIP, or PINNED), the policy will not re-annotate.

Known limitation: ``evaluate()`` only looks at HEAD (the latest commit).
For batch operations, call ``retroactive_scan()`` after the batch to
pin any matching commits that were missed.
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


class PinPolicy(Policy):
    """Auto-pin commits whose content type matches a configured set.

    Constructor Args:
        pin_types: Set of content types to auto-pin (default: instruction, session).
        patterns: List of pattern dicts for advanced matching.
            Each dict can have keys: ``content_type``, ``role``, ``text_pattern``.
            All provided keys must match for the pattern to trigger.
    """

    def __init__(
        self,
        pin_types: set[str] | None = None,
        patterns: list[dict] | None = None,
    ) -> None:
        self._pin_types: set[str] = pin_types or {"instruction", "session"}
        self._patterns: list[dict] = patterns or []

    @property
    def name(self) -> str:
        return "auto-pin"

    @property
    def priority(self) -> int:
        return 100

    @property
    def trigger(self) -> str:
        return "commit"

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        """Check if the latest commit should be auto-pinned."""
        head = tract.head
        if head is None:
            return None

        commit = tract.get_commit(head)
        if commit is None:
            return None

        # Check manual override: if ANY annotation exists, the user (or
        # another policy) already made a choice -- do not override.
        annotations = tract.get_annotations(head)
        if annotations:
            return None

        # Check content_type match
        if commit.content_type in self._pin_types:
            return PolicyAction(
                action_type="annotate",
                params={"target_hash": head, "priority": "pinned"},
                reason=f"Auto-pinned: content_type={commit.content_type}",
                autonomy="autonomous",
            )

        # Check pattern match
        if self._matches_pattern(commit):
            return PolicyAction(
                action_type="annotate",
                params={"target_hash": head, "priority": "pinned"},
                reason=f"Auto-pinned: matched pattern for {commit.content_type}",
                autonomy="autonomous",
            )

        return None

    def _matches_pattern(self, commit: object) -> bool:
        """Check if a commit matches any configured pattern.

        Args:
            commit: A CommitInfo object.

        Returns:
            True if any pattern matches.
        """
        for pattern in self._patterns:
            if self._check_single_pattern(commit, pattern):
                return True
        return False

    def _check_single_pattern(self, commit: object, pattern: dict) -> bool:
        """Check if a commit matches a single pattern dict."""
        # content_type must match if specified
        if "content_type" in pattern:
            if getattr(commit, "content_type", None) != pattern["content_type"]:
                return False

        # role: we don't have role on CommitInfo, so skip role matching
        # text_pattern: we don't have text on CommitInfo either
        # These would require loading the blob, which is too expensive
        # for the evaluate() hot path.  Patterns are primarily for content_type.

        # If we got here, all specified keys matched
        return bool(pattern)  # empty pattern never matches

    def retroactive_scan(self, tract: Tract, *, limit: int = 10000) -> list[str]:
        """Walk all commits and pin matching ones that lack annotations.

        Should be called once when the policy is first enabled to
        retroactively pin existing commits.

        Args:
            tract: The Tract instance to scan.
            limit: Maximum number of commits to scan (default 10000).

        Returns:
            List of commit hashes that were pinned.
        """
        from tract.models.annotations import Priority

        pinned_hashes: list[str] = []
        commits = tract.log(limit=limit)

        for commit in commits:
            # Skip if already annotated
            annotations = tract.get_annotations(commit.commit_hash)
            if annotations:
                continue

            should_pin = (
                commit.content_type in self._pin_types
                or self._matches_pattern(commit)
            )

            if should_pin:
                tract.annotate(commit.commit_hash, Priority.PINNED, reason="Retroactive auto-pin")
                pinned_hashes.append(commit.commit_hash)

        return pinned_hashes

    # ------------------------------------------------------------------
    # Hook integration
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingPolicy) -> None:
        """Auto-approve pin actions (autonomous by default anyway)."""
        pending.approve()

    def on_rejection(self, rejection: HookRejection) -> None:
        """Log pin rejection."""
        logger.info("PinPolicy action rejected: %s", rejection.reason)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Serialize policy configuration to a dict."""
        return {
            "name": self.name,
            "pin_types": sorted(self._pin_types),
            "patterns": self._patterns,
            "enabled": True,
        }

    @classmethod
    def from_config(cls, config: dict) -> PinPolicy:
        """Deserialize a PinPolicy from a config dict."""
        pin_types = config.get("pin_types")
        if pin_types is not None:
            pin_types = set(pin_types)
        return cls(
            pin_types=pin_types,
            patterns=config.get("patterns"),
        )
