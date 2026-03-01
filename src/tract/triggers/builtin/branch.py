"""BranchTrigger -- auto-detect content type tangents and propose branching.

Fires on ``commit`` trigger.  Examines recent commits for rapid content
type transitions (e.g., switching between dialogue and artifact rapidly),
which may indicate the conversation has taken a tangent.

When a tangent is detected, proposes creating a new branch in
collaborative mode so the user can approve or reject.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from tract.models.trigger import TriggerAction
from tract.triggers.protocols import Trigger

if TYPE_CHECKING:
    from tract.hooks.trigger import PendingTrigger
    from tract.tract import Tract


class BranchTrigger(Trigger):
    """Detect content type switching patterns and propose tangent branches.

    Constructor Args:
        content_type_window: Number of recent commits to examine (default 5).
        switch_threshold: Number of content type transitions to trigger (default 4).
        ignore_transitions: Set of (from_type, to_type) tuples to ignore
            (default: dialogue/tool_io transitions, which are normal back-and-forth).
    """

    def __init__(
        self,
        content_type_window: int = 5,
        switch_threshold: int = 4,
        ignore_transitions: set[tuple[str, str]] | None = None,
    ) -> None:
        self._content_type_window = content_type_window
        self._switch_threshold = switch_threshold
        self._ignore_transitions: set[tuple[str, str]] = (
            ignore_transitions
            if ignore_transitions is not None
            else {("dialogue", "tool_io"), ("tool_io", "dialogue")}
        )

    @property
    def name(self) -> str:
        return "auto-branch"

    @property
    def priority(self) -> int:
        return 300

    @property
    def fires_on(self) -> str:
        return "commit"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        """Check if recent commits show rapid content type switching."""
        # Detached HEAD or no branch -- skip
        current_branch = tract.current_branch
        if current_branch is None:
            return None

        # Get recent commits
        commits = tract.log(limit=self._content_type_window)
        if len(commits) < 3:
            return None

        # Count content type transitions (excluding ignored ones)
        transitions = 0
        # commits are newest-first; reverse to get chronological order
        ordered = list(reversed(commits))
        for i in range(1, len(ordered)):
            prev_type = ordered[i - 1].content_type
            curr_type = ordered[i].content_type
            if prev_type != curr_type:
                pair = (prev_type, curr_type)
                if pair not in self._ignore_transitions:
                    transitions += 1

        if transitions >= self._switch_threshold:
            branch_name = f"tangent/{current_branch}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            return TriggerAction(
                action_type="branch",
                params={"name": branch_name, "switch": False},
                reason=(
                    f"Detected {transitions} content type transitions in "
                    f"last {len(commits)} commits (threshold: {self._switch_threshold})"
                ),
                autonomy="collaborative",
            )

        return None

    # ------------------------------------------------------------------
    # Hook integration
    # ------------------------------------------------------------------

    def default_handler(self, pending: PendingTrigger) -> None:
        """Auto-approve branch creation proposals."""
        pending.approve()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Serialize trigger configuration to a dict."""
        return {
            "name": self.name,
            "content_type_window": self._content_type_window,
            "switch_threshold": self._switch_threshold,
            "ignore_transitions": [list(t) for t in sorted(self._ignore_transitions)],
            "enabled": True,
        }

    @classmethod
    def from_config(cls, config: dict) -> BranchTrigger:
        """Deserialize a BranchTrigger from a config dict."""
        ignore = config.get("ignore_transitions")
        if ignore is not None:
            ignore = {tuple(t) for t in ignore}
        return cls(
            content_type_window=config.get("content_type_window", 5),
            switch_threshold=config.get("switch_threshold", 4),
            ignore_transitions=ignore,
        )
