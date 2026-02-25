"""Orchestrator result and step models.

Provides ToolCall, ToolCallDecision, StepResult, and OrchestratorResult
for the orchestrator's hook-based tool-calling loop.

Note: OrchestratorProposal, ProposalResponse, and ProposalDecision have
been removed in favour of the unified hook system. Collaborative-mode
review is now handled by a simple ``on_tool_call`` callback that returns
a ToolCallDecision.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.orchestrator.config import OrchestratorState


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation with its arguments.

    This is the CANONICAL location for ToolCall. Other packages
    (e.g., toolkit) should re-export from here.
    """

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


class ToolCallDecision(str, enum.Enum):
    """Decision outcomes for an orchestrator tool-call review.

    Replaces the old ProposalDecision enum. Used by the
    ``on_tool_call`` callback to signal approve/reject/modify.
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass(frozen=True)
class ToolCallReview:
    """Response from a tool-call review callback.

    Replaces the old ProposalResponse dataclass. Returned by the
    ``on_tool_call`` callback in collaborative mode.
    """

    decision: ToolCallDecision
    modified_action: ToolCall | None = None
    reason: str = ""


@dataclass(frozen=True)
class StepResult:
    """Result of a single orchestrator step (one tool call).

    Frozen: step results are immutable records of what happened.
    """

    step: int
    tool_call: ToolCall
    result_output: str = ""
    result_error: str = ""
    success: bool = True
    review_decision: str = ""


@dataclass(frozen=True)
class OrchestratorResult:
    """Final result of an orchestrator run.

    Frozen: the result is immutable once the run completes.
    """

    steps: list[StepResult] = field(default_factory=list)
    state: OrchestratorState = field(default=None)  # type: ignore[assignment]
    assessment: str = ""
    total_tool_calls: int = 0

    def __post_init__(self) -> None:
        # Lazy import to avoid circular dependency at module level
        if self.state is None:
            from tract.orchestrator.config import OrchestratorState

            object.__setattr__(self, "state", OrchestratorState.IDLE)

    @property
    def succeeded(self) -> list[StepResult]:
        """Return all steps that completed successfully."""
        return [s for s in self.steps if s.success]
