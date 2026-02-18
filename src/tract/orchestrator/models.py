"""Orchestrator result and proposal models.

Provides ToolCall, ProposalDecision, OrchestratorProposal,
ProposalResponse, StepResult, and OrchestratorResult for the
orchestrator's proposal-review-execute loop.
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


class ProposalDecision(str, enum.Enum):
    """Decision outcomes for an orchestrator proposal."""

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class OrchestratorProposal:
    """A proposed action from the orchestrator awaiting review.

    Mutable: the decision field changes as the proposal is
    approved, rejected, or modified.
    """

    proposal_id: str
    recommended_action: ToolCall
    reasoning: str
    alternatives: list[ToolCall] = field(default_factory=list)
    context_summary: str = ""
    decision: ProposalDecision | str = "pending"
    modified_action: ToolCall | None = None


@dataclass(frozen=True)
class ProposalResponse:
    """Response to an orchestrator proposal from a callback.

    Frozen: once a callback decides, the response is immutable.
    """

    decision: ProposalDecision
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
    proposal: OrchestratorProposal | None = None


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
