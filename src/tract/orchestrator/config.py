"""Orchestrator configuration types.

Provides AutonomyLevel, OrchestratorState, TriggerConfig, and
OrchestratorConfig for configuring the context management orchestrator.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tract.orchestrator.models import (
        OrchestratorProposal,
        ProposalResponse,
        StepResult,
    )


class AutonomyLevel(str, enum.Enum):
    """Autonomy levels for the orchestrator.

    Controls how much independence the orchestrator has in executing
    context management actions.
    """

    MANUAL = "manual"
    COLLABORATIVE = "collaborative"
    AUTONOMOUS = "autonomous"


class OrchestratorState(str, enum.Enum):
    """States the orchestrator can be in during its lifecycle."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    STOPPED = "stopped"


@dataclass(frozen=True)
class TriggerConfig:
    """Configuration for when the orchestrator should activate.

    All triggers default to None/False (disabled). Enable triggers
    selectively based on your use case.

    **Important:** Triggers fire synchronously inside ``commit()`` or
    ``compile()``.  When a trigger activates, the calling method blocks
    until the orchestrator run completes (including any LLM API calls).

    Note: on_schedule_seconds is deferred -- periodic scheduling
    requires a background timer which is incompatible with Tract's
    synchronous design. May be added in a future milestone.

    Attributes:
        on_commit_count: Fire after every N commits. Resets after firing.
        on_token_threshold: Fire when token usage exceeds this fraction
            of the budget (0.0-1.0). Fires once until usage drops below.
        on_compile: Fire on every compile() call.
        autonomy: Override autonomy level for trigger-invoked runs.
            When set, effective autonomy is min(ceiling, autonomy).
            When None, the orchestrator's autonomy_ceiling is used as-is.
    """

    on_commit_count: int | None = None
    on_token_threshold: float | None = None
    on_compile: bool = False
    autonomy: AutonomyLevel | None = None


@dataclass
class OrchestratorConfig:
    """Configuration for the context management orchestrator.

    Mutable dataclass (like TractConfig pattern) -- users may adjust
    settings between orchestrator runs.

    Attributes:
        autonomy_ceiling: Maximum autonomy level for actions.
        max_steps: Maximum number of tool calls per orchestrator run.
        profile: Agent profile identifier (e.g., "self" for self-management).
        system_prompt: Override for the default orchestrator system prompt.
        task_context: Optional description of current task for relevance assessment.
        triggers: Configuration for automatic orchestrator activation.
        model: LLM model identifier (None = use default from LLM client).
        temperature: LLM temperature for orchestrator calls.
        on_proposal: Callback invoked when the orchestrator proposes an action.
        on_step: Callback invoked after each orchestrator step completes.
    """

    autonomy_ceiling: AutonomyLevel = AutonomyLevel.COLLABORATIVE
    max_steps: int = 10
    profile: str = "self"
    system_prompt: str | None = None
    task_context: str | None = None
    triggers: TriggerConfig | None = None
    model: str | None = None
    temperature: float = 0.0
    on_proposal: Callable[[OrchestratorProposal], ProposalResponse] | None = None
    on_step: Callable[[StepResult], None] | None = None
