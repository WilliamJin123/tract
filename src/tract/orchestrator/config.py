"""Orchestrator configuration types.

Provides AutonomyLevel, OrchestratorState, TriggerConfig, and
OrchestratorConfig for configuring the context management orchestrator.

Autonomy levels map to hook configurations:
- AUTONOMOUS: no hooks (auto-approve everything)
- COLLABORATIVE: on_tool_call callback reviews each tool call
- MANUAL: all tool calls skipped
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tract.orchestrator.models import (
        StepResult,
        ToolCall,
        ToolCallReview,
    )


class AutonomyLevel(str, enum.Enum):
    """Autonomy levels for the orchestrator.

    Controls how much independence the orchestrator has in executing
    context management actions.

    - ``MANUAL``: All tool calls are skipped.
    - ``COLLABORATIVE``: Tool calls go through the ``on_tool_call``
      review callback before execution.
    - ``AUTONOMOUS``: Tool calls execute directly; hookable operations
      are still gated by Tract's hook system.
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
        max_tokens: Maximum tokens for LLM response in orchestrator calls.
        extra_llm_kwargs: Additional LLM kwargs (top_p, seed, etc.) forwarded to client.chat().
        on_tool_call: Callback invoked in collaborative mode to review a
            tool call. Takes a ToolCall, returns a ToolCallReview.
            Replaces the old ``on_proposal`` callback.
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
    max_tokens: int | None = None
    extra_llm_kwargs: dict | None = None
    on_tool_call: Callable[[ToolCall], ToolCallReview] | None = None
    on_step: Callable[[StepResult], None] | None = None
