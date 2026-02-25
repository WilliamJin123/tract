"""Orchestrator package -- context management orchestrator types and templates.

Provides the Orchestrator class, configuration, step/result types,
tool-call review models, and pre-built hook handler templates for
the context management orchestrator loop.
"""

from tract.hooks.templates.orchestrator import (
    auto_approve,
    auto_approve_tool_call,
    cli_prompt,
    log_and_approve,
    log_and_approve_tool_call,
    make_log_handler,
    make_reject_handler,
    reject_all,
    reject_all_tool_call,
)
from tract.orchestrator.config import (
    AutonomyLevel,
    OrchestratorConfig,
    OrchestratorState,
    TriggerConfig,
)
from tract.orchestrator.loop import Orchestrator
from tract.orchestrator.models import (
    OrchestratorResult,
    StepResult,
    ToolCall,
    ToolCallDecision,
    ToolCallReview,
)

__all__ = [
    # Core
    "Orchestrator",
    # Config
    "AutonomyLevel",
    "OrchestratorConfig",
    "OrchestratorState",
    "TriggerConfig",
    # Models
    "ToolCall",
    "ToolCallDecision",
    "ToolCallReview",
    "StepResult",
    "OrchestratorResult",
    # Hook handler templates
    "auto_approve",
    "log_and_approve",
    "cli_prompt",
    "reject_all",
    "make_log_handler",
    "make_reject_handler",
    # Tool-call review callbacks
    "auto_approve_tool_call",
    "log_and_approve_tool_call",
    "reject_all_tool_call",
]
