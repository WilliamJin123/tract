"""Orchestrator package -- context management orchestrator types and callbacks.

Provides configuration, proposal models, step/result types, and
built-in callbacks for the context management orchestrator loop.
"""

from tract.orchestrator.callbacks import (
    auto_approve,
    cli_prompt,
    log_and_approve,
    reject_all,
)
from tract.orchestrator.config import (
    AutonomyLevel,
    OrchestratorConfig,
    OrchestratorState,
    TriggerConfig,
)
from tract.orchestrator.models import (
    OrchestratorProposal,
    OrchestratorResult,
    ProposalDecision,
    ProposalResponse,
    StepResult,
    ToolCall,
)

__all__ = [
    # Config
    "AutonomyLevel",
    "OrchestratorConfig",
    "OrchestratorState",
    "TriggerConfig",
    # Models
    "ToolCall",
    "ProposalDecision",
    "OrchestratorProposal",
    "ProposalResponse",
    "StepResult",
    "OrchestratorResult",
    # Callbacks
    "auto_approve",
    "log_and_approve",
    "cli_prompt",
    "reject_all",
]
