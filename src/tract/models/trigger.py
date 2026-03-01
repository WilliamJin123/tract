"""Domain models for the trigger engine subsystem.

Provides data classes for trigger actions, evaluation results,
and log entries used by the trigger evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class TriggerAction:
    """An action that a trigger wants to perform.

    Immutable: once a trigger determines an action, it doesn't change.
    The autonomy level determines whether it executes immediately or
    requires approval.
    """

    action_type: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    autonomy: str = "collaborative"  # "autonomous", "collaborative", "supervised"


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a single trigger against current state.

    Immutable: once evaluation completes, the result is final.
    """

    trigger_name: str
    triggered: bool
    action: TriggerAction | None = None
    outcome: str = "skipped"  # "executed", "proposed", "skipped", "error"
    error: str | None = None
    commit_hash: str | None = None


@dataclass(frozen=True)
class TriggerLogEntry:
    """A log entry recording a trigger evaluation for audit purposes.

    Maps 1:1 with TriggerLogRow in the database.
    """

    id: int
    tract_id: str
    trigger_name: str
    trigger: str  # "compile" or "commit"
    action_type: str | None
    reason: str | None
    outcome: str  # "executed", "proposed", "skipped", "error"
    commit_hash: str | None
    error_message: str | None
    created_at: datetime
