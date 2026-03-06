"""Rule engine data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.rules.index import RuleIndex
    from tract.tract import Tract


@dataclass(frozen=True)
class RuleEntry:
    """A rule with its DAG context."""

    name: str
    trigger: str
    condition: dict | None
    action: dict
    commit_hash: str
    dag_distance: int
    provenance: dict | None = None


@dataclass(frozen=True)
class EvalContext:
    """Immutable context passed to condition evaluators and action handlers."""

    event: str
    commit: CommitInfo | None
    branch: str
    head: str
    tract: Tract
    metrics: dict[str, Any] | None = None
    rule_index: RuleIndex | None = None


@dataclass(frozen=True)
class ActionResult:
    """Result from executing a single action."""

    action_type: str
    success: bool
    data: dict = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class EvalResult:
    """Result from evaluating rules for an event."""

    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    action_results: list[ActionResult] = field(default_factory=list)
    rules_evaluated: int = 0
    rules_fired: int = 0
