"""Trace domain models.

Re-exports key models for convenient access.
"""

from trace_context.models.annotations import DEFAULT_TYPE_PRIORITIES, Priority, PriorityAnnotation
from trace_context.models.commit import CommitInfo, CommitOperation
from trace_context.models.config import BudgetAction, RepoConfig, TokenBudgetConfig
from trace_context.models.content import (
    BUILTIN_TYPE_HINTS,
    ArtifactContent,
    ContentPayload,
    ContentTypeHints,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    OutputContent,
    ReasoningContent,
    ToolIOContent,
    validate_content,
)

__all__ = [
    # Content types
    "InstructionContent",
    "DialogueContent",
    "ToolIOContent",
    "ReasoningContent",
    "ArtifactContent",
    "OutputContent",
    "FreeformContent",
    "ContentPayload",
    "ContentTypeHints",
    "BUILTIN_TYPE_HINTS",
    "validate_content",
    # Commit
    "CommitOperation",
    "CommitInfo",
    # Annotations
    "Priority",
    "PriorityAnnotation",
    "DEFAULT_TYPE_PRIORITIES",
    # Config
    "BudgetAction",
    "TokenBudgetConfig",
    "RepoConfig",
]
