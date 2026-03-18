"""Tract domain models.

Re-exports key models for convenient access.
"""

from tract.models.annotations import DEFAULT_TYPE_PRIORITIES, Priority, PriorityAnnotation, RetentionCriteria
from tract.models.commit import CommitInfo, CommitMetadata, CommitOperation
from tract.models.config import (
    BudgetAction,
    LLMConfig,
    Operator,
    OperationClients,
    OperationConfigs,
    OperationPrompts,
    RetryConfig,
    TractConfig,
    TokenBudgetConfig,
    ToolSummarizationConfig,
)
from tract.models.merge import MergeStrategy
from tract.models.content import (
    BUILTIN_TYPE_HINTS,
    ArtifactContent,
    ConfigContent,
    ContentPayload,
    ContentTypeHints,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    MetadataContent,
    OutputContent,
    ReasoningContent,
    ToolIOContent,
    validate_content,
)

__all__ = [
    # Content types
    "InstructionContent",
    "ConfigContent",
    "DialogueContent",
    "ToolIOContent",
    "ReasoningContent",
    "ArtifactContent",
    "OutputContent",
    "FreeformContent",
    "MetadataContent",
    "ContentPayload",
    "ContentTypeHints",
    "BUILTIN_TYPE_HINTS",
    "validate_content",
    # Commit
    "CommitOperation",
    "CommitInfo",
    "CommitMetadata",
    # Annotations
    "Priority",
    "PriorityAnnotation",
    "RetentionCriteria",
    "DEFAULT_TYPE_PRIORITIES",
    # Config
    "BudgetAction",
    "LLMConfig",
    "Operator",
    "OperationClients",
    "OperationConfigs",
    "OperationPrompts",
    "RetryConfig",
    "TokenBudgetConfig",
    "ToolSummarizationConfig",
    "TractConfig",
    # Merge
    "MergeStrategy",
]
