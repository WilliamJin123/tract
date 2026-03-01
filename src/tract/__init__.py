"""Trace: Git-like version control for LLM context windows.

Agents produce better outputs when their context is clean, coherent, and relevant.
Trace makes context a managed, version-controlled resource.
"""

from tract._version import __version__

# Core entry point
from tract.tract import Tract

# Content types
from tract.models.content import (
    BUILTIN_TYPE_HINTS,
    ContentPayload,
    ContentTypeHints,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    ArtifactContent,
    OutputContent,
    ReasoningContent,
    ToolIOContent,
    validate_content,
)

# Commit and annotation types
from tract.models.commit import CommitInfo, CommitMetadata, CommitOperation
from tract.models.annotations import Priority, PriorityAnnotation, RetentionCriteria

# Configuration
from tract.models.config import TractConfig, TokenBudgetConfig, BudgetAction, LLMConfig, Operator, OperationConfigs, OperationClients, OperationPrompts, ToolSummarizationConfig

# Protocols and output types
from tract.protocols import (
    TokenCounter,
    ContextCompiler,
    Message,
    CompiledContext,
    CompileSnapshot,
    TokenUsage,
    ChatResponse,
    ToolCall,
    ToolCallDict,
    ToolCallOpenAIDict,
    ToolTurn,
)

# Branch model
from tract.models.branch import BranchInfo

# Merge models
from tract.models.merge import (
    ImportIssue,
    ImportResult,
    ConflictInfo,
    MergeResult,
    RebaseResult,
    RebaseWarning,
)

# Compression models
from tract.models.compression import CompressResult, GCResult, ReorderWarning, ToolCompactResult, ToolDropResult

# Hook system
from tract.hooks import Pending, PendingCompress, PendingToolResult, ValidationResult, HookRejection

# Compression prompts (for extending or selecting system prompts)
from tract.prompts.summarize import (
    DEFAULT_SUMMARIZE_SYSTEM,
    CONVERSATION_SUMMARIZE_SYSTEM,
    TOOL_SUMMARIZE_SYSTEM,
    TOOL_CONTEXT_SUMMARIZE_SYSTEM,
)

# Trigger engine
from tract.triggers import Trigger, TriggerEvaluator
from tract.triggers.builtin import (
    ArchiveTrigger,
    BranchTrigger,
    CompressTrigger,
    GCTrigger,
    MergeTrigger,
    PinTrigger,
    RebaseTrigger,
)
from tract.models.trigger import TriggerAction, EvaluationResult, TriggerLogEntry

# Session and spawn models
from tract.session import Session
from tract.models.session import SessionContent, SpawnInfo, CollapseResult

# Operations data models
from tract.operations.history import StatusInfo
from tract.operations.diff import DiffResult, MessageDiff, DiffStat

# LLM protocol
from tract.llm.protocols import LLMClient

# Agent toolkit
from tract.toolkit.models import ToolDefinition, ToolProfile, ToolConfig, ToolResult
from tract.toolkit.executor import ToolExecutor

# Orchestrator
from tract.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    AutonomyLevel,
    OrchestratorState,
    TriggerConfig,
    ToolCallDecision,
    ToolCallReview,
    StepResult,
    OrchestratorResult,
    auto_approve,
    log_and_approve,
    cli_prompt,
    reject_all,
    auto_approve_tool_call,
    log_and_approve_tool_call,
    reject_all_tool_call,
    make_log_handler,
    make_reject_handler,
)

# Tool tracking
from tract.models.tools import hash_tool_schema

# Retry protocol
from tract.retry import RetryResult, retry_with_steering

# Exceptions
from tract.exceptions import (
    TraceError,
    CommitNotFoundError,
    BlobNotFoundError,
    ContentValidationError,
    BudgetExceededError,
    EditTargetError,
    DetachedHeadError,
    AmbiguousPrefixError,
    BranchExistsError,
    BranchNotFoundError,
    InvalidBranchNameError,
    UnmergedBranchError,
    MergeError,
    MergeConflictError,
    NothingToMergeError,
    RebaseError,
    ImportCommitError,
    SemanticSafetyError,
    CompressionError,
    GCError,
    SpawnError,
    SessionError,
    TriggerExecutionError,
    TriggerConfigError,
    OrchestratorError,
    RetryExhaustedError,
    TagNotRegisteredError,
    CurationError,
)

__all__ = [
    "__version__",
    "Tract",
    # Content types
    "ContentPayload",
    "InstructionContent",
    "DialogueContent",
    "ToolIOContent",
    "ReasoningContent",
    "ArtifactContent",
    "OutputContent",
    "FreeformContent",
    "validate_content",
    "BUILTIN_TYPE_HINTS",
    "ContentTypeHints",
    # Commit types
    "CommitInfo",
    "CommitMetadata",
    "CommitOperation",
    # Annotations
    "Priority",
    "PriorityAnnotation",
    "RetentionCriteria",
    # Config
    "TractConfig",
    "TokenBudgetConfig",
    "BudgetAction",
    "LLMConfig",
    "Operator",
    "OperationConfigs",
    "OperationClients",
    "OperationPrompts",
    "ToolSummarizationConfig",
    # Protocols
    "TokenCounter",
    "ContextCompiler",
    "Message",
    "CompiledContext",
    "CompileSnapshot",
    "TokenUsage",
    "ChatResponse",
    "ToolCall",
    "ToolCallDict",
    "ToolCallOpenAIDict",
    "ToolTurn",
    # Branch model
    "BranchInfo",
    # Merge models
    "ConflictInfo",
    "MergeResult",
    "RebaseWarning",
    "ImportIssue",
    "RebaseResult",
    "ImportResult",
    # Operations
    "StatusInfo",
    "DiffResult",
    "MessageDiff",
    "DiffStat",
    # Exceptions
    "TraceError",
    "CommitNotFoundError",
    "BlobNotFoundError",
    "ContentValidationError",
    "BudgetExceededError",
    "EditTargetError",
    "DetachedHeadError",
    "AmbiguousPrefixError",
    "BranchExistsError",
    "BranchNotFoundError",
    "InvalidBranchNameError",
    "UnmergedBranchError",
    "MergeError",
    "MergeConflictError",
    "NothingToMergeError",
    "RebaseError",
    "ImportCommitError",
    "SemanticSafetyError",
    "CompressionError",
    "GCError",
    "SpawnError",
    "SessionError",
    "TriggerExecutionError",
    "TriggerConfigError",
    "OrchestratorError",
    "RetryExhaustedError",
    "TagNotRegisteredError",
    "CurationError",
    # Tool tracking
    "hash_tool_schema",
    # Retry protocol
    "RetryResult",
    "retry_with_steering",
    # Trigger engine
    "Trigger",
    "TriggerEvaluator",
    "CompressTrigger",
    "PinTrigger",
    "BranchTrigger",
    "MergeTrigger",
    "RebaseTrigger",
    "GCTrigger",
    "ArchiveTrigger",
    "TriggerAction",
    "EvaluationResult",
    "TriggerLogEntry",
    # Compression models
    "CompressResult",
    "GCResult",
    "ToolCompactResult",
    "ToolDropResult",
    "ReorderWarning",
    "DEFAULT_SUMMARIZE_SYSTEM",
    "CONVERSATION_SUMMARIZE_SYSTEM",
    "TOOL_SUMMARIZE_SYSTEM",
    "TOOL_CONTEXT_SUMMARIZE_SYSTEM",
    # Hook system
    "Pending",
    "PendingCompress",
    "PendingToolResult",
    "ValidationResult",
    "HookRejection",
    # Multi-agent / session
    "Session",
    "SessionContent",
    "SpawnInfo",
    "CollapseResult",
    # LLM protocol
    "LLMClient",
    # Agent toolkit
    "ToolDefinition",
    "ToolProfile",
    "ToolConfig",
    "ToolResult",
    "ToolExecutor",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "AutonomyLevel",
    "OrchestratorState",
    "TriggerConfig",
    "ToolCallDecision",
    "ToolCallReview",
    "StepResult",
    "OrchestratorResult",
    "auto_approve",
    "log_and_approve",
    "cli_prompt",
    "reject_all",
    "auto_approve_tool_call",
    "log_and_approve_tool_call",
    "reject_all_tool_call",
    "make_log_handler",
    "make_reject_handler",
]
