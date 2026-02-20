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
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.annotations import Priority, PriorityAnnotation

# Configuration
from tract.models.config import TractConfig, TokenBudgetConfig, BudgetAction, LLMConfig, OperationConfigs

# Protocols and output types
from tract.protocols import (
    TokenCounter,
    ContextCompiler,
    Message,
    CompiledContext,
    CompileSnapshot,
    TokenUsage,
    ChatResponse,
)

# Branch model
from tract.models.branch import BranchInfo

# Merge models
from tract.models.merge import (
    CherryPickIssue,
    CherryPickResult,
    ConflictInfo,
    MergeResult,
    RebaseResult,
    RebaseWarning,
)

# Compression models
from tract.models.compression import CompressResult, GCResult, PendingCompression, ReorderWarning

# Policy engine
from tract.policy import Policy, PolicyEvaluator
from tract.policy.builtin import ArchivePolicy, CompressPolicy, PinPolicy, BranchPolicy, RebasePolicy
from tract.models.policy import PolicyAction, PolicyProposal, EvaluationResult, PolicyLogEntry

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
    OrchestratorProposal,
    ProposalDecision,
    ProposalResponse,
    StepResult,
    OrchestratorResult,
    ToolCall,
    auto_approve,
    log_and_approve,
    cli_prompt,
    reject_all,
)

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
    CherryPickError,
    SemanticSafetyError,
    CompressionError,
    GCError,
    SpawnError,
    SessionError,
    PolicyExecutionError,
    PolicyConfigError,
    OrchestratorError,
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
    "CommitOperation",
    # Annotations
    "Priority",
    "PriorityAnnotation",
    # Config
    "TractConfig",
    "TokenBudgetConfig",
    "BudgetAction",
    "LLMConfig",
    "OperationConfigs",
    # Protocols
    "TokenCounter",
    "ContextCompiler",
    "Message",
    "CompiledContext",
    "CompileSnapshot",
    "TokenUsage",
    "ChatResponse",
    # Branch model
    "BranchInfo",
    # Merge models
    "ConflictInfo",
    "MergeResult",
    "RebaseWarning",
    "CherryPickIssue",
    "RebaseResult",
    "CherryPickResult",
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
    "CherryPickError",
    "SemanticSafetyError",
    "CompressionError",
    "GCError",
    "SpawnError",
    "SessionError",
    "PolicyExecutionError",
    "PolicyConfigError",
    "OrchestratorError",
    # Policy engine
    "Policy",
    "PolicyEvaluator",
    "CompressPolicy",
    "PinPolicy",
    "BranchPolicy",
    "ArchivePolicy",
    "RebasePolicy",
    "PolicyAction",
    "PolicyProposal",
    "EvaluationResult",
    "PolicyLogEntry",
    # Compression models
    "CompressResult",
    "GCResult",
    "PendingCompression",
    "ReorderWarning",
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
    "OrchestratorProposal",
    "ProposalDecision",
    "ProposalResponse",
    "StepResult",
    "OrchestratorResult",
    "ToolCall",
    "auto_approve",
    "log_and_approve",
    "cli_prompt",
    "reject_all",
]
