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
from tract.models.config import TractConfig, TokenBudgetConfig, BudgetAction

# Protocols and output types
from tract.protocols import (
    TokenCounter,
    ContextCompiler,
    Message,
    CompiledContext,
    CompileSnapshot,
    TokenUsage,
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

# Operations data models
from tract.operations.history import StatusInfo
from tract.operations.diff import DiffResult, MessageDiff, DiffStat

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
    # Protocols
    "TokenCounter",
    "ContextCompiler",
    "Message",
    "CompiledContext",
    "CompileSnapshot",
    "TokenUsage",
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
    # Compression models
    "CompressResult",
    "GCResult",
    "PendingCompression",
    "ReorderWarning",
    "CompressionError",
    "GCError",
]
