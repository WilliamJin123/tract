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
    ConfigContent,
    ContentPayload,
    ContentTypeHints,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    ArtifactContent,
    MetadataContent,
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

# Compression prompts (for extending or selecting system prompts)
from tract.prompts.summarize import (
    DEFAULT_SUMMARIZE_SYSTEM,
    CONVERSATION_SUMMARIZE_SYSTEM,
    TOOL_SUMMARIZE_SYSTEM,
    TOOL_CONTEXT_SUMMARIZE_SYSTEM,
)

# Session and spawn models
from tract.session import Session
from tract.models.session import SessionContent, SpawnInfo, CollapseResult

# Operations data models
from tract.operations.history import StatusInfo
from tract.operations.diff import DiffResult, MessageDiff, DiffStat

# Config index and middleware
from tract.operations.config_index import ConfigIndex
from tract.middleware import MiddlewareContext, MiddlewareEvent

# LLM protocol (always available — these are just Protocol definitions)
from tract.llm.protocols import LLMClient, AgentLoop

# Runner components (require optional dependencies: pip install tract-ai[runner])
try:
    from tract.llm.client import OpenAIClient
    from tract.llm.anthropic_client import (
        AnthropicClient,
        StreamEvent,
        TextDelta,
        ToolCallStart,
        ToolCallDelta,
        ThinkingDelta,
        UsageEvent,
        MessageDone,
    )
except ImportError:
    pass

try:
    from tract.toolkit.models import ToolDefinition, ToolName, ToolProfile, ToolConfig, ToolResult
    from tract.toolkit.profiles import ProfileName
    from tract.toolkit.executor import ToolExecutor
except ImportError:
    pass

try:
    from tract.loop import LoopConfig, LoopResult, run_loop
except ImportError:
    pass

# Type aliases for IDE autocomplete
from tract.tract import CompileStrategy

# Tool tracking
from tract.models.tools import hash_tool_schema

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
    TagNotRegisteredError,
    CurationError,
    BlockedError,
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
    "ConfigContent",
    "MetadataContent",
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
    # Config index and middleware
    "ConfigIndex",
    "MiddlewareContext",
    "MiddlewareEvent",
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
    "TagNotRegisteredError",
    "CurationError",
    "BlockedError",
    # Tool tracking
    "hash_tool_schema",
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
    # Multi-agent / session
    "Session",
    "SessionContent",
    "SpawnInfo",
    "CollapseResult",
    # LLM protocol
    "LLMClient",
    "AgentLoop",
    # LLM clients
    "OpenAIClient",
    "AnthropicClient",
    # Stream event types
    "StreamEvent",
    "TextDelta",
    "ToolCallStart",
    "ToolCallDelta",
    "ThinkingDelta",
    "UsageEvent",
    "MessageDone",
    # Agent toolkit
    "ToolDefinition",
    "ToolName",
    "ToolProfile",
    "ProfileName",
    "ToolConfig",
    "ToolResult",
    "ToolExecutor",
    # Default loop
    "LoopConfig",
    "LoopResult",
    "run_loop",
    # Type aliases
    "CompileStrategy",
]
