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
from tract.models.config import TractConfig, TokenBudgetConfig, BudgetAction, LLMConfig, Operator, OperationConfigs, OperationClients, OperationPrompts, RetryConfig, ToolSummarizationConfig

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
    MergeStrategy,
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
from tract.operations.health import HealthReport
from tract.operations.history import StatusInfo
from tract.operations.diff import DiffResult, MessageDiff, DiffStat

# Config index and middleware
from tract.operations.config_index import ConfigIndex
from tract.middleware import MiddlewareContext, MiddlewareEvent

# Semantic gates
from tract.gate import SemanticGate, GateResult

# Semantic maintainers
from tract.maintain import SemanticMaintainer, MaintainResult

# Directive templates
from tract.templates import DirectiveTemplate, list_templates, get_template, register_template

# Workflow profiles
from tract.profiles import (
    WorkflowProfile,
    get_profile as get_workflow_profile,
    list_profiles as list_workflow_profiles,
    register_profile as register_workflow_profile,
)

# LLM protocol (always available — these are just Protocol definitions)
from tract.llm.protocols import LLMClient, AgentLoop
from tract.llm.protocols import AsyncLLMClient, acall_llm

# LLM fallback client (no external deps — always available)
from tract.llm.fallback import FallbackClient

# LLM test utilities (no external deps — always available)
from tract.llm.testing import MockLLMClient, ReplayLLMClient, FunctionLLMClient

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
    from tract.toolkit.presentation import ToolPresenter, PresentationConfig
    from tract.toolkit.discovery import get_discovery_tools
except ImportError:
    pass

try:
    from tract.loop import LoopConfig, LoopResult, StepMetrics, run_loop, arun_loop
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
    ClosedError,
    ThreadSafetyError,
    RetryExhaustedError,
)

# Formatting utilities
from tract.formatting import StreamPrinter

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
    "RetryConfig",
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
    "MergeStrategy",
    "RebaseWarning",
    "ImportIssue",
    "RebaseResult",
    "ImportResult",
    # Operations
    "HealthReport",
    "StatusInfo",
    "DiffResult",
    "MessageDiff",
    "DiffStat",
    # Config index and middleware
    "ConfigIndex",
    # Semantic gates
    "SemanticGate",
    "GateResult",
    # Semantic maintainers
    "SemanticMaintainer",
    "MaintainResult",
    "MiddlewareContext",
    "MiddlewareEvent",
    # Directive templates
    "DirectiveTemplate",
    "list_templates",
    "get_template",
    "register_template",
    # Workflow profiles
    "WorkflowProfile",
    "get_workflow_profile",
    "list_workflow_profiles",
    "register_workflow_profile",
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
    "ClosedError",
    "ThreadSafetyError",
    "RetryExhaustedError",
    # Formatting utilities
    "StreamPrinter",
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
    "AsyncLLMClient",
    "acall_llm",
    # LLM fallback client
    "FallbackClient",
    # LLM test utilities
    "MockLLMClient",
    "ReplayLLMClient",
    "FunctionLLMClient",
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
    "ToolPresenter",
    "PresentationConfig",
    "get_discovery_tools",
    # Default loop
    "LoopConfig",
    "LoopResult",
    "StepMetrics",
    "run_loop",
    "arun_loop",
    # Type aliases
    "CompileStrategy",
]
