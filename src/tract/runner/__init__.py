"""Tract Runner: LLM clients, agent toolkit, and default loop.

This package provides the framework layer on top of tract's core DAG primitives.
Requires optional dependencies: pip install tract-ai[runner]

Core (always available):
    from tract import Tract, CommitInfo, CompiledContext, ...

Runner (requires tract-ai[runner]):
    from tract.runner import OpenAIClient, AnthropicClient, run_loop, ToolExecutor, ...
"""

# LLM clients
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
from tract.llm.cache import CachingLLMClient
from tract.llm.claude_code import load_claude_code_credentials, create_claude_code_client
from tract.llm.protocols import LLMClient, AgentLoop, Resolution, ResolverCallable
from tract.llm.resolver import OpenAIResolver
from tract.llm.errors import (
    LLMClientError,
    LLMResponseError,
    LLMConfigError,
    LLMRateLimitError,
    LLMAuthError,
    LLMToolUseError,
)

# Agent toolkit
from tract.toolkit.models import ToolDefinition, ToolName, ToolProfile, ToolConfig, ToolResult
from tract.toolkit.profiles import ProfileName, get_profile
from tract.toolkit.executor import ToolExecutor
from tract.toolkit.definitions import get_all_tools
from tract.toolkit.callables import tools_to_callables

# Default loop
from tract.loop import LoopConfig, LoopResult, run_loop

__all__ = [
    # LLM
    "OpenAIClient",
    "AnthropicClient",
    "CachingLLMClient",
    "load_claude_code_credentials",
    "create_claude_code_client",
    "LLMClient",
    "AgentLoop",
    "Resolution",
    "ResolverCallable",
    "OpenAIResolver",
    "LLMClientError",
    "LLMResponseError",
    "LLMConfigError",
    "LLMRateLimitError",
    "LLMAuthError",
    "LLMToolUseError",
    "StreamEvent",
    "TextDelta",
    "ToolCallStart",
    "ToolCallDelta",
    "ThinkingDelta",
    "UsageEvent",
    "MessageDone",
    # Toolkit
    "ToolDefinition",
    "ToolName",
    "ToolProfile",
    "ToolConfig",
    "ToolResult",
    "ProfileName",
    "ToolExecutor",
    "get_profile",
    "get_all_tools",
    "tools_to_callables",
    # Loop
    "LoopConfig",
    "LoopResult",
    "run_loop",
]
