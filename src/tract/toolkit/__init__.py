"""Agent Toolkit: LLM-consumable tool definitions for Tract operations.

Provides tool definitions, profiles, and an executor that expose
Tract operations as function-calling schemas for LLM agents.
"""

from tract.toolkit.definitions import get_all_tools
from tract.toolkit.models import ToolConfig, ToolDefinition, ToolProfile, ToolResult
from tract.toolkit.profiles import (
    FULL_PROFILE,
    SELF_PROFILE,
    SUPERVISOR_PROFILE,
    get_profile,
)

# ToolCall is canonically defined in tract.orchestrator.models (Plan 02).
# Re-export here for convenience. Plan 02 may not have run yet.
try:
    from tract.orchestrator.models import ToolCall
except ImportError:
    ToolCall = None  # type: ignore[assignment,misc]

# Lazy import to avoid circular dependency (executor imports definitions)
def __getattr__(name: str):
    if name == "ToolExecutor":
        from tract.toolkit.executor import ToolExecutor
        return ToolExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ToolDefinition",
    "ToolProfile",
    "ToolConfig",
    "ToolResult",
    "ToolExecutor",
    "ToolCall",
    "get_all_tools",
    "get_profile",
    "SELF_PROFILE",
    "SUPERVISOR_PROFILE",
    "FULL_PROFILE",
]
