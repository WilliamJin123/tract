"""Toolkit data models for Trace agent tool definitions.

Frozen dataclasses for tool definitions, profiles, configs, and results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical tool name type — single source of truth for autocomplete & validation
# ---------------------------------------------------------------------------
ToolName = Literal[
    "commit",
    "compile",
    "annotate",
    "status",
    "log",
    "diff",
    "compress",
    "branch",
    "switch",
    "merge",
    "reset",
    "checkout",
    "gc",
    "list_branches",
    "get_commit",
    "configure_model",
    "tag",
    "untag",
    "query_by_tags",
    "register_tag",
    "get_tags",
    "list_tags",
    "configure",
    "create_metadata",
    "get_config",
    "transition",
    "directive",
    "create_middleware",
    "remove_middleware",
]


@dataclass(frozen=True)
class ToolDefinition:
    """A single tool definition for LLM consumption.

    Attributes:
        name: Tool name (e.g. "commit", "compile").
        description: Human-readable description of when/why to use this tool.
        parameters: JSON Schema dict describing tool parameters.
        handler: Callable that executes the tool.
    """

    name: str  # str (not ToolName) to allow dynamic fire_* tools
    description: str
    parameters: dict
    handler: Callable[..., object]

    def to_openai(self) -> dict:
        """Convert to OpenAI function-calling format.

        Returns:
            Dict with "type": "function" and nested "function" object.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict:
        """Convert to Anthropic tool-use format.

        Returns:
            Dict with "name", "description", and "input_schema".
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass(frozen=True)
class ToolConfig:
    """Per-tool configuration within a profile.

    Attributes:
        enabled: Whether this tool is included in the profile.
        description: Override description, or None to use default.
    """

    enabled: bool = True
    description: str | None = None


@dataclass
class ToolProfile:
    """A named profile that curates a subset of tools with optional description overrides.

    Attributes:
        name: Profile identifier (e.g. "self", "supervisor", "full").
        tool_configs: Mapping of tool_name -> ToolConfig.
    """

    name: str
    tool_configs: dict[ToolName | str, ToolConfig] = field(default_factory=dict)

    def filter_tools(self, all_tools: list[ToolDefinition]) -> list[ToolDefinition]:
        """Filter and optionally override tool descriptions based on this profile.

        Only tools present in ``tool_configs`` with ``enabled=True`` are included.
        If a ``ToolConfig`` provides a description override, the tool's description
        is replaced using ``dataclasses.replace()``.

        Args:
            all_tools: Complete list of available tool definitions.

        Returns:
            Filtered (and possibly description-overridden) tool definitions.
        """
        from dataclasses import replace

        result: list[ToolDefinition] = []
        for tool in all_tools:
            config = self.tool_configs.get(tool.name)
            if config is None or not config.enabled:
                continue
            if config.description is not None:
                tool = replace(tool, description=config.description)
            result.append(tool)
        return result


@dataclass(frozen=True)
class ToolResult:
    """Structured result from executing a tool.

    Attributes:
        tool_name: Name of the tool that was executed.
        success: Whether execution succeeded.
        output: String output on success.
        error: Error message on failure.
        hint: Actionable suggestion (from exception or discovery).
        duration_ms: Execution time in milliseconds.
        truncated: Whether output was truncated by presentation layer.
    """

    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    hint: str = ""
    duration_ms: float = 0
    truncated: bool = False

    def __str__(self) -> str:
        """Return LLM-friendly string: output on success, error on failure."""
        if self.success:
            return self.output
        msg = f"[error] {self.tool_name}: {self.error}" if self.error else f"[error] {self.tool_name}: unknown error"
        if self.hint:
            msg += f"\n[hint] {self.hint}"
        return msg
