"""Toolkit data models for Trace agent tool definitions.

Frozen dataclasses for tool definitions, profiles, configs, and results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDefinition:
    """A single tool definition for LLM consumption.

    Attributes:
        name: Tool name (e.g. "commit", "compile").
        description: Human-readable description of when/why to use this tool.
        parameters: JSON Schema dict describing tool parameters.
        handler: Callable that executes the tool.
    """

    name: str
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
    tool_configs: dict[str, ToolConfig] = field(default_factory=dict)

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
    """

    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
