"""ToolExecutor: dispatches tool calls to Tract methods.

Provides a single ``execute()`` method that looks up the tool by name,
invokes its handler with the provided arguments, and returns a structured
``ToolResult``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tract.toolkit.models import ToolResult

if TYPE_CHECKING:
    from tract.toolkit.models import ToolDefinition
    from tract.tract import Tract

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Dispatches tool calls to Tract methods and returns structured results.

    Usage::

        executor = ToolExecutor(tract)
        result = executor.execute("status", {})
        if result.success:
            print(result.output)
        else:
            print(result.error)
    """

    def __init__(self, tract: Tract) -> None:
        self._tract = tract
        self._tools: dict[str, ToolDefinition] = {}
        self._rebuild_tools()

    def _rebuild_tools(self) -> None:
        """Rebuild the internal tool lookup from current Tract state."""
        from tract.toolkit.definitions import get_all_tools

        self._tools.clear()
        for tool in get_all_tools(self._tract):
            self._tools[tool.name] = tool

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dict of arguments matching the tool's parameter schema.

        Returns:
            ToolResult with success/failure status and output/error.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        try:
            result = tool.handler(**arguments)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=str(result),
            )
        except Exception as exc:
            logger.debug("Tool %s failed: %s", tool_name, exc, exc_info=True)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def available_tools(self) -> list[str]:
        """Return the names of all available tools.

        Returns:
            List of tool name strings.
        """
        return list(self._tools.keys())
