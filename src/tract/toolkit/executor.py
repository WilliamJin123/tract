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
    from tract.toolkit.models import ToolDefinition, ToolProfile
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
        self._profile: ToolProfile | None = None
        self._tool_overrides: dict[str, bool] = {}
        self._rebuild_tools()

    def _rebuild_tools(self) -> None:
        """Rebuild the internal tool lookup from current Tract state."""
        from tract.toolkit.definitions import get_all_tools

        all_tools = get_all_tools(self._tract)

        # Apply profile filtering
        if self._profile is not None:
            all_tools = self._profile.filter_tools(all_tools)

        self._tools.clear()
        for tool in all_tools:
            self._tools[tool.name] = tool

        # Apply overrides: unlock/lock specific tools
        if self._tool_overrides:
            from tract.toolkit.definitions import get_all_tools as _get_all

            # Get unfiltered tools for unlock lookups
            full_tools = {t.name: t for t in _get_all(self._tract)}
            for name, enabled in self._tool_overrides.items():
                if enabled and name not in self._tools and name in full_tools:
                    self._tools[name] = full_tools[name]
                elif not enabled and name in self._tools:
                    del self._tools[name]

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

    def set_profile(self, profile: str | ToolProfile) -> None:
        """Change the active tool profile.

        Clears any tool overrides and rebuilds the tool set.

        Args:
            profile: Profile name ("self", "supervisor", "full") or
                a ToolProfile instance.
        """
        from tract.toolkit.models import ToolProfile as _TP
        from tract.toolkit.profiles import get_profile

        if isinstance(profile, str):
            profile = get_profile(profile)
        elif not isinstance(profile, _TP):
            raise TypeError(f"Expected str or ToolProfile, got {type(profile).__name__}")
        self._profile = profile
        self._tool_overrides.clear()
        self._rebuild_tools()

    def unlock_tool(self, tool_name: str) -> None:
        """Force-enable a tool regardless of profile restrictions.

        Args:
            tool_name: Name of the tool to unlock.
        """
        self._tool_overrides[tool_name] = True
        self._rebuild_tools()

    def lock_tool(self, tool_name: str) -> None:
        """Force-disable a tool regardless of profile permissions.

        Args:
            tool_name: Name of the tool to lock.
        """
        self._tool_overrides[tool_name] = False
        self._rebuild_tools()
