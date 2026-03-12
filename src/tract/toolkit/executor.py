"""ToolExecutor: dispatches tool calls to Tract methods.

Provides a single ``execute()`` method that looks up the tool by name,
invokes its handler with the provided arguments, and returns a structured
``ToolResult``.
"""

from __future__ import annotations

import inspect
import logging
import time
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
        # Handle compact profile specially: domain-grouped tools
        if self._profile is not None and self._profile.name == "compact":
            from tract.toolkit.compact import get_compact_tools

            compact_tools = get_compact_tools(self._tract)
            self._tools.clear()
            for tool in compact_tools:
                self._tools[tool.name] = tool
            return

        # Handle discovery profile specially: 3 meta-tools
        if self._profile is not None and self._profile.name == "discovery":
            from tract.toolkit.discovery import get_discovery_tools

            discovery_tools = get_discovery_tools(self._tract)
            self._tools.clear()
            for tool in discovery_tools:
                self._tools[tool.name] = tool
            return

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
            available = ", ".join(sorted(self._tools.keys()))
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
                hint=f"Available tools: {available}",
            )

        # Strip hallucinated kwargs that the handler doesn't accept.
        # Small LLMs often invent extra parameters not in the tool schema.
        sig = inspect.signature(tool.handler)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            valid_params = set(sig.parameters.keys())
            arguments = {k: v for k, v in arguments.items() if k in valid_params}

        start = time.perf_counter()
        try:
            result = tool.handler(**arguments)
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=str(result),
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug("Tool %s failed: %s", tool_name, exc, exc_info=True)
            hint = getattr(exc, "hint", "") or ""
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                hint=hint,
                duration_ms=duration_ms,
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
