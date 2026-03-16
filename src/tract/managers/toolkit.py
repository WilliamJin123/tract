"""Toolkit manager for Tract.

Extracted from Tract (tract.py) into a standalone class with explicit
constructor dependencies.  Handles tool definition resolution, profile
switching, tool locking/unlocking, custom tool registration, and format
conversion for LLM consumption.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tract.toolkit.executor import ToolExecutor
    from tract.toolkit.models import ToolName, ToolProfile
    from tract.toolkit.profiles import ProfileName


# Sentinel for "use the default profile"
_PROFILE_SENTINEL = object()


class ToolkitManager:
    """Tool definitions, profiles, locking, custom tools, format conversion."""

    def __init__(
        self,
        tract_ref: Any,  # The Tract instance (needed by get_all_tools/get_compact_tools)
        check_open: Callable[[], None],
    ) -> None:
        self._tract_ref = tract_ref
        self._check_open = check_open

        # Owned state
        self._tool_profile: str | ToolProfile | None = None
        self._tool_executor: ToolExecutor | None = None
        self._custom_tools: dict[str, Any] = {}  # name -> ToolDefinition
        self._tool_result_format: Literal["minimal", "json", "verbose"] = "minimal"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def custom_tools(self) -> dict[str, Any]:
        """Read-only view of registered custom tools (name -> ToolDefinition)."""
        return dict(self._custom_tools)

    # ------------------------------------------------------------------
    # Internal tool resolution
    # ------------------------------------------------------------------

    def _resolve_tools(
        self,
        *,
        profile: ProfileName | ToolProfile | str = "self",
        tool_names: list[ToolName | str] | None = None,
        overrides: dict[ToolName | str, str] | None = None,
    ) -> list:
        """Shared tool resolution: profile filtering, name filtering, overrides.

        Returns list of ToolDefinition objects.
        """
        from tract.toolkit.definitions import get_all_tools
        from tract.toolkit.models import ToolProfile as _ToolProfile
        from tract.toolkit.profiles import get_profile

        # Resolve profile
        if isinstance(profile, str):
            resolved_profile = get_profile(profile)
        elif isinstance(profile, _ToolProfile):
            resolved_profile = profile
        else:
            raise TypeError(
                f"profile must be a string or ToolProfile, got {type(profile).__name__}"
            )

        # Special case: compact profile generates domain-grouped tools
        if resolved_profile.name == "compact":
            from tract.toolkit.compact import ACTION_TO_DOMAIN, get_compact_tools

            compact_tools = get_compact_tools(self._tract_ref)
            if tool_names is not None:
                allowed = set(tool_names)
                # Translate action-level names (e.g. "commit", "status") to
                # the compact domain tools that contain them (e.g.
                # "tract_context").  This lets callers use the same tool_names
                # regardless of whether the profile is "full" or "compact".
                expanded = set(allowed)  # keep any direct compact names
                for name in allowed:
                    domain = ACTION_TO_DOMAIN.get(name)
                    if domain is not None:
                        expanded.add(f"tract_{domain}")
                compact_tools = [t for t in compact_tools if t.name in expanded]
            if overrides:
                compact_tools = [
                    replace(t, description=overrides[t.name])
                    if t.name in overrides else t
                    for t in compact_tools
                ]
            return compact_tools

        all_tools = get_all_tools(self._tract_ref)

        # Apply profile filtering
        filtered = resolved_profile.filter_tools(all_tools)

        # Include dynamic operation tools (not in static profile configs)
        filtered_names = {t.name for t in filtered}
        for tool in all_tools:
            if tool.name.startswith("fire_") and tool.name not in filtered_names:
                filtered.append(tool)

        # Filter to specific tool names if requested
        if tool_names is not None:
            allowed = set(tool_names)
            filtered = [t for t in filtered if t.name in allowed]

        # Apply description overrides
        if overrides:
            new_filtered = []
            for tool in filtered:
                if tool.name in overrides:
                    tool = replace(tool, description=overrides[tool.name])
                new_filtered.append(tool)
            filtered = new_filtered

        # Append custom tools registered via @t.tool
        if self._custom_tools:
            existing_names = {t.name for t in filtered}
            for ct in self._custom_tools.values():
                if ct.name not in existing_names:
                    # Apply tool_names filter if active
                    if tool_names is not None and ct.name not in set(tool_names):
                        continue
                    filtered.append(ct)

        return filtered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def as_tools(
        self,
        *,
        profile: ProfileName | ToolProfile | str | object = _PROFILE_SENTINEL,
        tool_names: list[ToolName | str] | None = None,
        overrides: dict[ToolName | str, str] | None = None,
        format: Literal["openai", "anthropic"] = "openai",
    ) -> list[dict]:
        """Get tool definitions for this tract in LLM-consumable format.

        Combines tool definitions, profile filtering, optional description
        overrides, and format conversion in one call.

        Args:
            profile: A profile name (``"compact"``, ``"self"``, ``"supervisor"``,
                ``"full"``) or a :class:`~tract.toolkit.models.ToolProfile`
                instance.  Falls back to ``tool_profile`` from :meth:`open`,
                then ``"compact"``.
            tool_names: Optional list of tool names to include. When provided,
                only tools whose names are in this list are returned (applied
                after profile filtering).
            overrides: Optional dict mapping tool names to replacement
                descriptions.  Applied on top of the profile's descriptions.
            format: Output format -- ``"openai"`` (default) or ``"anthropic"``.

        Returns:
            List of tool definition dicts in the requested format.
        """
        effective_profile = (
            self._tool_profile or "compact"
        ) if profile is _PROFILE_SENTINEL else profile
        filtered = self._resolve_tools(
            profile=effective_profile, tool_names=tool_names, overrides=overrides,
        )
        if format == "openai":
            return [tool.to_openai() for tool in filtered]
        elif format == "anthropic":
            return [tool.to_anthropic() for tool in filtered]
        else:
            raise ValueError(
                f"Unknown format '{format}'. Supported: 'openai', 'anthropic'."
            )

    def as_callable_tools(
        self,
        *,
        profile: ProfileName | ToolProfile | str | object = _PROFILE_SENTINEL,
        tool_names: list[ToolName | str] | None = None,
        overrides: dict[ToolName | str, str] | None = None,
    ) -> list:
        """Get tools as typed Python callables for framework integration.

        Returns tract tools as functions with proper ``__name__``, ``__doc__``,
        ``__signature__``, and type annotations.  Works with any framework
        that introspects callables: Agno, LangChain, CrewAI, LangGraph, etc.

        Args:
            profile: A profile name (``"compact"``, ``"self"``, ``"supervisor"``,
                ``"full"``) or a :class:`~tract.toolkit.models.ToolProfile`
                instance.  Falls back to ``tool_profile`` from :meth:`open`,
                then ``"compact"``.
            tool_names: Optional list of tool names to include.
            overrides: Optional dict mapping tool names to replacement
                descriptions.  Applied on top of the profile's descriptions.

        Returns:
            List of typed Python callables, one per tool.
        """
        from tract.toolkit.callables import tools_to_callables

        effective_profile = (
            self._tool_profile or "compact"
        ) if profile is _PROFILE_SENTINEL else profile
        filtered = self._resolve_tools(
            profile=effective_profile, tool_names=tool_names, overrides=overrides,
        )
        return tools_to_callables(filtered)

    def switch_profile(self, profile: ProfileName | ToolProfile | str) -> None:
        """Switch the active tool profile.

        Changes which tools are available for the current session.
        Clears any per-tool overrides.

        Args:
            profile: Profile name (``"self"``, ``"supervisor"``, ``"full"``) or
                a ToolProfile instance.
        """
        if self._tool_executor is None:
            from tract.toolkit.executor import ToolExecutor
            self._tool_executor = ToolExecutor(self._tract_ref)
        self._tool_executor.set_profile(profile)

    def lock_tool(self, tool_name: ToolName | str) -> None:
        """Force-disable a tool regardless of current profile.

        Args:
            tool_name: Name of the tool to lock.
        """
        if self._tool_executor is None:
            from tract.toolkit.executor import ToolExecutor
            self._tool_executor = ToolExecutor(self._tract_ref)
        self._tool_executor.lock_tool(tool_name)

    def unlock_tool(self, tool_name: ToolName | str) -> None:
        """Force-enable a tool regardless of current profile.

        Args:
            tool_name: Name of the tool to unlock.
        """
        if self._tool_executor is None:
            from tract.toolkit.executor import ToolExecutor
            self._tool_executor = ToolExecutor(self._tract_ref)
        self._tool_executor.unlock_tool(tool_name)

    def tool(
        self,
        fn: Any | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a custom tool from a typed Python function.

        Works as a decorator (with or without arguments)::

            @t.tool
            def search(query: str) -> str:
                \"\"\"Search the database.\"\"\"
                ...

            @t.tool(name="calc", description="Math evaluator")
            def calculator(expression: str) -> str:
                ...

        Registered tools are automatically included in :meth:`run` and
        :meth:`as_tools` alongside tract's built-in tools.

        Args:
            fn: The function to register (when used as ``@t.tool``
                without parentheses).
            name: Override the tool name (defaults to ``fn.__name__``).
            description: Override the description (defaults to the first
                line of the docstring).

        Returns:
            The original function (unmodified), or a decorator if called
            with keyword arguments.
        """
        from tract.toolkit.callables import callable_to_tool

        def _register(func: Any) -> Any:
            tool_def = callable_to_tool(func, name=name, description=description)
            self._custom_tools[tool_def.name] = tool_def
            return func

        if fn is not None:
            # Used as @t.tool (no parentheses)
            return _register(fn)
        # Used as @t.tool(...) (with parentheses)
        return _register

    def remove_tool(self, tool_name: str) -> None:
        """Unregister a custom tool previously added via :meth:`tool`.

        Args:
            tool_name: Name of the custom tool to remove.

        Raises:
            KeyError: If no custom tool with that name is registered.
        """
        if tool_name not in self._custom_tools:
            available = ", ".join(sorted(self._custom_tools.keys())) or "(none)"
            raise KeyError(
                f"No custom tool '{tool_name}'. Registered: {available}"
            )
        del self._custom_tools[tool_name]
