"""Tests for simplified toolkit: new tools, direct dispatch, no framework coupling."""

from __future__ import annotations

import pytest

from tract.toolkit.definitions import get_all_tools
from tract.toolkit.executor import ToolExecutor
from tract.toolkit.models import ToolDefinition
from tract.toolkit.profiles import (
    FULL_PROFILE,
    SELF_PROFILE,
    SUPERVISOR_PROFILE,
    _ALL_TOOL_NAMES,
    get_profile,
)


@pytest.fixture()
def tract_instance(tmp_path):
    from tract import Tract

    db = str(tmp_path / "test.db")
    return Tract.open(db)


# ---------------------------------------------------------------------------
# New tool definitions
# ---------------------------------------------------------------------------


class TestNewToolDefinitions:
    def test_configure_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "configure" in names

    def test_create_metadata_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "create_metadata" in names

    def test_get_config_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "get_config" in names

    def test_transition_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "transition" in names

    def test_directive_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "directive" in names

    def test_create_middleware_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "create_middleware" in names

    def test_remove_middleware_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "remove_middleware" in names

    def test_total_tool_count(self, tract_instance):
        """Should have 29 tools (22 original + 4 new + 3 config/middleware)."""
        tools = get_all_tools(tract_instance)
        assert len(tools) == 29


# ---------------------------------------------------------------------------
# Tool execution (direct dispatch)
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_executor_direct_dispatch(self, tract_instance):
        """Executor dispatches directly to handler, no middleware routing."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute("status", {})
        assert result.success
        assert "Branch:" in result.output

    def test_configure_tool(self, tract_instance):
        """configure tool creates a config commit."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "configure",
            {
                "settings": {"model": "gpt-4o", "temperature": 0.7},
            },
        )
        assert result.success
        assert "Configured" in result.output

    def test_configure_tool_single_key(self, tract_instance):
        """configure tool with a single key-value setting."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "configure",
            {
                "settings": {"compile_strategy": "adaptive"},
            },
        )
        assert result.success
        assert "compile_strategy" in result.output

    def test_create_metadata_tool(self, tract_instance):
        """create_metadata tool creates a metadata commit."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "create_metadata",
            {
                "kind": "file_tree",
                "data": {"root": "/", "files": ["a.py", "b.py"]},
            },
        )
        assert result.success
        assert "file_tree" in result.output

    def test_create_metadata_with_path(self, tract_instance):
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "create_metadata",
            {
                "kind": "config",
                "data": {"key": "value"},
                "path": "/etc/config.yml",
            },
        )
        assert result.success

    def test_get_config_tool_not_set(self, tract_instance):
        """get_config when no config is set returns 'not set'."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute("get_config", {"key": "nonexistent"})
        assert result.success
        assert "not set" in result.output

    def test_get_config_tool_with_configure(self, tract_instance):
        """get_config resolves value from DAG config commits."""
        tract_instance.config.set(model="gpt-4o")
        executor = ToolExecutor(tract_instance)
        result = executor.execute("get_config", {"key": "model"})
        assert result.success
        assert "gpt-4o" in result.output

    def test_transition_tool_no_middleware(self, tract_instance):
        """transition without middleware just transitions."""
        # Need some content first
        tract_instance.system("Hello")
        executor = ToolExecutor(tract_instance)
        result = executor.execute("transition", {"target": "feature-x"})
        assert result.success
        assert "feature-x" in result.output

    def test_transition_tool_blocked(self, tract_instance):
        """transition blocked by middleware returns error with blocked message."""
        from tract.exceptions import BlockedError

        tract_instance.system("Setup")
        # Register a pre_transition handler that blocks
        def block_transition(ctx):
            if ctx.target == "blocked-branch":
                raise BlockedError("pre_transition", "Not allowed")

        tract_instance.middleware.add("pre_transition", block_transition)
        executor = ToolExecutor(tract_instance)
        result = executor.execute("transition", {"target": "blocked-branch"})
        assert not result.success  # Blocked raises, caught by executor
        assert "blocked" in result.error.lower()

    def test_directive_tool(self, tract_instance):
        """directive tool creates a named instruction commit."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "directive",
            {"name": "tone", "text": "Be concise and direct."},
        )
        assert result.success
        assert "tone" in result.output

    def test_directive_tool_with_priority(self, tract_instance):
        """directive tool with explicit priority."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "directive",
            {"name": "safety", "text": "Never share secrets.", "priority": "normal"},
        )
        assert result.success
        assert "normal" in result.output

    def test_directive_dedup_via_compile(self, tract_instance):
        """Two directives with same name: only latest in compiled context."""
        tract_instance.directive("proto", "Old protocol")
        tract_instance.directive("proto", "New protocol")
        compiled = tract_instance.compile()
        texts = [m.content for m in compiled.messages]
        assert any("New protocol" in t for t in texts)
        assert not any("Old protocol" in t for t in texts)

    def test_create_middleware_tool(self, tract_instance):
        """create_middleware registers a handler from Python code."""
        executor = ToolExecutor(tract_instance)
        code = "def handler(ctx):\n    pass  # no-op middleware"
        result = executor.execute(
            "create_middleware",
            {"event": "post_commit", "code": code, "description": "no-op"},
        )
        assert result.success
        assert "post_commit" in result.output
        assert "no-op" in result.output

    def test_create_middleware_blocks_commit(self, tract_instance):
        """create_middleware can create a handler that blocks commits."""
        executor = ToolExecutor(tract_instance)
        # pre_commit middleware receives ctx.pending (content model), not ctx.commit.
        # Block when pending content text contains "BLOCKED".
        code = (
            "def handler(ctx):\n"
            '    pending = ctx.pending\n'
            '    if pending and hasattr(pending, "text") and "BLOCKED" in (pending.text or ""):\n'
            '        raise BlockedError("pre_commit", "Blocked by middleware")\n'
        )
        result = executor.execute(
            "create_middleware",
            {"event": "pre_commit", "code": code},
        )
        assert result.success

        # Normal commit should work
        tract_instance.user("Normal message")

        # Commit with BLOCKED in text should fail
        from tract.exceptions import BlockedError

        with pytest.raises(BlockedError):
            tract_instance.commit(
                {"content_type": "dialogue", "role": "user", "text": "BLOCKED content"},
                message="test commit",
            )

    def test_create_middleware_syntax_error(self, tract_instance):
        """create_middleware returns error for syntax errors."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "create_middleware",
            {"event": "post_commit", "code": "def handler(ctx)\n    pass"},
        )
        assert result.success  # executor returns success, error in output
        assert "ERROR" in result.output
        assert "Syntax error" in result.output

    def test_create_middleware_no_handler(self, tract_instance):
        """create_middleware returns error if handler() not defined."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "create_middleware",
            {"event": "post_commit", "code": "def my_func(ctx): pass"},
        )
        assert result.success
        assert "ERROR" in result.output
        assert "handler" in result.output

    def test_create_middleware_restricted_imports(self, tract_instance):
        """create_middleware blocks dangerous imports."""
        executor = ToolExecutor(tract_instance)
        code = "import os\ndef handler(ctx): pass"
        result = executor.execute(
            "create_middleware",
            {"event": "post_commit", "code": code},
        )
        assert result.success
        assert "ERROR" in result.output

    def test_create_middleware_uses_re(self, tract_instance):
        """create_middleware allows re module (available as global, not via import)."""
        executor = ToolExecutor(tract_instance)
        code = (
            "def handler(ctx):\n"
            '    if ctx.commit and ctx.commit.message:\n'
            '        if not re.match(r"^[A-Z]", ctx.commit.message):\n'
            '            raise BlockedError("pre_commit", "Must start with uppercase")\n'
        )
        result = executor.execute(
            "create_middleware",
            {"event": "pre_commit", "code": code},
        )
        assert result.success
        assert "pre_commit" in result.output

    def test_remove_middleware_tool(self, tract_instance):
        """remove_middleware removes a handler by ID."""
        # First create one
        handler_id = tract_instance.middleware.add("post_commit", lambda ctx: None)
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "remove_middleware",
            {"handler_id": handler_id},
        )
        assert result.success
        assert "removed" in result.output

    def test_remove_middleware_invalid_id(self, tract_instance):
        """remove_middleware returns error for unknown ID."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "remove_middleware",
            {"handler_id": "nonexistent_id"},
        )
        assert result.success  # executor catches errors, returned in output
        assert "ERROR" in result.output


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_all_tool_names_includes_new(self):
        """_ALL_TOOL_NAMES includes the 4+3 new tools."""
        assert "configure" in _ALL_TOOL_NAMES
        assert "create_metadata" in _ALL_TOOL_NAMES
        assert "get_config" in _ALL_TOOL_NAMES
        assert "transition" in _ALL_TOOL_NAMES
        assert "directive" in _ALL_TOOL_NAMES
        assert "create_middleware" in _ALL_TOOL_NAMES
        assert "remove_middleware" in _ALL_TOOL_NAMES

    def test_self_profile_includes_new(self, tract_instance):
        tools = get_all_tools(tract_instance)
        filtered = SELF_PROFILE.filter_tools(tools)
        names = {t.name for t in filtered}
        assert "configure" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names
        assert "directive" in names
        assert "create_middleware" in names
        assert "remove_middleware" in names

    def test_supervisor_profile_includes_new(self, tract_instance):
        tools = get_all_tools(tract_instance)
        filtered = SUPERVISOR_PROFILE.filter_tools(tools)
        names = {t.name for t in filtered}
        assert "configure" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names
        assert "directive" in names
        assert "create_middleware" in names
        assert "remove_middleware" in names

    def test_full_profile_includes_new(self, tract_instance):
        tools = get_all_tools(tract_instance)
        filtered = FULL_PROFILE.filter_tools(tools)
        names = {t.name for t in filtered}
        assert "configure" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names
        assert "directive" in names
        assert "create_middleware" in names
        assert "remove_middleware" in names

    def test_profiles_still_work(self, tract_instance):
        """Existing profiles continue to work after adding new tools."""
        for name in ["self", "supervisor", "full"]:
            profile = get_profile(name)
            tools = get_all_tools(tract_instance)
            filtered = profile.filter_tools(tools)
            assert len(filtered) > 0


# ---------------------------------------------------------------------------
# as_tools / as_callable_tools
# ---------------------------------------------------------------------------


class TestAsTools:
    def test_as_tools_includes_new(self, tract_instance):
        """as_tools() includes new tools in output."""
        tools = tract_instance.runtime.tools.as_tools(profile="self", format="openai")
        names = {t["function"]["name"] for t in tools}
        assert "configure" in names
        assert "get_config" in names
        assert "directive" in names
        assert "create_middleware" in names
        assert "remove_middleware" in names

    def test_as_tools_full_profile(self, tract_instance):
        tools = tract_instance.runtime.tools.as_tools(profile="full", format="openai")
        names = {t["function"]["name"] for t in tools}
        assert "configure" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names
        assert "directive" in names
        assert "create_middleware" in names
        assert "remove_middleware" in names

    def test_as_callable_tools(self, tract_instance):
        """as_callable_tools includes new tools."""
        callables = tract_instance.runtime.tools.as_callable_tools(profile="self")
        names = {c.__name__ for c in callables}
        assert "configure" in names
        assert "get_config" in names
