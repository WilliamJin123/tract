"""Tests for simplified toolkit: new tools, direct dispatch, no orchestrator coupling."""

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
    def test_create_rule_exists(self, tract_instance):
        tools = get_all_tools(tract_instance)
        names = {t.name for t in tools}
        assert "create_rule" in names

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

    def test_total_tool_count(self, tract_instance):
        """Should have 26 tools (22 original + 4 new)."""
        tools = get_all_tools(tract_instance)
        assert len(tools) == 26


# ---------------------------------------------------------------------------
# Tool execution (direct dispatch)
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_executor_direct_dispatch(self, tract_instance):
        """Executor dispatches directly to handler, no hook routing."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute("status", {})
        assert result.success
        assert "Branch:" in result.output

    def test_create_rule_tool(self, tract_instance):
        """create_rule tool creates a rule commit."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "create_rule",
            {
                "name": "test-rule",
                "trigger": "active",
                "action": {"type": "set_config", "key": "foo", "value": "bar"},
            },
        )
        assert result.success
        assert "test-rule" in result.output

    def test_create_rule_with_condition(self, tract_instance):
        """create_rule with a condition dict."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute(
            "create_rule",
            {
                "name": "conditional-rule",
                "trigger": "active",
                "condition": {"token_budget_exceeded": True},
                "action": {"type": "set_config", "key": "compress", "value": True},
            },
        )
        assert result.success
        assert "conditional-rule" in result.output

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
        """get_config when no rules are set returns 'not set'."""
        executor = ToolExecutor(tract_instance)
        result = executor.execute("get_config", {"key": "nonexistent"})
        assert result.success
        assert "not set" in result.output

    def test_get_config_tool_with_rule(self, tract_instance):
        """get_config resolves value from active rules."""
        # set_config action format: {"type": "set_config", "key": ..., "value": ...}
        tract_instance.rule(
            name="set-model",
            trigger="active",
            action={"type": "set_config", "key": "model", "value": "gpt-4o"},
        )
        executor = ToolExecutor(tract_instance)
        result = executor.execute("get_config", {"key": "model"})
        assert result.success
        assert "gpt-4o" in result.output

    def test_transition_tool_no_rules(self, tract_instance):
        """transition without rules just transitions."""
        # Need some content first
        tract_instance.system("Hello")
        executor = ToolExecutor(tract_instance)
        result = executor.execute("transition", {"target": "feature-x"})
        assert result.success
        assert "feature-x" in result.output

    def test_transition_tool_blocked(self, tract_instance):
        """transition blocked by rules returns blocked message."""
        tract_instance.system("Setup")
        # Create a blocking rule
        tract_instance.rule(
            name="block-transition",
            trigger="transition:blocked-branch",
            action={"type": "block", "reason": "Not allowed"},
        )
        executor = ToolExecutor(tract_instance)
        result = executor.execute("transition", {"target": "blocked-branch"})
        assert result.success  # Tool execution succeeds
        assert "blocked" in result.output.lower()


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_all_tool_names_includes_new(self):
        """_ALL_TOOL_NAMES includes the 4 new tools."""
        assert "create_rule" in _ALL_TOOL_NAMES
        assert "create_metadata" in _ALL_TOOL_NAMES
        assert "get_config" in _ALL_TOOL_NAMES
        assert "transition" in _ALL_TOOL_NAMES

    def test_self_profile_includes_new(self, tract_instance):
        tools = get_all_tools(tract_instance)
        filtered = SELF_PROFILE.filter_tools(tools)
        names = {t.name for t in filtered}
        assert "create_rule" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names

    def test_supervisor_profile_includes_new(self, tract_instance):
        tools = get_all_tools(tract_instance)
        filtered = SUPERVISOR_PROFILE.filter_tools(tools)
        names = {t.name for t in filtered}
        assert "create_rule" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names

    def test_full_profile_includes_new(self, tract_instance):
        tools = get_all_tools(tract_instance)
        filtered = FULL_PROFILE.filter_tools(tools)
        names = {t.name for t in filtered}
        assert "create_rule" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names

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
        tools = tract_instance.as_tools(format="openai")
        names = {t["function"]["name"] for t in tools}
        assert "create_rule" in names
        assert "get_config" in names

    def test_as_tools_full_profile(self, tract_instance):
        tools = tract_instance.as_tools(profile="full", format="openai")
        names = {t["function"]["name"] for t in tools}
        assert "create_rule" in names
        assert "create_metadata" in names
        assert "get_config" in names
        assert "transition" in names

    def test_as_callable_tools(self, tract_instance):
        """as_callable_tools includes new tools."""
        callables = tract_instance.as_callable_tools()
        names = {c.__name__ for c in callables}
        assert "create_rule" in names
        assert "get_config" in names
