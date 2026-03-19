"""Tests for the compact tool profile and discover meta-tool.

Verifies that:
- Compact profile produces 7 domain-grouped tools (6 domains + discover)
- Domain tools dispatch correctly to underlying individual tools
- Discover meta-tool returns action lists and parameter schemas
- All three consumption paths work (openai, anthropic, callable)
- Token savings are real (compact << full schema size)
"""

from __future__ import annotations

import json

import pytest

from tract import DialogueContent, InstructionContent, Tract
from tract.toolkit import (
    COMPACT_DOMAINS,
    COMPACT_PROFILE,
    get_compact_tools,
    get_profile,
    ToolExecutor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tract(tmp_path):
    t = Tract.open(str(tmp_path / "test.db"))
    yield t
    t.close()


@pytest.fixture()
def tract_with_commits(tract):
    tract.commit(InstructionContent(text="You are helpful."), message="system")
    tract.commit(DialogueContent(role="user", text="Hello"), message="greeting")
    tract.commit(
        DialogueContent(role="assistant", text="Hi there!"), message="reply"
    )
    return tract


# ---------------------------------------------------------------------------
# Profile registration
# ---------------------------------------------------------------------------


class TestCompactProfileRegistration:
    def test_profile_lookup(self):
        profile = get_profile("compact")
        assert profile.name == "compact"
        assert profile is COMPACT_PROFILE

    def test_profile_in_literal(self):
        from tract.toolkit.profiles import ProfileName
        from typing import get_args

        assert "compact" in get_args(ProfileName)


# ---------------------------------------------------------------------------
# Compact tool generation
# ---------------------------------------------------------------------------


class TestCompactToolGeneration:
    def test_generates_seven_tools(self, tract):
        tools = get_compact_tools(tract)
        assert len(tools) == 7

    def test_domain_tool_names(self, tract):
        tools = get_compact_tools(tract)
        names = {t.name for t in tools}
        expected = {
            "tract_context",
            "tract_branch",
            "tract_annotate",
            "tract_tag",
            "tract_config",
            "tract_middleware",
            "tract_discover",
        }
        assert names == expected

    def test_domain_tools_have_action_enum(self, tract):
        tools = get_compact_tools(tract)
        for tool in tools:
            if tool.name == "tract_discover":
                continue
            action_prop = tool.parameters["properties"]["action"]
            assert "enum" in action_prop
            domain = tool.name.removeprefix("tract_")
            assert action_prop["enum"] == COMPACT_DOMAINS[domain]

    def test_domain_tools_have_params_property(self, tract):
        tools = get_compact_tools(tract)
        for tool in tools:
            if tool.name == "tract_discover":
                continue
            assert "params" in tool.parameters["properties"]
            assert tool.parameters["properties"]["params"]["type"] == "object"

    def test_discover_has_domain_enum(self, tract):
        tools = get_compact_tools(tract)
        discover = next(t for t in tools if t.name == "tract_discover")
        domain_prop = discover.parameters["properties"]["domain"]
        assert "enum" in domain_prop
        assert set(domain_prop["enum"]) == set(COMPACT_DOMAINS.keys())


# ---------------------------------------------------------------------------
# Schema size reduction
# ---------------------------------------------------------------------------


class TestTokenSavings:
    def test_compact_fewer_schemas_than_full(self, tract):
        full_tools = tract.runtime.tools.as_tools(profile="full")
        compact_tools = tract.runtime.tools.as_tools(profile="compact")
        assert len(compact_tools) < len(full_tools)
        assert len(compact_tools) == 7

    def test_compact_smaller_json_than_full(self, tract):
        full_json = json.dumps(tract.runtime.tools.as_tools(profile="full"))
        compact_json = json.dumps(tract.runtime.tools.as_tools(profile="compact"))
        # Compact should be significantly smaller
        assert len(compact_json) < len(full_json) * 0.5


# ---------------------------------------------------------------------------
# Domain tool dispatch
# ---------------------------------------------------------------------------


class TestDomainDispatch:
    def test_context_status(self, tract_with_commits):
        tools = get_compact_tools(tract_with_commits)
        ctx_tool = next(t for t in tools if t.name == "tract_context")
        result = ctx_tool.handler(action="status")
        assert "main" in result  # branch name
        assert "HEAD" in result or "head" in result.lower()

    def test_context_log(self, tract_with_commits):
        tools = get_compact_tools(tract_with_commits)
        ctx_tool = next(t for t in tools if t.name == "tract_context")
        result = ctx_tool.handler(action="log", params={"limit": 2})
        assert "greeting" in result or "reply" in result

    def test_context_compile(self, tract_with_commits):
        tools = get_compact_tools(tract_with_commits)
        ctx_tool = next(t for t in tools if t.name == "tract_context")
        result = ctx_tool.handler(action="compile")
        assert "token" in result.lower() or "message" in result.lower()

    def test_branch_list(self, tract_with_commits):
        tools = get_compact_tools(tract_with_commits)
        branch_tool = next(t for t in tools if t.name == "tract_branch")
        result = branch_tool.handler(action="list_branches")
        assert "main" in result

    def test_branch_create_and_switch(self, tract_with_commits):
        tools = get_compact_tools(tract_with_commits)
        branch_tool = next(t for t in tools if t.name == "tract_branch")
        result = branch_tool.handler(
            action="branch", params={"name": "test-branch"}
        )
        assert "test-branch" in result or "Created" in result or "branch" in result.lower()

    def test_config_get(self, tract_with_commits):
        tools = get_compact_tools(tract_with_commits)
        config_tool = next(t for t in tools if t.name == "tract_config")
        result = config_tool.handler(
            action="get_config", params={"key": "model"}
        )
        # May return None/null — just shouldn't error
        assert isinstance(result, str)

    def test_unknown_action_error(self, tract):
        tools = get_compact_tools(tract)
        ctx_tool = next(t for t in tools if t.name == "tract_context")
        result = ctx_tool.handler(action="nonexistent")
        assert "Unknown action" in result

    def test_params_default_to_empty(self, tract):
        """Calling without params should work for no-arg actions."""
        tools = get_compact_tools(tract)
        ctx_tool = next(t for t in tools if t.name == "tract_context")
        result = ctx_tool.handler(action="status")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Discover meta-tool
# ---------------------------------------------------------------------------


class TestDiscoverTool:
    def test_list_domain_actions(self, tract):
        tools = get_compact_tools(tract)
        discover = next(t for t in tools if t.name == "tract_discover")
        result = json.loads(discover.handler(domain="context"))
        actions = [item["action"] for item in result]
        assert "commit" in actions
        assert "compile" in actions
        assert "status" in actions

    def test_list_domain_has_descriptions(self, tract):
        tools = get_compact_tools(tract)
        discover = next(t for t in tools if t.name == "tract_discover")
        result = json.loads(discover.handler(domain="context"))
        for item in result:
            assert "description" in item
            assert len(item["description"]) > 10

    def test_action_detail(self, tract):
        tools = get_compact_tools(tract)
        discover = next(t for t in tools if t.name == "tract_discover")
        result = json.loads(discover.handler(domain="context", action="commit"))
        assert "action" in result
        assert result["action"] == "commit"
        assert "description" in result
        assert "parameters" in result
        assert "properties" in result["parameters"]

    def test_unknown_domain(self, tract):
        tools = get_compact_tools(tract)
        discover = next(t for t in tools if t.name == "tract_discover")
        result = discover.handler(domain="nonexistent")
        assert "Unknown domain" in result

    def test_unknown_action_in_domain(self, tract):
        tools = get_compact_tools(tract)
        discover = next(t for t in tools if t.name == "tract_discover")
        result = discover.handler(domain="context", action="nonexistent")
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# Consumption paths (as_tools, as_callable_tools)
# ---------------------------------------------------------------------------


class TestConsumptionPaths:
    def test_as_tools_openai(self, tract):
        tools = tract.runtime.tools.as_tools(profile="compact", format="openai")
        assert len(tools) == 7
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]

    def test_as_tools_anthropic(self, tract):
        tools = tract.runtime.tools.as_tools(profile="compact", format="anthropic")
        assert len(tools) == 7
        for tool in tools:
            assert "name" in tool
            assert "input_schema" in tool

    def test_as_callable_tools(self, tract):
        callables = tract.runtime.tools.as_callable_tools(profile="compact")
        assert len(callables) == 7
        names = {c.__name__ for c in callables}
        assert "tract_context" in names
        assert "tract_discover" in names


# ---------------------------------------------------------------------------
# Executor compact mode
# ---------------------------------------------------------------------------


class TestExecutorCompact:
    def test_executor_compact_profile(self, tract):
        executor = ToolExecutor(tract)
        executor.set_profile("compact")
        assert set(executor.available_tools()) == {
            "tract_context",
            "tract_branch",
            "tract_annotate",
            "tract_tag",
            "tract_config",
            "tract_middleware",
            "tract_discover",
        }

    def test_executor_dispatch_status(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        executor.set_profile("compact")
        result = executor.execute(
            "tract_context", {"action": "status"}
        )
        assert result.success
        assert "main" in result.output

    def test_executor_dispatch_discover(self, tract):
        executor = ToolExecutor(tract)
        executor.set_profile("compact")
        result = executor.execute(
            "tract_discover", {"domain": "branch"}
        )
        assert result.success
        parsed = json.loads(result.output)
        actions = [item["action"] for item in parsed]
        assert "switch" in actions
        assert "merge" in actions


# ---------------------------------------------------------------------------
# Domain coverage — all 29 tools are reachable
# ---------------------------------------------------------------------------


class TestActionNameTranslation:
    """Verify that passing full-profile action names (e.g. 'commit', 'status')
    transparently selects the correct compact domain tools."""

    def test_action_names_resolve_to_domain_tools(self, tract):
        """tool_names=["commit", "status"] should include tract_context."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=["commit", "status"],
        )
        names = {t["function"]["name"] for t in tools}
        assert "tract_context" in names

    def test_action_names_across_domains(self, tract):
        """Names from different domains should include all relevant domains."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=["commit", "branch", "tag"],
        )
        names = {t["function"]["name"] for t in tools}
        assert "tract_context" in names
        assert "tract_branch" in names
        assert "tract_tag" in names

    def test_action_names_no_extra_domains(self, tract):
        """Only relevant domains should be included, not all 7."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=["commit", "status"],
        )
        names = {t["function"]["name"] for t in tools}
        # Should include tract_context but not unrelated domains
        assert "tract_context" in names
        assert "tract_middleware" not in names

    def test_compact_domain_names_still_work(self, tract):
        """Direct compact names like 'tract_context' should still work."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=["tract_context", "tract_branch"],
        )
        names = {t["function"]["name"] for t in tools}
        assert names == {"tract_context", "tract_branch"}

    def test_mixed_action_and_domain_names(self, tract):
        """Mix of action names and compact domain names should work."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=["commit", "tract_branch"],
        )
        names = {t["function"]["name"] for t in tools}
        assert "tract_context" in names
        assert "tract_branch" in names

    def test_discover_always_excluded_unless_named(self, tract):
        """tract_discover should not appear unless explicitly requested."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=["commit"],
        )
        names = {t["function"]["name"] for t in tools}
        assert "tract_discover" not in names

    def test_empty_tool_names_returns_nothing(self, tract):
        """Empty tool_names list should return no tools."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=[],
        )
        assert tools == []

    def test_unrecognized_names_silently_ignored(self, tract):
        """Names that match neither action nor domain are silently skipped."""
        tools = tract.runtime.tools.as_tools(
            profile="compact",
            tool_names=["nonexistent_tool"],
        )
        assert tools == []


class TestDomainCoverage:
    def test_all_individual_tools_mapped(self, tract):
        from tract.toolkit.definitions import get_all_tools
        from tract.toolkit.compact import ACTION_TO_DOMAIN

        all_tools = get_all_tools(tract)
        for tool in all_tools:
            assert tool.name in ACTION_TO_DOMAIN, (
                f"Tool '{tool.name}' not mapped to any compact domain"
            )
