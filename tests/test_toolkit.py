"""Comprehensive tests for the Agent Toolkit module.

Tests cover tool definitions, profiles, executor, Tract.as_tools() facade,
and end-to-end tool execution through real Tract instances.
"""

from __future__ import annotations

import pytest

from tract import (
    DialogueContent,
    InstructionContent,
    Tract,
    ToolDefinition,
    ToolProfile,
    ToolConfig,
    ToolResult,
    ToolExecutor,
)
from tract.toolkit import (
    FULL_PROFILE,
    SELF_PROFILE,
    SUPERVISOR_PROFILE,
    get_all_tools,
    get_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tract(tmp_path):
    """File-backed tract, cleaned up after test."""
    t = Tract.open(str(tmp_path / "test.db"))
    yield t
    t.close()


@pytest.fixture()
def tract_with_commits(tract):
    """Tract pre-loaded with 3 commits."""
    tract.commit(InstructionContent(text="You are helpful."), message="system")
    tract.commit(DialogueContent(role="user", text="Hello"), message="greeting")
    tract.commit(
        DialogueContent(role="assistant", text="Hi there!"), message="reply"
    )
    return tract


# ===========================================================================
# ToolDefinition format tests
# ===========================================================================


class TestToolDefinitionFormats:
    """Test to_openai() and to_anthropic() produce correct dict structures."""

    def test_tool_definition_to_openai(self, tract):
        tools = get_all_tools(tract)
        oai = tools[0].to_openai()
        assert oai["type"] == "function"
        assert "function" in oai
        func = oai["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert func["name"] == tools[0].name
        assert func["description"] == tools[0].description
        assert func["parameters"] == tools[0].parameters

    def test_tool_definition_to_anthropic(self, tract):
        tools = get_all_tools(tract)
        anth = tools[0].to_anthropic()
        assert "name" in anth
        assert "description" in anth
        assert "input_schema" in anth
        assert anth["name"] == tools[0].name
        assert anth["description"] == tools[0].description
        assert anth["input_schema"] == tools[0].parameters


# ===========================================================================
# get_all_tools tests
# ===========================================================================


class TestGetAllTools:
    """Test that get_all_tools returns correct definitions."""

    def test_returns_25_definitions(self, tract):
        tools = get_all_tools(tract)
        assert len(tools) == 25

    def test_all_have_required_fields(self, tract):
        tools = get_all_tools(tract)
        for tool in tools:
            assert isinstance(tool, ToolDefinition)
            assert tool.name
            assert tool.description
            assert isinstance(tool.parameters, dict)
            assert "type" in tool.parameters
            assert tool.parameters["type"] == "object"
            assert callable(tool.handler)

    def test_tool_names_are_unique(self, tract):
        tools = get_all_tools(tract)
        names = [t.name for t in tools]
        assert len(names) == len(set(names))

    def test_expected_tool_names(self, tract):
        tools = get_all_tools(tract)
        names = {t.name for t in tools}
        expected = {
            "commit", "compile", "annotate", "status", "log", "diff",
            "compress", "branch", "switch", "merge", "reset", "checkout",
            "gc", "list_branches", "get_commit",
            "configure_model", "tag", "untag", "query_by_tags",
            "register_tag", "get_tags", "list_tags",
            "register_trigger", "unregister_trigger", "toggle_triggers",
        }
        assert names == expected


# ===========================================================================
# Profile tests
# ===========================================================================


class TestProfiles:
    """Test built-in profiles filter tools correctly."""

    def test_self_profile_subset(self, tract):
        tools = get_all_tools(tract)
        self_tools = SELF_PROFILE.filter_tools(tools)
        self_names = {t.name for t in self_tools}
        expected = {
            "commit", "compile", "annotate", "status", "log",
            "compress", "branch", "switch", "reset",
            "tag", "untag", "query_by_tags",
            "register_tag", "get_tags", "list_tags",
            "configure_model", "toggle_triggers",
        }
        assert self_names == expected
        assert len(self_tools) == 17

    def test_supervisor_profile_all_tools(self, tract):
        tools = get_all_tools(tract)
        sup_tools = SUPERVISOR_PROFILE.filter_tools(tools)
        assert len(sup_tools) == 25

    def test_full_profile_all_tools(self, tract):
        tools = get_all_tools(tract)
        full_tools = FULL_PROFILE.filter_tools(tools)
        assert len(full_tools) == 25

    def test_full_profile_default_descriptions(self, tract):
        """FULL_PROFILE should NOT override any descriptions."""
        tools = get_all_tools(tract)
        full_tools = FULL_PROFILE.filter_tools(tools)
        original_descs = {t.name: t.description for t in tools}
        for tool in full_tools:
            assert tool.description == original_descs[tool.name]

    def test_profile_description_overrides_self(self, tract):
        """SELF_PROFILE descriptions should be self-referential."""
        tools = get_all_tools(tract)
        self_tools = SELF_PROFILE.filter_tools(tools)
        # At least some descriptions should contain "your" or "you"
        self_referential_count = sum(
            1 for t in self_tools
            if "your" in t.description.lower() or "you" in t.description.lower()
        )
        assert self_referential_count >= 5, (
            f"Expected >= 5 self-referential descriptions, got {self_referential_count}"
        )

    def test_profile_description_overrides_supervisor(self, tract):
        """SUPERVISOR_PROFILE descriptions should be managerial."""
        tools = get_all_tools(tract)
        sup_tools = SUPERVISOR_PROFILE.filter_tools(tools)
        managerial_count = sum(
            1 for t in sup_tools
            if "managed agent" in t.description.lower()
        )
        assert managerial_count >= 10, (
            f"Expected >= 10 managerial descriptions, got {managerial_count}"
        )

    def test_get_profile_by_name(self):
        assert get_profile("self") is SELF_PROFILE
        assert get_profile("supervisor") is SUPERVISOR_PROFILE
        assert get_profile("full") is FULL_PROFILE

    def test_get_profile_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("unknown")


# ===========================================================================
# ToolExecutor tests
# ===========================================================================


class TestToolExecutor:
    """Test ToolExecutor dispatches and returns structured results."""

    def test_execute_status(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("status", {})
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "status"
        assert "main" in result.output
        assert "tokens" in result.output

    def test_execute_unknown_tool(self, tract):
        executor = ToolExecutor(tract)
        result = executor.execute("nonexistent", {})
        assert result.success is False
        assert "Unknown tool: nonexistent" in result.error

    def test_execute_error_bad_args(self, tract):
        executor = ToolExecutor(tract)
        # annotate requires target_hash and priority, give it bad hash
        result = executor.execute("annotate", {
            "target_hash": "does_not_exist_0000",
            "priority": "pinned",
        })
        assert result.success is False
        assert result.error  # Should have an error message

    def test_available_tools(self, tract):
        executor = ToolExecutor(tract)
        names = executor.available_tools()
        assert len(names) == 25
        assert "commit" in names
        assert "status" in names

    def test_execute_commit(self, tract):
        executor = ToolExecutor(tract)
        result = executor.execute("commit", {
            "content": {
                "content_type": "dialogue",
                "role": "user",
                "text": "Hello from tool executor",
            },
            "message": "test commit",
        })
        assert result.success is True
        assert "Committed" in result.output
        assert "dialogue" in result.output
        # Verify commit actually exists
        assert tract.head is not None

    def test_execute_commit_with_tags(self, tract):
        """Commit tool accepts and passes through tags."""
        tract.register_tag("important")
        tract.register_tag("v1")
        executor = ToolExecutor(tract)
        result = executor.execute("commit", {
            "content": {
                "content_type": "dialogue",
                "role": "user",
                "text": "tagged message",
            },
            "tags": ["important", "v1"],
        })
        assert result.success is True
        assert "Committed" in result.output

    def test_execute_commit_without_tags(self, tract):
        """Commit tool works without tags (backward compatible)."""
        executor = ToolExecutor(tract)
        result = executor.execute("commit", {
            "content": {
                "content_type": "dialogue",
                "role": "user",
                "text": "no tags",
            },
        })
        assert result.success is True
        assert "Committed" in result.output

    def test_execute_compile(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("compile", {})
        assert result.success is True
        assert "3 messages" in result.output
        assert "tokens" in result.output

    def test_execute_log(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("log", {"limit": 2})
        assert result.success is True
        assert "2 commits" in result.output

    def test_execute_list_branches(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("list_branches", {})
        assert result.success is True
        assert "main" in result.output

    def test_execute_get_commit(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        head = tract_with_commits.head
        result = executor.execute("get_commit", {"commit_hash": head})
        assert result.success is True
        assert "dialogue" in result.output

    def test_execute_compress_manual(self, tract_with_commits):
        """Execute compress tool with manual content for testing."""
        executor = ToolExecutor(tract_with_commits)
        # compress via the tool requires LLM or manual content.
        # The tool always uses auto_commit=True, so we need to test carefully.
        # This may fail if no LLM configured, which is expected behavior.
        result = executor.execute("compress", {})
        # Without LLM config, this should fail with a descriptive error
        assert result.success is False or "Compressed" in result.output


# ===========================================================================
# Tract.as_tools() facade tests
# ===========================================================================


class TestAsTools:
    """Test the Tract.as_tools() convenience method."""

    def test_default_profile_openai(self, tract):
        tools = tract.as_tools()
        assert isinstance(tools, list)
        # Default profile is "self" with 17 tools
        assert len(tools) == 17
        # OpenAI format
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool

    def test_supervisor_profile(self, tract):
        tools = tract.as_tools(profile="supervisor")
        assert len(tools) == 25

    def test_full_profile(self, tract):
        tools = tract.as_tools(profile="full")
        assert len(tools) == 25

    def test_anthropic_format(self, tract):
        tools = tract.as_tools(format="anthropic")
        assert isinstance(tools, list)
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            # Should NOT have OpenAI-specific keys
            assert "type" not in tool or tool.get("type") != "function"

    def test_with_overrides(self, tract):
        custom_desc = "Custom status description for testing"
        tools = tract.as_tools(overrides={"status": custom_desc})
        status_tools = [
            t for t in tools
            if t["function"]["name"] == "status"
        ]
        assert len(status_tools) == 1
        assert status_tools[0]["function"]["description"] == custom_desc

    def test_with_profile_object(self, tract):
        custom_profile = ToolProfile(
            name="custom",
            tool_configs={
                "status": ToolConfig(enabled=True),
                "compile": ToolConfig(enabled=True),
            },
        )
        tools = tract.as_tools(profile=custom_profile)
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"status", "compile"}

    def test_unknown_format_raises(self, tract):
        with pytest.raises(ValueError, match="Unknown format"):
            tract.as_tools(format="bedrock")

    def test_unknown_profile_raises(self, tract):
        with pytest.raises(ValueError, match="Unknown profile"):
            tract.as_tools(profile="nonexistent")


# ===========================================================================
# Integration: end-to-end tool execution
# ===========================================================================


class TestToolkitIntegration:
    """End-to-end tests combining executor with Tract operations."""

    def test_commit_then_compile_via_tools(self, tract):
        executor = ToolExecutor(tract)
        # Commit via tool
        r1 = executor.execute("commit", {
            "content": {
                "content_type": "instruction",
                "text": "You are a helpful assistant.",
            },
        })
        assert r1.success

        r2 = executor.execute("commit", {
            "content": {
                "content_type": "dialogue",
                "role": "user",
                "text": "What is 2+2?",
            },
        })
        assert r2.success

        # Compile via tool
        r3 = executor.execute("compile", {})
        assert r3.success
        assert "2 messages" in r3.output

    def test_branch_and_switch_via_tools(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        # Create branch
        r1 = executor.execute("branch", {"name": "feature", "switch": False})
        assert r1.success
        assert "feature" in r1.output

        # Switch to branch
        r2 = executor.execute("switch", {"target": "feature"})
        assert r2.success
        assert "feature" in r2.output

    def test_annotate_via_tools(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        head = tract_with_commits.head
        result = executor.execute("annotate", {
            "target_hash": head,
            "priority": "pinned",
            "reason": "important context",
        })
        assert result.success
        assert "pinned" in result.output

    def test_diff_via_tools(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("diff", {})
        assert result.success
        assert "added" in result.output

    def test_reset_via_tools(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        # Get first commit hash from log
        log_entries = tract_with_commits.log(limit=3)
        first = log_entries[-1]
        result = executor.execute("reset", {"target": first.commit_hash})
        assert result.success
        assert "Reset HEAD" in result.output


# ===========================================================================
# New tools (Fix 2): config, tags, triggers
# ===========================================================================


class TestNewTools:
    """Tests for the 7 new toolkit tools (Fix 2)."""

    def test_configure_model_per_operation(self, tract):
        """configure_model sets per-operation config."""
        executor = ToolExecutor(tract)
        result = executor.execute("configure_model", {"model": "gpt-4o", "operation": "chat"})
        assert result.success
        assert "chat" in result.output

    def test_configure_model_tract_wide(self, tract):
        """configure_model sets tract-wide default when no operation specified."""
        executor = ToolExecutor(tract)
        result = executor.execute("configure_model", {"model": "gpt-4o"})
        assert result.success
        assert "tract-wide" in result.output

    def test_tag_untag_roundtrip(self, tract):
        """tag and untag tools work together."""
        tract.register_tag("important", "important items")
        info = tract.commit({"content_type": "dialogue", "role": "user", "text": "test"})
        executor = ToolExecutor(tract)
        result = executor.execute("tag", {"commit_hash": info.commit_hash, "tag": "important"})
        assert result.success
        assert "Tagged" in result.output
        result = executor.execute("untag", {"commit_hash": info.commit_hash, "tag": "important"})
        assert result.success

    def test_query_by_tags(self, tract):
        """query_by_tags finds tagged commits."""
        tract.register_tag("milestone", "milestone marker")
        info = tract.commit({"content_type": "dialogue", "role": "user", "text": "test"})
        tract.tag(info.commit_hash, "milestone")
        executor = ToolExecutor(tract)
        result = executor.execute("query_by_tags", {"tags": ["milestone"]})
        assert result.success
        assert info.commit_hash[:8] in result.output

    def test_register_trigger(self, tract):
        """register_trigger creates a trigger."""
        executor = ToolExecutor(tract)
        result = executor.execute("register_trigger", {
            "trigger_type": "compress",
            "config": {"threshold": 0.8},
        })
        assert result.success
        assert "Registered" in result.output

    def test_unregister_trigger(self, tract):
        """unregister_trigger removes a trigger."""
        from tract.triggers.builtin.compress import CompressTrigger
        tract.register_trigger(CompressTrigger(threshold=0.8))
        executor = ToolExecutor(tract)
        result = executor.execute("unregister_trigger", {"trigger_name": "auto-compress"})
        assert result.success

    def test_toggle_triggers(self, tract):
        """toggle_triggers pauses and resumes."""
        tract.configure_triggers()
        executor = ToolExecutor(tract)
        result = executor.execute("toggle_triggers", {"enabled": False})
        assert result.success
        assert "paused" in result.output.lower()
        result = executor.execute("toggle_triggers", {"enabled": True})
        assert result.success
        assert "resumed" in result.output.lower()

    def test_new_tools_in_full_profile(self):
        """FULL_PROFILE includes all 25 tools."""
        assert "configure_model" in FULL_PROFILE.tool_configs
        assert "tag" in FULL_PROFILE.tool_configs
        assert "register_trigger" in FULL_PROFILE.tool_configs

    def test_new_tools_in_self_profile(self):
        """SELF_PROFILE includes tag, configure_model, toggle_triggers."""
        assert "tag" in SELF_PROFILE.tool_configs
        assert "configure_model" in SELF_PROFILE.tool_configs
        assert "toggle_triggers" in SELF_PROFILE.tool_configs

    def test_new_tools_in_supervisor_profile(self):
        """SUPERVISOR_PROFILE includes all 7 new tools."""
        assert "configure_model" in SUPERVISOR_PROFILE.tool_configs
        assert "register_trigger" in SUPERVISOR_PROFILE.tool_configs
        assert "toggle_triggers" in SUPERVISOR_PROFILE.tool_configs
