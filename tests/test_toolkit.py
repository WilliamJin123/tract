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

    def test_returns_29_definitions(self, tract):
        tools = get_all_tools(tract)
        assert len(tools) == 29

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
            "configure", "create_metadata", "get_config", "transition",
            "directive", "create_middleware", "remove_middleware",
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
            "configure_model",
            "configure", "create_metadata", "get_config", "transition",
            "directive", "create_middleware", "remove_middleware",
        }
        assert self_names == expected
        assert len(self_tools) == 23

    def test_supervisor_profile_all_tools(self, tract):
        tools = get_all_tools(tract)
        sup_tools = SUPERVISOR_PROFILE.filter_tools(tools)
        assert len(sup_tools) == 29

    def test_full_profile_all_tools(self, tract):
        tools = get_all_tools(tract)
        full_tools = FULL_PROFILE.filter_tools(tools)
        assert len(full_tools) == 29

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
        assert len(names) == 29
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

    def test_execute_compress_no_llm_fails(self, tract_with_commits):
        """Compress without LLM or content fails gracefully."""
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("compress", {})
        assert result.success is False

    def test_execute_compress_with_content(self, tract_with_commits):
        """Compress with manual content succeeds via tool."""
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("compress", {"content": "Summary of the chat."})
        assert result.success is True
        assert "Compressed" in result.output

    def test_execute_compress_with_content_and_preserve(self):
        """Compress with content + preserve keeps pinned commits."""
        from tract import Tract, DialogueContent
        t = Tract.open()
        t.commit(DialogueContent(role="user", text="First"))
        t.commit(DialogueContent(role="user", text="Second"))
        t.commit(DialogueContent(role="user", text="Third"))
        entries = list(t.log(limit=10))
        # entries: [Third, Second, First] -- preserve the middle
        middle_hash = entries[1].commit_hash
        executor = ToolExecutor(t)
        result = executor.execute("compress", {
            "content": "Summary of first and third",
            "preserve": [middle_hash],
        })
        assert result.success is True
        msgs = t.compile().to_dicts()
        # summary + preserved middle
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Summary of first and third"
        assert msgs[1]["content"] == "Second"


# ===========================================================================
# Tract.as_tools() facade tests
# ===========================================================================


class TestAsTools:
    """Test the Tract.as_tools() convenience method."""

    def test_default_profile_openai(self, tract):
        tools = tract.runtime.tools.as_tools()
        assert isinstance(tools, list)
        # Default profile is "compact" with 7 domain-grouped tools
        assert len(tools) == 7
        # OpenAI format
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool

    def test_supervisor_profile(self, tract):
        tools = tract.runtime.tools.as_tools(profile="supervisor")
        assert len(tools) == 29

    def test_full_profile(self, tract):
        tools = tract.runtime.tools.as_tools(profile="full")
        assert len(tools) == 29

    def test_anthropic_format(self, tract):
        tools = tract.runtime.tools.as_tools(format="anthropic")
        assert isinstance(tools, list)
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            # Should NOT have OpenAI-specific keys
            assert "type" not in tool or tool.get("type") != "function"

    def test_with_overrides(self, tract):
        custom_desc = "Custom status description for testing"
        tools = tract.runtime.tools.as_tools(profile="self", overrides={"status": custom_desc})
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
        tools = tract.runtime.tools.as_tools(profile=custom_profile)
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"status", "compile"}

    def test_unknown_format_raises(self, tract):
        with pytest.raises(ValueError, match="Unknown format"):
            tract.runtime.tools.as_tools(format="bedrock")

    def test_unknown_profile_raises(self, tract):
        with pytest.raises(ValueError, match="Unknown profile"):
            tract.runtime.tools.as_tools(profile="nonexistent")


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
# New tools (Fix 2): config, tags
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

    def test_configure_model_merges_tract_default(self, tract):
        """configure_model merges with existing default instead of replacing."""
        from tract.models.config import LLMConfig
        executor = ToolExecutor(tract)

        # Set model first
        executor.execute("configure_model", {"model": "gpt-4o"})
        assert tract.default_config.model == "gpt-4o"

        # Set temperature -- model should be preserved
        executor.execute("configure_model", {"temperature": 0.8})
        assert tract.default_config.temperature == 0.8
        assert tract.default_config.model == "gpt-4o", (
            "model should survive a temperature-only configure_model call"
        )

        # Override model again -- temperature should be preserved
        executor.execute("configure_model", {"model": "gpt-3.5-turbo"})
        assert tract.default_config.model == "gpt-3.5-turbo"
        assert tract.default_config.temperature == 0.8, (
            "temperature should survive a model-only configure_model call"
        )

    def test_configure_model_merges_operation_config(self, tract):
        """configure_model merges per-operation configs."""
        executor = ToolExecutor(tract)

        executor.execute("configure_model", {"model": "gpt-4o", "operation": "chat"})
        chat_cfg = tract.operation_configs.chat
        assert chat_cfg.model == "gpt-4o"

        # Add temperature to chat -- model preserved
        executor.execute("configure_model", {"temperature": 0.5, "operation": "chat"})
        chat_cfg = tract.operation_configs.chat
        assert chat_cfg.temperature == 0.5
        assert chat_cfg.model == "gpt-4o"

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

    def test_new_tools_in_full_profile(self):
        """FULL_PROFILE includes all 22 tools."""
        assert "configure_model" in FULL_PROFILE.tool_configs
        assert "tag" in FULL_PROFILE.tool_configs

    def test_new_tools_in_self_profile(self):
        """SELF_PROFILE includes tag, configure_model."""
        assert "tag" in SELF_PROFILE.tool_configs
        assert "configure_model" in SELF_PROFILE.tool_configs

    def test_new_tools_in_supervisor_profile(self):
        """SUPERVISOR_PROFILE includes configure_model and tag tools."""
        assert "configure_model" in SUPERVISOR_PROFILE.tool_configs
        assert "tag" in SUPERVISOR_PROFILE.tool_configs


# ---------------------------------------------------------------------------
# as_callable_tools() tests
# ---------------------------------------------------------------------------


class TestAsCallableTools:
    """Tests for Tract.as_callable_tools() framework integration."""

    def test_returns_callables(self, tract):
        """as_callable_tools() returns a list of callables."""
        tools = tract.runtime.tools.as_callable_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        for tool in tools:
            assert callable(tool)

    def test_callable_has_name(self, tract):
        """Each callable has __name__ matching the tool name."""
        tools = tract.runtime.tools.as_callable_tools(profile="self")
        names = {t.__name__ for t in tools}
        assert "status" in names
        assert "compile" in names
        assert "log" in names

    def test_callable_has_docstring(self, tract):
        """Each callable has __doc__ from the tool description."""
        tools = tract.runtime.tools.as_callable_tools(profile="self")
        status_tool = next(t for t in tools if t.__name__ == "status")
        assert status_tool.__doc__ is not None
        assert len(status_tool.__doc__) > 10

    def test_callable_has_signature(self, tract):
        """Each callable has a proper inspect.Signature with typed params."""
        import inspect

        tools = tract.runtime.tools.as_callable_tools(profile="self")
        annotate_tool = next(t for t in tools if t.__name__ == "annotate")

        sig = inspect.signature(annotate_tool)
        assert "target_hash" in sig.parameters
        assert "priority" in sig.parameters

        # Check type annotations
        assert sig.parameters["target_hash"].annotation is str
        assert sig.parameters["priority"].annotation is str

    def test_callable_has_return_annotation(self, tract):
        """Each callable's signature has str return annotation."""
        import inspect

        tools = tract.runtime.tools.as_callable_tools(profile="self")
        status_tool = next(t for t in tools if t.__name__ == "status")
        sig = inspect.signature(status_tool)
        assert sig.return_annotation is str

    def test_callable_executes(self, tract_with_commits):
        """Calling a callable actually executes the tool handler."""
        tools = tract_with_commits.runtime.tools.as_callable_tools(profile="self")
        status_tool = next(t for t in tools if t.__name__ == "status")
        result = status_tool()
        assert isinstance(result, str)
        assert "main" in result  # branch name in status output

    def test_callable_with_args(self, tract_with_commits):
        """Callables with parameters accept kwargs correctly."""
        tools = tract_with_commits.runtime.tools.as_callable_tools(profile="self")
        log_tool = next(t for t in tools if t.__name__ == "log")
        result = log_tool(limit=2)
        assert isinstance(result, str)

    def test_profile_filtering(self, tract):
        """as_callable_tools() respects profile filtering."""
        self_tools = tract.runtime.tools.as_callable_tools(profile="self")
        full_tools = tract.runtime.tools.as_callable_tools(profile="full")
        self_names = {t.__name__ for t in self_tools}
        full_names = {t.__name__ for t in full_tools}
        # Full profile has more tools than self
        assert len(full_names) >= len(self_names)

    def test_description_overrides(self, tract):
        """as_callable_tools() applies description overrides."""
        tools = tract.runtime.tools.as_callable_tools(profile="self", overrides={"status": "Custom description"})
        status_tool = next(t for t in tools if t.__name__ == "status")
        assert status_tool.__doc__ == "Custom description"

    def test_optional_params_have_defaults(self, tract):
        """Optional JSON Schema params become kwargs with defaults."""
        import inspect

        tools = tract.runtime.tools.as_callable_tools(profile="self")
        log_tool = next(t for t in tools if t.__name__ == "log")
        sig = inspect.signature(log_tool)
        # 'limit' is optional in the log tool
        if "limit" in sig.parameters:
            assert sig.parameters["limit"].default is not inspect.Parameter.empty
