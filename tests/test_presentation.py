"""Tests for Layer 2 presentation and agent-friendly error handling."""

from __future__ import annotations

import re
import time

import pytest

from tract import Tract, InstructionContent, DialogueContent
from tract.toolkit.models import ToolResult
from tract.toolkit.executor import ToolExecutor
from tract.exceptions import (
    TraceError,
    BranchNotFoundError,
    CommitNotFoundError,
    BudgetExceededError,
    MergeConflictError,
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
# Presentation tests
# ---------------------------------------------------------------------------


class TestPresentSuccess:
    def test_present_success_short_output(self, tract):
        """Output under threshold passes through with metadata footer."""
        from tract.toolkit.presentation import ToolPresenter

        presenter = ToolPresenter(tract)
        result = presenter.present_success("hello world", "status", 12.5)
        # The raw content should be preserved
        assert "hello world" in result
        # Should have a metadata footer
        assert "[ok" in result.lower() or "ok" in result.lower()

    def test_present_success_overflow(self, tract):
        """Output over max_output_lines gets truncated with hint."""
        from tract.toolkit.presentation import ToolPresenter, PresentationConfig

        config = PresentationConfig(max_output_lines=5)
        presenter = ToolPresenter(tract, config)
        long_output = "\n".join(f"line {i}" for i in range(100))
        result = presenter.present_success(long_output, "log", 50.0)
        assert "truncated" in result

    def test_present_success_overflow_bytes(self, tract):
        """Output over max_output_bytes gets truncated marker."""
        from tract.toolkit.presentation import ToolPresenter, PresentationConfig

        config = PresentationConfig(max_output_bytes=50)
        presenter = ToolPresenter(tract, config)
        long_output = "x" * 200
        result = presenter.present_success(long_output, "compile", 10.0)
        assert "truncated" in result
        # Truncation hint should be present
        assert "Tip:" in result or "narrow" in result

    def test_present_error_with_hint(self, tract):
        """Error from TraceError with hint includes both error and hint."""
        from tract.toolkit.presentation import ToolPresenter

        presenter = ToolPresenter(tract)
        exc = BranchNotFoundError("feature")
        result = presenter.present_error(str(exc), "switch", 1.0, exception=exc)
        assert "Branch not found" in result or "feature" in result
        # hint should be surfaced
        assert "hint" in result.lower() or "list_branches" in result

    def test_present_error_without_hint(self, tract):
        """Error from non-TraceError has no hint line."""
        from tract.toolkit.presentation import ToolPresenter

        presenter = ToolPresenter(tract)
        exc = ValueError("bad input")
        result = presenter.present_error(str(exc), "commit", 2.0, exception=exc)
        assert "bad input" in result

    def test_metadata_footer_format(self, tract):
        """Footer matches expected pattern with ok/ms/branch info."""
        from tract.toolkit.presentation import ToolPresenter

        presenter = ToolPresenter(tract)
        result = presenter.present_success("test output", "status", 42.0)
        # Should contain timing info
        assert "42" in result or "ms" in result.lower()

    def test_executor_returns_raw_output(self, tract_with_commits):
        """Executor returns raw output without presentation footer."""
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("status", {})
        assert result.success
        assert result.duration_ms > 0
        assert result.output  # non-empty
        # Executor is Layer 1 only — no metadata footer
        assert "[ok" not in result.output

    def test_presenter_adds_footer_to_result(self, tract_with_commits):
        """ToolPresenter.present_result adds metadata footer to executor output."""
        from tract.toolkit.presentation import ToolPresenter

        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("status", {})
        assert result.success

        presenter = ToolPresenter(tract_with_commits)
        presented = presenter.present_result(result)
        assert "[ok" in presented

    def test_present_result_convenience_success(self, tract_with_commits):
        """present_result formats a successful ToolResult."""
        from tract.toolkit.presentation import ToolPresenter

        result = ToolResult(
            tool_name="status", success=True,
            output="Branch: main\nCommits: 3", duration_ms=5.0,
        )
        presenter = ToolPresenter(tract_with_commits)
        presented = presenter.present_result(result)
        assert "Branch: main" in presented
        assert "[ok" in presented

    def test_present_result_convenience_error(self, tract_with_commits):
        """present_result formats a failed ToolResult with hint."""
        from tract.toolkit.presentation import ToolPresenter

        result = ToolResult(
            tool_name="switch", success=False,
            error="BranchNotFoundError: feature",
            hint="Use list_branches() to see available branches",
            duration_ms=2.0,
        )
        presenter = ToolPresenter(tract_with_commits)
        presented = presenter.present_result(result)
        assert "[error] switch:" in presented
        assert "[hint]" in presented
        assert "list_branches" in presented


# ---------------------------------------------------------------------------
# Exception hint tests
# ---------------------------------------------------------------------------


class TestExceptionHints:
    def test_branch_not_found_hint(self):
        exc = BranchNotFoundError("feature")
        assert exc.hint
        assert "list_branches" in exc.hint

    def test_commit_not_found_hint(self):
        exc = CommitNotFoundError("abc123")
        assert exc.hint
        assert "log" in exc.hint.lower()

    def test_budget_exceeded_hint(self):
        exc = BudgetExceededError(5000, 4000)
        assert exc.hint
        assert "compress" in exc.hint.lower()

    def test_merge_conflict_hint(self):
        exc = MergeConflictError(3)
        assert exc.hint
        assert "strategy" in exc.hint.lower() or "merge" in exc.hint.lower()

    def test_trace_error_base_hint(self):
        exc = TraceError("something went wrong")
        assert exc.hint == ""


# ---------------------------------------------------------------------------
# ToolResult enhancement tests
# ---------------------------------------------------------------------------


class TestToolResultStr:
    def test_tool_result_str_success(self):
        result = ToolResult(tool_name="status", success=True, output="all good")
        assert str(result) == "all good"

    def test_tool_result_str_error_with_hint(self):
        result = ToolResult(
            tool_name="switch",
            success=False,
            error="Branch not found",
            hint="Use t.list_branches()",
        )
        s = str(result)
        assert "[error] switch: Branch not found" in s
        assert "[hint] Use t.list_branches()" in s

    def test_tool_result_str_error_no_hint(self):
        result = ToolResult(
            tool_name="commit",
            success=False,
            error="Validation failed",
        )
        s = str(result)
        assert "[error] commit: Validation failed" in s
        assert "[hint]" not in s

    def test_tool_result_new_fields_defaults(self):
        result = ToolResult(tool_name="test", success=True)
        assert result.hint == ""
        assert result.duration_ms == 0
        assert result.truncated is False


# ---------------------------------------------------------------------------
# Executor integration tests
# ---------------------------------------------------------------------------


class TestExecutorIntegration:
    def test_executor_unknown_tool_lists_available(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("nonexistent_tool", {})
        assert not result.success
        assert "Unknown tool" in result.error
        assert result.hint
        assert "Available tools:" in result.hint
        # Should list at least some known tools
        assert "commit" in result.hint or "status" in result.hint

    def test_executor_timing(self, tract_with_commits):
        executor = ToolExecutor(tract_with_commits)
        result = executor.execute("status", {})
        assert result.success
        assert result.duration_ms > 0

    def test_executor_error_extracts_hint(self, tract):
        """When tool raises TraceError, hint is in result."""
        executor = ToolExecutor(tract)
        # Try to switch to a non-existent branch
        result = executor.execute("switch", {"target": "nonexistent_branch"})
        assert not result.success
        assert result.hint  # Should have extracted hint from BranchNotFoundError
        assert "list_branches" in result.hint


# ---------------------------------------------------------------------------
# Discovery profile tests
# ---------------------------------------------------------------------------


class TestDiscoveryProfile:
    def test_discovery_profile_registered(self):
        """'discovery' profile exists in get_profile()."""
        from tract.toolkit.profiles import get_profile

        profile = get_profile("discovery")
        assert profile.name == "discovery"

    def test_discovery_tools_count(self, tract):
        """get_discovery_tools returns 3 tools."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        assert len(tools) == 3

    def test_discovery_help_no_args(self, tract):
        """tract_help with no topic returns domain overview."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        help_tool = next(t for t in tools if t.name == "tract_help")
        result = help_tool.handler()
        assert isinstance(result, str)
        # Should mention domains
        assert "context" in result.lower() or "branch" in result.lower()

    def test_discovery_help_domain(self, tract):
        """tract_help with domain returns actions."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        help_tool = next(t for t in tools if t.name == "tract_help")
        result = help_tool.handler(topic="context")
        assert isinstance(result, str)
        assert "commit" in result.lower()

    def test_discovery_help_action(self, tract):
        """tract_help with action name returns schema."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        help_tool = next(t for t in tools if t.name == "tract_help")
        result = help_tool.handler(topic="commit")
        assert isinstance(result, str)
        # Should include parameter info
        assert "param" in result.lower() or "content" in result.lower()

    def test_discovery_do_executes(self, tract_with_commits):
        """tract_do with valid action executes it."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract_with_commits)
        do_tool = next(t for t in tools if t.name == "tract_do")
        result = do_tool.handler(action="status")
        assert isinstance(result, str)
        assert "main" in result.lower() or "branch" in result.lower()

    def test_discovery_do_unknown_action(self, tract):
        """tract_do with bad action lists available."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        do_tool = next(t for t in tools if t.name == "tract_do")
        result = do_tool.handler(action="nonexistent_action")
        assert isinstance(result, str)
        assert "unknown" in result.lower() or "available" in result.lower() or "error" in result.lower()

    def test_discovery_inspect_dashboard(self, tract_with_commits):
        """tract_inspect with no args returns dashboard."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract_with_commits)
        inspect_tool = next(t for t in tools if t.name == "tract_inspect")
        result = inspect_tool.handler()
        assert isinstance(result, str)
        # Dashboard should show branch/status info
        assert "main" in result.lower() or "branch" in result.lower() or "commit" in result.lower()

    def test_discovery_inspect_branches(self, tract_with_commits):
        """tract_inspect what=branches returns branch info including main."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract_with_commits)
        inspect_tool = next(t for t in tools if t.name == "tract_inspect")
        result = inspect_tool.handler(what="branches")
        assert isinstance(result, str)
        assert "main" in result.lower()

    def test_discovery_inspect_log(self, tract_with_commits):
        """tract_inspect what=history returns commit history."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract_with_commits)
        inspect_tool = next(t for t in tools if t.name == "tract_inspect")
        result = inspect_tool.handler(what="history")
        assert isinstance(result, str)
        assert "commit" in result.lower() or "log" in result.lower()

    def test_discovery_inspect_config(self, tract):
        """tract_inspect what=config returns config info or empty message."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        inspect_tool = next(t for t in tools if t.name == "tract_inspect")
        result = inspect_tool.handler(what="config")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_discovery_inspect_tags(self, tract):
        """tract_inspect what=tags returns tag info or empty message."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        inspect_tool = next(t for t in tools if t.name == "tract_inspect")
        result = inspect_tool.handler(what="tags")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_discovery_inspect_directives(self, tract_with_commits):
        """tract_inspect what=directives returns directive info."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract_with_commits)
        inspect_tool = next(t for t in tools if t.name == "tract_inspect")
        result = inspect_tool.handler(what="directives")
        assert isinstance(result, str)
        # tract_with_commits has an InstructionContent commit
        assert "directive" in result.lower() or "system" in result.lower()

    def test_discovery_inspect_unknown(self, tract):
        """tract_inspect what=badthing lists available options (error-as-navigation)."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        inspect_tool = next(t for t in tools if t.name == "tract_inspect")
        result = inspect_tool.handler(what="badthing")
        assert isinstance(result, str)
        assert "unknown" in result.lower()
        assert "branches" in result.lower()
        assert "history" in result.lower()

    def test_discovery_help_unknown_topic(self, tract):
        """tract_help with unknown topic lists available domains/actions."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        help_tool = next(t for t in tools if t.name == "tract_help")
        result = help_tool.handler(topic="nonexistent")
        assert isinstance(result, str)
        assert "unknown" in result.lower()
        assert "available" in result.lower()

    def test_discovery_do_param_error(self, tract):
        """tract_do with action=commit and no params shows schema, not crash."""
        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(tract)
        do_tool = next(t for t in tools if t.name == "tract_do")
        result = do_tool.handler(action="commit")
        assert isinstance(result, str)
        # Should show parameter error and schema, not an unhandled exception
        assert "param" in result.lower() or "error" in result.lower()
