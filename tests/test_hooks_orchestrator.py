"""Tests for orchestrator hook handler templates.

Tests the pre-built hook handlers in tract.hooks.templates.orchestrator,
verifying they correctly approve/reject Pending objects and that the
factory functions produce correctly parameterized handlers.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from tract import Tract, InstructionContent, DialogueContent
from tract.hooks.pending import Pending
from tract.hooks.templates.orchestrator import (
    auto_approve,
    auto_approve_tool_call,
    cli_prompt,
    log_and_approve,
    log_and_approve_tool_call,
    make_log_handler,
    make_reject_handler,
    reject_all,
    reject_all_tool_call,
)
from tract.orchestrator.models import (
    ToolCall,
    ToolCallDecision,
    ToolCallReview,
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
def mock_pending(tract):
    """A minimal Pending object with a mock execute function."""
    execute_called = []

    def mock_execute(pending):
        execute_called.append(True)
        return "executed"

    p = Pending(
        operation="compress",
        tract=tract,
        _execute_fn=mock_execute,
    )
    p._execute_called = execute_called
    return p


# ---------------------------------------------------------------------------
# Hook handler tests (Pending-based)
# ---------------------------------------------------------------------------


class TestAutoApprove:
    def test_auto_approve_approves(self, mock_pending):
        """auto_approve calls pending.approve()."""
        auto_approve(mock_pending)
        assert mock_pending.status == "approved"
        assert len(mock_pending._execute_called) == 1

    def test_auto_approve_on_different_operation(self, tract):
        """auto_approve works with any Pending subclass."""
        execute_called = []

        p = Pending(
            operation="gc",
            tract=tract,
            _execute_fn=lambda pending: execute_called.append(True) or "gc_done",
        )
        auto_approve(p)
        assert p.status == "approved"
        assert len(execute_called) == 1


class TestLogAndApprove:
    def test_log_and_approve_approves(self, mock_pending):
        """log_and_approve logs and then approves."""
        log_and_approve(mock_pending)
        assert mock_pending.status == "approved"

    def test_log_and_approve_with_logger(self, mock_pending, caplog):
        """log_and_approve uses provided logger."""
        test_logger = logging.getLogger("test.hooks")
        with caplog.at_level(logging.INFO, logger="test.hooks"):
            log_and_approve(mock_pending, logger=test_logger)
        assert mock_pending.status == "approved"
        assert "compress" in caplog.text
        assert mock_pending.pending_id in caplog.text

    def test_log_and_approve_default_logger(self, mock_pending, caplog):
        """log_and_approve falls back to module logger."""
        with caplog.at_level(logging.INFO, logger="tract.hooks.templates.orchestrator"):
            log_and_approve(mock_pending)
        assert mock_pending.status == "approved"
        assert "compress" in caplog.text


class TestRejectAll:
    def test_reject_all_rejects(self, mock_pending):
        """reject_all calls pending.reject()."""
        reject_all(mock_pending)
        assert mock_pending.status == "rejected"
        assert mock_pending.rejection_reason == "Rejected by policy"

    def test_reject_all_custom_reason(self, mock_pending):
        """reject_all accepts custom reason."""
        reject_all(mock_pending, reason="Custom reason")
        assert mock_pending.status == "rejected"
        assert mock_pending.rejection_reason == "Custom reason"


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestMakeLogHandler:
    def test_make_log_handler_creates_handler(self, mock_pending, caplog):
        """make_log_handler returns a handler that logs and approves."""
        test_logger = logging.getLogger("test.factory")
        handler = make_log_handler(test_logger)

        with caplog.at_level(logging.INFO, logger="test.factory"):
            handler(mock_pending)

        assert mock_pending.status == "approved"
        assert "compress" in caplog.text

    def test_make_log_handler_callable(self):
        """make_log_handler returns a callable."""
        handler = make_log_handler(logging.getLogger("test"))
        assert callable(handler)


class TestMakeRejectHandler:
    def test_make_reject_handler_default_reason(self, mock_pending):
        """make_reject_handler with default reason."""
        handler = make_reject_handler()
        handler(mock_pending)
        assert mock_pending.status == "rejected"
        assert mock_pending.rejection_reason == "Rejected by policy"

    def test_make_reject_handler_custom_reason(self, mock_pending):
        """make_reject_handler with custom reason."""
        handler = make_reject_handler("Budget exceeded")
        handler(mock_pending)
        assert mock_pending.status == "rejected"
        assert mock_pending.rejection_reason == "Budget exceeded"

    def test_make_reject_handler_callable(self):
        """make_reject_handler returns a callable."""
        handler = make_reject_handler()
        assert callable(handler)


# ---------------------------------------------------------------------------
# Tool-call review callback tests
# ---------------------------------------------------------------------------


class TestAutoApproveToolCall:
    def test_auto_approve_tool_call(self):
        """auto_approve_tool_call returns approved decision."""
        tc = ToolCall(id="tc-1", name="status", arguments={})
        result = auto_approve_tool_call(tc)
        assert isinstance(result, ToolCallReview)
        assert result.decision == ToolCallDecision.APPROVED

    def test_auto_approve_tool_call_no_side_effects(self):
        """auto_approve_tool_call doesn't modify the tool call."""
        tc = ToolCall(id="tc-1", name="compress", arguments={"threshold": 0.8})
        _ = auto_approve_tool_call(tc)
        assert tc.name == "compress"
        assert tc.arguments == {"threshold": 0.8}


class TestLogAndApproveToolCall:
    def test_log_and_approve_tool_call(self, caplog):
        """log_and_approve_tool_call logs and approves."""
        tc = ToolCall(id="tc-1", name="compress", arguments={})
        with caplog.at_level(logging.INFO, logger="tract.hooks.templates.orchestrator"):
            result = log_and_approve_tool_call(tc)
        assert result.decision == ToolCallDecision.APPROVED
        assert "compress" in caplog.text


class TestRejectAllToolCall:
    def test_reject_all_tool_call(self):
        """reject_all_tool_call rejects with reason."""
        tc = ToolCall(id="tc-1", name="status", arguments={})
        result = reject_all_tool_call(tc)
        assert result.decision == ToolCallDecision.REJECTED
        assert result.reason == "Auto-rejected"


# ---------------------------------------------------------------------------
# Integration: hook handlers with Tract.on()
# ---------------------------------------------------------------------------


class TestHookHandlerWithTract:
    def test_auto_approve_registered_as_hook(self, tract):
        """auto_approve can be registered with t.on() for any hookable op."""
        tract.on("compress", auto_approve)
        assert "compress" in tract.hooks
        assert tract.hooks["compress"] is auto_approve

    def test_reject_all_registered_as_hook(self, tract):
        """reject_all can be registered with t.on()."""
        tract.on("gc", lambda p: reject_all(p))
        assert "gc" in tract.hooks

    def test_factory_handler_registered_as_hook(self, tract):
        """Factory-created handler can be registered with t.on()."""
        handler = make_reject_handler("Not allowed")
        tract.on("merge", handler)
        assert "merge" in tract.hooks

    def test_catch_all_hook(self, tract):
        """Handler registered with '*' catches all operations."""
        tract.on("*", auto_approve)
        assert "*" in tract.hooks

    def test_hook_off_removes_handler(self, tract):
        """t.off() removes the handler."""
        tract.on("compress", auto_approve)
        tract.off("compress")
        assert "compress" not in tract.hooks


# ---------------------------------------------------------------------------
# Integration: orchestrator config with autonomy levels
# ---------------------------------------------------------------------------


class TestAutonomyLevelMapping:
    def test_autonomous_no_review(self):
        """AUTONOMOUS mode: no on_tool_call callback needed."""
        from tract.orchestrator.config import OrchestratorConfig, AutonomyLevel
        config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
        )
        assert config.on_tool_call is None  # No review needed

    def test_collaborative_with_approve_callback(self):
        """COLLABORATIVE mode: on_tool_call = auto_approve_tool_call."""
        from tract.orchestrator.config import OrchestratorConfig, AutonomyLevel
        config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.COLLABORATIVE,
            on_tool_call=auto_approve_tool_call,
        )
        assert config.on_tool_call is auto_approve_tool_call

    def test_collaborative_with_reject_callback(self):
        """COLLABORATIVE mode: on_tool_call = reject_all_tool_call."""
        from tract.orchestrator.config import OrchestratorConfig, AutonomyLevel
        config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.COLLABORATIVE,
            on_tool_call=reject_all_tool_call,
        )
        assert config.on_tool_call is reject_all_tool_call

    def test_manual_no_callback_needed(self):
        """MANUAL mode: all skipped regardless of callback."""
        from tract.orchestrator.config import OrchestratorConfig, AutonomyLevel
        config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.MANUAL,
        )
        assert config.on_tool_call is None  # All skipped anyway
