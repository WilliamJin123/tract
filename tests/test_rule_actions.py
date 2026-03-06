"""Tests for rule engine action handlers in src/tract/rules/actions.py.

Covers: SetConfigAction, BlockAction, RequireAction, OperationAction,
CompileFilterAction, LLMAction, CreateRuleAction, ACTION_SEMANTICS,
ACTION_CATEGORIES registries.
"""

from __future__ import annotations

import pytest

from tract import Tract
from tract.rules.actions import (
    ACTION_CATEGORIES,
    ACTION_SEMANTICS,
    BUILTIN_ACTIONS,
    BlockAction,
    CompileFilterAction,
    CreateRuleAction,
    LLMAction,
    OperationAction,
    RequireAction,
    SetConfigAction,
)
from tract.rules.models import ActionResult, EvalContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(t: Tract, commit=None, event: str = "test") -> EvalContext:
    return EvalContext(
        event=event,
        commit=commit,
        branch=t.current_branch or "",
        head=t.head or "",
        tract=t,
        metrics={"total_tokens": 0},
        rule_index=t.rule_index,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def t():
    """In-memory tract with a few commits for operations to work with."""
    tract = Tract.open()
    tract.user("first message")
    tract.user("second message")
    tract.assistant("response")
    yield tract
    tract.close()


# ===========================================================================
# SetConfigAction
# ===========================================================================


class TestSetConfigAction:
    def test_set_config_action(self, t: Tract):
        """SetConfigAction returns key/value in data with success=True."""
        action = SetConfigAction()
        ctx = _make_ctx(t)
        result = action.execute({"key": "temperature", "value": 0.7}, ctx)
        assert isinstance(result, ActionResult)
        assert result.action_type == "set_config"
        assert result.success is True
        assert result.data["key"] == "temperature"
        assert result.data["value"] == 0.7


# ===========================================================================
# BlockAction
# ===========================================================================


class TestBlockAction:
    def test_block_action(self, t: Tract):
        """BlockAction returns blocked=True with default reason."""
        action = BlockAction()
        ctx = _make_ctx(t)
        result = action.execute({}, ctx)
        assert result.action_type == "block"
        assert result.success is True
        assert result.data["blocked"] is True
        assert result.reason == "Blocked by rule"

    def test_block_action_with_reason(self, t: Tract):
        """BlockAction uses a custom reason when provided."""
        action = BlockAction()
        ctx = _make_ctx(t)
        result = action.execute({"reason": "Token limit exceeded"}, ctx)
        assert result.action_type == "block"
        assert result.success is True
        assert result.data["blocked"] is True
        assert result.reason == "Token limit exceeded"


# ===========================================================================
# RequireAction
# ===========================================================================


class TestRequireAction:
    def test_require_action_met(self, t: Tract):
        """RequireAction with condition=None (always True) succeeds."""
        action = RequireAction()
        ctx = _make_ctx(t)
        result = action.execute({"condition": None}, ctx)
        assert result.action_type == "require"
        assert result.success is True
        assert result.data["met"] is True

    def test_require_action_not_met(self, t: Tract):
        """RequireAction fails when the embedded condition is not met."""
        action = RequireAction()
        # Use a tag condition that won't match (no tags on commits)
        ctx = _make_ctx(t, commit=None)
        result = action.execute(
            {"condition": {"type": "tag", "tag": "nonexistent", "present": True}},
            ctx,
        )
        assert result.action_type == "require"
        assert result.success is False
        assert result.data["met"] is False
        assert "not met" in result.reason.lower()


# ===========================================================================
# OperationAction
# ===========================================================================


class TestOperationAction:
    def test_operation_action_compress(self, t: Tract):
        """OperationAction dispatches compress with content= param."""
        action = OperationAction()
        ctx = _make_ctx(t)
        result = action.execute(
            {"op": "compress", "params": {"content": "summary of conversation"}},
            ctx,
        )
        assert result.action_type == "operation"
        assert result.success is True
        assert result.data["op"] == "compress"

    def test_operation_action_branch(self, t: Tract):
        """OperationAction dispatches branch creation."""
        action = OperationAction()
        ctx = _make_ctx(t)
        result = action.execute(
            {"op": "branch", "params": {"name": "feature"}},
            ctx,
        )
        assert result.action_type == "operation"
        assert result.success is True
        assert result.data["op"] == "branch"
        assert result.data["result"] == "feature"
        # Verify branch was actually created
        branch_names = [b.name for b in t.list_branches()]
        assert "feature" in branch_names

    def test_operation_action_unknown(self, t: Tract):
        """OperationAction raises ValueError for unsupported operations."""
        action = OperationAction()
        ctx = _make_ctx(t)
        with pytest.raises(ValueError, match="Unknown operation"):
            action.execute({"op": "nonexistent_op"}, ctx)


# ===========================================================================
# CompileFilterAction
# ===========================================================================


class TestCompileFilterAction:
    def test_compile_filter_action(self, t: Tract):
        """CompileFilterAction stores mode params in data."""
        action = CompileFilterAction()
        ctx = _make_ctx(t)
        params = {"mode": "recent", "limit": 10, "exclude_system": True}
        result = action.execute(params, ctx)
        assert result.action_type == "compile_filter"
        assert result.success is True
        assert result.data["mode"] == "recent"
        assert result.data["limit"] == 10
        assert result.data["exclude_system"] is True


# ===========================================================================
# LLMAction
# ===========================================================================


class TestLLMAction:
    def test_llm_action_placeholder(self, t: Tract):
        """LLMAction returns deferred note (placeholder for R4)."""
        action = LLMAction()
        ctx = _make_ctx(t)
        result = action.execute({"instruction": "Summarize the conversation"}, ctx)
        assert result.action_type == "llm"
        assert result.success is True
        assert result.data["instruction"] == "Summarize the conversation"
        assert "deferred" in result.data["note"].lower()


# ===========================================================================
# CreateRuleAction
# ===========================================================================


class TestCreateRuleAction:
    def test_create_rule_action_valid(self, t: Tract):
        """CreateRuleAction returns template with deferred=True for valid input."""
        action = CreateRuleAction()
        ctx = _make_ctx(t)
        template = {
            "name": "auto_pin",
            "trigger": "commit",
            "action": {"type": "annotate", "priority": "PINNED"},
        }
        result = action.execute({"template": template}, ctx)
        assert result.action_type == "create_rule"
        assert result.success is True
        assert result.data["template"] == template
        assert result.data["deferred"] is True

    def test_create_rule_action_invalid(self, t: Tract):
        """CreateRuleAction fails when template is missing required fields."""
        action = CreateRuleAction()
        ctx = _make_ctx(t)
        # Missing 'action' field
        incomplete_template = {"name": "broken_rule", "trigger": "commit"}
        result = action.execute({"template": incomplete_template}, ctx)
        assert result.action_type == "create_rule"
        assert result.success is False
        assert "missing required fields" in result.reason.lower()


# ===========================================================================
# Registry / Semantics / Categories
# ===========================================================================


class TestRegistries:
    def test_builtin_actions_contains_all_types(self):
        """BUILTIN_ACTIONS has entries for all seven action types."""
        expected = {"set_config", "operation", "block", "require", "compile_filter", "llm", "create_rule"}
        assert set(BUILTIN_ACTIONS.keys()) == expected

    def test_action_semantics(self):
        """ACTION_SEMANTICS maps each action to 'override' or 'accumulate'."""
        assert ACTION_SEMANTICS["set_config"] == "override"
        assert ACTION_SEMANTICS["compile_filter"] == "override"
        assert ACTION_SEMANTICS["block"] == "accumulate"
        assert ACTION_SEMANTICS["require"] == "accumulate"
        assert ACTION_SEMANTICS["llm"] == "accumulate"
        assert ACTION_SEMANTICS["operation"] == "accumulate"
        assert ACTION_SEMANTICS["create_rule"] == "accumulate"
        # All values are either 'override' or 'accumulate'
        for v in ACTION_SEMANTICS.values():
            assert v in ("override", "accumulate")

    def test_action_categories(self):
        """ACTION_CATEGORIES maps each action to its category."""
        assert ACTION_CATEGORIES["require"] == "gate"
        assert ACTION_CATEGORIES["block"] == "gate"
        assert ACTION_CATEGORIES["llm"] == "work"
        assert ACTION_CATEGORIES["operation"] == "work"
        assert ACTION_CATEGORIES["compile_filter"] == "handoff"
        assert ACTION_CATEGORIES["set_config"] == "work"
        assert ACTION_CATEGORIES["create_rule"] == "post"
        # All values are in the expected set
        valid_categories = {"gate", "work", "handoff", "post"}
        for v in ACTION_CATEGORIES.values():
            assert v in valid_categories
