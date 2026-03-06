"""Tests for rule engine data models."""

import pytest
from unittest.mock import MagicMock
from tract.rules.models import RuleEntry, EvalContext, ActionResult, EvalResult


class TestRuleEntry:
    def test_creation_all_fields(self):
        e = RuleEntry(name="auto_compress", trigger="commit", condition={"type": "threshold"}, action={"type": "compress"}, commit_hash="abc123", dag_distance=3, provenance={"source": "developer"})
        assert e.name == "auto_compress"
        assert e.trigger == "commit"
        assert e.condition == {"type": "threshold"}
        assert e.action == {"type": "compress"}
        assert e.commit_hash == "abc123"
        assert e.dag_distance == 3
        assert e.provenance == {"source": "developer"}

    def test_creation_minimal(self):
        e = RuleEntry(name="r1", trigger="active", condition=None, action={"type": "set_config"}, commit_hash="abc", dag_distance=0)
        assert e.condition is None
        assert e.provenance is None

    def test_frozen(self):
        e = RuleEntry(name="r1", trigger="active", condition=None, action={}, commit_hash="abc", dag_distance=0)
        with pytest.raises(AttributeError):
            e.name = "changed"

    def test_equality(self):
        e1 = RuleEntry(name="r1", trigger="active", condition=None, action={}, commit_hash="abc", dag_distance=0)
        e2 = RuleEntry(name="r1", trigger="active", condition=None, action={}, commit_hash="abc", dag_distance=0)
        assert e1 == e2

    def test_inequality_different_distance(self):
        e1 = RuleEntry(name="r1", trigger="active", condition=None, action={}, commit_hash="abc", dag_distance=0)
        e2 = RuleEntry(name="r1", trigger="active", condition=None, action={}, commit_hash="abc", dag_distance=1)
        assert e1 != e2


class TestEvalContext:
    def test_creation(self):
        mock_tract = MagicMock()
        ctx = EvalContext(event="commit", commit=None, branch="main", head="abc123", tract=mock_tract)
        assert ctx.event == "commit"
        assert ctx.commit is None
        assert ctx.branch == "main"
        assert ctx.head == "abc123"
        assert ctx.metrics is None
        assert ctx.rule_index is None

    def test_with_metrics(self):
        mock_tract = MagicMock()
        ctx = EvalContext(event="commit", commit=None, branch="main", head="abc", tract=mock_tract, metrics={"total_tokens": 5000})
        assert ctx.metrics["total_tokens"] == 5000

    def test_frozen(self):
        mock_tract = MagicMock()
        ctx = EvalContext(event="commit", commit=None, branch="main", head="abc", tract=mock_tract)
        with pytest.raises(AttributeError):
            ctx.event = "other"


class TestActionResult:
    def test_creation(self):
        r = ActionResult(action_type="compress", success=True)
        assert r.action_type == "compress"
        assert r.success is True
        assert r.data == {}
        assert r.reason is None

    def test_with_data_and_reason(self):
        r = ActionResult(action_type="block", success=False, data={"count": 1}, reason="too many tokens")
        assert r.data == {"count": 1}
        assert r.reason == "too many tokens"

    def test_frozen(self):
        r = ActionResult(action_type="x", success=True)
        with pytest.raises(AttributeError):
            r.success = False


class TestEvalResult:
    def test_defaults(self):
        r = EvalResult()
        assert r.blocked is False
        assert r.block_reasons == []
        assert r.action_results == []
        assert r.rules_evaluated == 0
        assert r.rules_fired == 0

    def test_with_values(self):
        ar = ActionResult(action_type="compress", success=True)
        r = EvalResult(blocked=True, block_reasons=["token limit"], action_results=[ar], rules_evaluated=3, rules_fired=1)
        assert r.blocked is True
        assert len(r.block_reasons) == 1
        assert len(r.action_results) == 1
        assert r.rules_evaluated == 3
        assert r.rules_fired == 1

    def test_frozen(self):
        r = EvalResult()
        with pytest.raises(AttributeError):
            r.blocked = True
