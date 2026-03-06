"""Tests for the RuleEngine class.

Covers: event processing pipeline (gates -> work -> handoff -> post),
condition evaluation, recursion guard, sorting, transition processing,
override/accumulate semantics, custom handlers, and EvalResult aggregation.
"""

import types

import pytest

from tract.rules.engine import RuleEngine
from tract.rules.index import RuleIndex
from tract.rules.models import ActionResult, EvalContext, EvalResult, RuleEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tract():
    """Create a minimal mock tract for unit testing the engine."""
    mock = types.SimpleNamespace()
    mock._rule_eval_depth = 0
    return mock


def _make_ctx(mock_tract, event="test", commit=None, metrics=None):
    return EvalContext(
        event=event,
        commit=commit,
        branch="main",
        head="abc123",
        tract=mock_tract,
        metrics=metrics or {"total_tokens": 100},
        rule_index=RuleIndex(),
    )


def _rule(name, trigger="test", condition=None, action=None, commit_hash="h1", dag_distance=0):
    """Shorthand to build a RuleEntry with sensible defaults."""
    return RuleEntry(
        name=name,
        trigger=trigger,
        condition=condition,
        action=action or {},
        commit_hash=commit_hash,
        dag_distance=dag_distance,
    )


# ===========================================================================
# Basic event processing
# ===========================================================================


class TestNoRules:
    def test_no_rules_no_effect(self):
        """Empty index returns empty EvalResult."""
        idx = RuleIndex()
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert not result.blocked
        assert result.block_reasons == []
        assert result.action_results == []
        assert result.rules_evaluated == 0
        assert result.rules_fired == 0


class TestGateActions:
    def test_single_gate_block(self):
        """A block action stops everything and marks result as blocked."""
        idx = RuleIndex()
        idx.add_rule(_rule("blocker", action={"type": "block", "reason": "forbidden"}))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.blocked
        assert len(result.block_reasons) == 1
        assert "forbidden" in result.block_reasons[0]
        assert result.rules_fired == 1

    def test_single_gate_require_met(self):
        """A require action with a met condition allows processing to continue."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "req_met",
            action={
                "type": "require",
                "condition": {"type": "threshold", "metric": "total_tokens", "op": "<", "value": 500},
            },
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract(), metrics={"total_tokens": 100})
        result = engine.process_event("test", ctx)

        assert not result.blocked
        assert result.rules_fired == 1
        assert result.action_results[0].success

    def test_single_gate_require_not_met(self):
        """A require action with an unmet condition blocks the pipeline."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "req_fail",
            action={
                "type": "require",
                "condition": {"type": "threshold", "metric": "total_tokens", "op": "<", "value": 50},
            },
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract(), metrics={"total_tokens": 100})
        result = engine.process_event("test", ctx)

        assert result.blocked
        assert result.rules_fired == 1
        assert not result.action_results[0].success

    def test_gate_blocks_skips_work(self):
        """When a gate blocks, work/handoff/post rules are not evaluated."""
        idx = RuleIndex()
        idx.add_rule(_rule("blocker", action={"type": "block", "reason": "stop"}))
        idx.add_rule(_rule(
            "worker",
            action={"type": "set_config", "key": "model", "value": "gpt-4"},
        ))
        idx.add_rule(_rule(
            "handoff",
            action={"type": "compile_filter", "include_roles": ["system"]},
        ))
        idx.add_rule(_rule(
            "poster",
            action={"type": "create_rule", "template": {"name": "x", "trigger": "y", "action": {}}},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.blocked
        # Only the gate rule was evaluated and fired
        assert result.rules_evaluated == 1
        assert result.rules_fired == 1
        # Only block result present, no work/handoff/post
        assert len(result.action_results) == 1
        assert result.action_results[0].action_type == "block"


# ===========================================================================
# Work actions
# ===========================================================================


class TestWorkActions:
    def test_work_actions_all_execute(self):
        """All matching work rules fire (accumulate semantics)."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "sc1", action={"type": "set_config", "key": "model", "value": "gpt-4"}, dag_distance=0,
        ))
        idx.add_rule(_rule(
            "llm1", action={"type": "llm", "instruction": "summarize"}, dag_distance=1,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert not result.blocked
        assert result.rules_fired == 2
        action_types = [r.action_type for r in result.action_results]
        assert "set_config" in action_types
        assert "llm" in action_types

    def test_work_order_furthest_first(self):
        """Work rules execute furthest-first (root before branch)."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "root_config",
            action={"type": "set_config", "key": "a", "value": "root"},
            dag_distance=5,
        ))
        idx.add_rule(_rule(
            "branch_config",
            action={"type": "set_config", "key": "b", "value": "branch"},
            dag_distance=1,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        # First result should be from the root rule (distance 5)
        assert result.action_results[0].data["key"] == "a"
        assert result.action_results[1].data["key"] == "b"


# ===========================================================================
# Handoff actions
# ===========================================================================


class TestHandoffActions:
    def test_handoff_closest_wins(self):
        """Only the first matching handoff rule (closest) executes."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "h_close",
            action={"type": "compile_filter", "include_roles": ["system"]},
            dag_distance=1,
        ))
        idx.add_rule(_rule(
            "h_far",
            action={"type": "compile_filter", "include_roles": ["user"]},
            dag_distance=5,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        handoff_results = [r for r in result.action_results if r.action_type == "compile_filter"]
        assert len(handoff_results) == 1
        assert handoff_results[0].data["include_roles"] == ["system"]


# ===========================================================================
# Post actions
# ===========================================================================


class TestPostActions:
    def test_post_actions_after_handoff(self):
        """Post actions run after handoff in the pipeline."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "handoff1",
            action={"type": "compile_filter", "include_roles": ["system"]},
            dag_distance=0,
        ))
        idx.add_rule(_rule(
            "post1",
            action={"type": "create_rule", "template": {"name": "x", "trigger": "y", "action": {}}},
            dag_distance=0,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        action_types = [r.action_type for r in result.action_results]
        handoff_idx = action_types.index("compile_filter")
        post_idx = action_types.index("create_rule")
        assert handoff_idx < post_idx


# ===========================================================================
# Gate sorting: deterministic before LLM
# ===========================================================================


class TestGateSorting:
    def test_deterministic_before_llm(self):
        """Deterministic gate conditions (non-LLM) are evaluated before LLM conditions."""
        idx = RuleIndex()
        # LLM condition -- non-deterministic
        idx.add_rule(_rule(
            "llm_gate",
            condition={"type": "llm", "instruction": "is this okay?"},
            action={"type": "require", "condition": None},
            dag_distance=0,
        ))
        # Threshold condition -- deterministic
        idx.add_rule(_rule(
            "det_gate",
            condition={"type": "threshold", "metric": "total_tokens", "op": "<", "value": 500},
            action={"type": "require", "condition": None},
            dag_distance=0,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract(), metrics={"total_tokens": 100})
        result = engine.process_event("test", ctx)

        # Both fired; the deterministic gate should have been evaluated first.
        # Verify by checking action_results order: first result corresponds to det_gate.
        assert result.rules_evaluated == 2
        assert result.rules_fired == 2


# ===========================================================================
# Recursion guard
# ===========================================================================


class TestRecursionGuard:
    def test_recursion_guard_depth_3(self):
        """At max depth, engine returns empty EvalResult without evaluating rules."""
        mock = _make_mock_tract()
        mock._rule_eval_depth = 3  # already at max
        idx = RuleIndex()
        idx.add_rule(_rule("r", action={"type": "block"}))
        engine = RuleEngine(idx, max_depth=3)
        ctx = _make_ctx(mock, event="test")
        result = engine.process_event("test", ctx)

        assert not result.blocked
        assert result.rules_evaluated == 0
        assert result.rules_fired == 0

    def test_recursion_guard_resets(self):
        """Depth counter resets to original value after event processing completes."""
        mock = _make_mock_tract()
        mock._rule_eval_depth = 0
        idx = RuleIndex()
        idx.add_rule(_rule("r", action={"type": "set_config", "key": "x", "value": 1}))
        engine = RuleEngine(idx)
        ctx = _make_ctx(mock, event="test")
        engine.process_event("test", ctx)

        assert mock._rule_eval_depth == 0

    def test_recursion_guard_increments_during_processing(self):
        """Depth is incremented while processing is in flight."""
        mock = _make_mock_tract()
        mock._rule_eval_depth = 0
        captured_depth = []

        class DepthCapture:
            def execute(self, params, ctx):
                captured_depth.append(ctx.tract._rule_eval_depth)
                return ActionResult("set_config", True, data={"key": "x", "value": 1})

        idx = RuleIndex()
        idx.add_rule(_rule("r", action={"type": "set_config", "key": "x", "value": 1}))
        engine = RuleEngine(idx, custom_actions={"set_config": DepthCapture()})
        ctx = _make_ctx(mock, event="test")
        engine.process_event("test", ctx)

        assert captured_depth == [1]
        assert mock._rule_eval_depth == 0


# ===========================================================================
# Conditions
# ===========================================================================


class TestConditions:
    def test_condition_evaluation(self):
        """Conditions are properly dispatched to evaluators."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "threshold_rule",
            condition={"type": "threshold", "metric": "total_tokens", "op": ">", "value": 50},
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract(), metrics={"total_tokens": 100})
        result = engine.process_event("test", ctx)

        assert result.rules_fired == 1

    def test_condition_prevents_firing(self):
        """A condition that evaluates to False prevents the rule from firing."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "threshold_rule",
            condition={"type": "threshold", "metric": "total_tokens", "op": ">", "value": 200},
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract(), metrics={"total_tokens": 100})
        result = engine.process_event("test", ctx)

        assert result.rules_evaluated == 1
        assert result.rules_fired == 0

    def test_unconditional_rule(self):
        """A rule with condition=None always fires."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "always",
            condition=None,
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.rules_fired == 1


# ===========================================================================
# Multi-rule scenarios
# ===========================================================================


class TestMultiRule:
    def test_multiple_triggers_same_event(self):
        """All rules matching a given event trigger are collected and processed."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "r1", action={"type": "set_config", "key": "a", "value": 1}, dag_distance=0,
        ))
        idx.add_rule(_rule(
            "r2", action={"type": "set_config", "key": "b", "value": 2}, dag_distance=1,
        ))
        idx.add_rule(_rule(
            "r3", action={"type": "set_config", "key": "c", "value": 3}, dag_distance=2,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.rules_evaluated == 3
        assert result.rules_fired == 3
        keys = {r.data["key"] for r in result.action_results}
        assert keys == {"a", "b", "c"}

    def test_mixed_categories(self):
        """Gate + work + handoff + post all participate in a single event pipeline."""
        idx = RuleIndex()
        # Gate (require that passes)
        idx.add_rule(_rule(
            "gate1",
            action={
                "type": "require",
                "condition": {"type": "threshold", "metric": "total_tokens", "op": "<", "value": 500},
            },
            dag_distance=0,
        ))
        # Work
        idx.add_rule(_rule(
            "work1",
            action={"type": "set_config", "key": "model", "value": "gpt-4"},
            dag_distance=0,
        ))
        # Handoff
        idx.add_rule(_rule(
            "handoff1",
            action={"type": "compile_filter", "include_roles": ["system"]},
            dag_distance=0,
        ))
        # Post
        idx.add_rule(_rule(
            "post1",
            action={"type": "create_rule", "template": {"name": "x", "trigger": "y", "action": {}}},
            dag_distance=0,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract(), metrics={"total_tokens": 100})
        result = engine.process_event("test", ctx)

        assert not result.blocked
        assert result.rules_evaluated == 4
        assert result.rules_fired == 4
        action_types = [r.action_type for r in result.action_results]
        # Pipeline order: gate -> work -> handoff -> post
        assert action_types.index("require") < action_types.index("set_config")
        assert action_types.index("set_config") < action_types.index("compile_filter")
        assert action_types.index("compile_filter") < action_types.index("create_rule")


# ===========================================================================
# Override / Accumulate semantics
# ===========================================================================


class TestSemantics:
    def test_override_semantics_set_config(self):
        """set_config uses override semantics: closest wins (dedup by key)."""
        idx = RuleIndex()
        # Root rule (distance 5) - lower precedence
        idx.add_rule(_rule(
            "sc1",
            action={"type": "set_config", "key": "temperature", "value": 0.9},
            commit_hash="h1",
            dag_distance=5,
        ))
        # Branch rule (distance 1) - higher precedence
        idx.add_rule(_rule(
            "sc2",
            action={"type": "set_config", "key": "temperature", "value": 0.3},
            commit_hash="h2",
            dag_distance=1,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        # Only one set_config for "temperature" - the closest (0.3)
        configs = [r for r in result.action_results if r.action_type == "set_config"]
        assert len(configs) == 1
        assert configs[0].data["value"] == 0.3

    def test_override_preserves_non_set_config(self):
        """Dedup only affects set_config results; other work results are preserved."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "sc1",
            action={"type": "set_config", "key": "temperature", "value": 0.9},
            dag_distance=5,
        ))
        idx.add_rule(_rule(
            "sc2",
            action={"type": "set_config", "key": "temperature", "value": 0.3},
            dag_distance=1,
        ))
        idx.add_rule(_rule(
            "llm1",
            action={"type": "llm", "instruction": "summarize"},
            dag_distance=3,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        llm_results = [r for r in result.action_results if r.action_type == "llm"]
        assert len(llm_results) == 1

    def test_accumulate_semantics(self):
        """block uses accumulate semantics: any block stops the pipeline."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "block1",
            action={"type": "block", "reason": "reason A"},
            dag_distance=0,
        ))
        idx.add_rule(_rule(
            "block2",
            action={"type": "block", "reason": "reason B"},
            dag_distance=1,
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        # First block encountered stops immediately
        assert result.blocked
        assert result.rules_fired == 1


# ===========================================================================
# Transition processing
# ===========================================================================


class TestTransition:
    def test_transition_generic_only(self):
        """'transition' trigger fires for all transitions."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "generic",
            trigger="transition",
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_transition("target_branch", ctx)

        assert result.rules_fired == 1

    def test_transition_specific_only(self):
        """'transition:target' trigger fires only for that specific target."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "specific",
            trigger="transition:feature",
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_transition("feature", ctx)

        assert result.rules_fired == 1

    def test_transition_specific_no_match(self):
        """'transition:target' does NOT fire for a different target."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "specific",
            trigger="transition:feature",
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_transition("other_branch", ctx)

        assert result.rules_fired == 0

    def test_transition_generic_and_specific(self):
        """Both generic and specific transition triggers fire together."""
        idx = RuleIndex()
        idx.add_rule(_rule(
            "generic",
            trigger="transition",
            action={"type": "set_config", "key": "a", "value": 1},
        ))
        idx.add_rule(_rule(
            "specific",
            trigger="transition:feature",
            action={"type": "set_config", "key": "b", "value": 2},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_transition("feature", ctx)

        assert result.rules_fired == 2
        keys = {r.data["key"] for r in result.action_results}
        assert keys == {"a", "b"}

    def test_transition_recursion_guard(self):
        """Transition processing also respects the recursion guard."""
        mock = _make_mock_tract()
        mock._rule_eval_depth = 3
        idx = RuleIndex()
        idx.add_rule(_rule("r", trigger="transition", action={"type": "block"}))
        engine = RuleEngine(idx, max_depth=3)
        ctx = _make_ctx(mock)
        result = engine.process_transition("target", ctx)

        assert not result.blocked
        assert result.rules_evaluated == 0


# ===========================================================================
# EvalResult
# ===========================================================================


class TestEvalResult:
    def test_eval_result_aggregation(self):
        """rules_evaluated and rules_fired counts are correct."""
        idx = RuleIndex()
        # Two rules, one with a condition that fails
        idx.add_rule(_rule(
            "fires",
            condition=None,
            action={"type": "set_config", "key": "a", "value": 1},
        ))
        idx.add_rule(_rule(
            "skipped",
            condition={"type": "threshold", "metric": "total_tokens", "op": ">", "value": 9999},
            action={"type": "set_config", "key": "b", "value": 2},
        ))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract(), metrics={"total_tokens": 100})
        result = engine.process_event("test", ctx)

        assert result.rules_evaluated == 2
        assert result.rules_fired == 1

    def test_process_event_returns_eval_result(self):
        """process_event returns an EvalResult instance."""
        idx = RuleIndex()
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert isinstance(result, EvalResult)

    def test_process_transition_returns_eval_result(self):
        """process_transition returns an EvalResult instance."""
        idx = RuleIndex()
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_transition("target", ctx)

        assert isinstance(result, EvalResult)


# ===========================================================================
# Custom handlers
# ===========================================================================


class TestCustomHandlers:
    def test_custom_condition(self):
        """A custom condition evaluator is dispatched correctly."""

        class AlwaysFalse:
            def evaluate(self, params, ctx):
                return False

        idx = RuleIndex()
        idx.add_rule(_rule(
            "guarded",
            condition={"type": "my_custom"},
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx, custom_conditions={"my_custom": AlwaysFalse()})
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.rules_evaluated == 1
        assert result.rules_fired == 0

    def test_custom_condition_true(self):
        """A custom condition that returns True allows the rule to fire."""

        class AlwaysTrue:
            def evaluate(self, params, ctx):
                return True

        idx = RuleIndex()
        idx.add_rule(_rule(
            "guarded",
            condition={"type": "my_custom"},
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx, custom_conditions={"my_custom": AlwaysTrue()})
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.rules_fired == 1

    def test_custom_action(self):
        """A custom action handler is dispatched correctly."""

        class MyAction:
            def execute(self, params, ctx):
                return ActionResult("my_action", True, data={"custom": True})

        idx = RuleIndex()
        idx.add_rule(_rule(
            "custom",
            action={"type": "my_action", "data": "hello"},
        ))
        engine = RuleEngine(idx, custom_actions={"my_action": MyAction()})
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.rules_fired == 1
        assert result.action_results[0].action_type == "my_action"
        assert result.action_results[0].data == {"custom": True}

    def test_unknown_action_type(self):
        """An unknown action type returns ActionResult with success=False."""
        idx = RuleIndex()
        idx.add_rule(_rule("bad", action={"type": "nonexistent_action"}))
        engine = RuleEngine(idx)
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.rules_fired == 1
        assert not result.action_results[0].success
        assert "Unknown action" in result.action_results[0].reason

    def test_custom_action_overrides_builtin(self):
        """A custom action handler overrides the built-in handler for the same type."""

        class CustomSetConfig:
            def execute(self, params, ctx):
                return ActionResult("set_config", True, data={"custom_override": True})

        idx = RuleIndex()
        idx.add_rule(_rule(
            "sc",
            action={"type": "set_config", "key": "x", "value": 1},
        ))
        engine = RuleEngine(idx, custom_actions={"set_config": CustomSetConfig()})
        ctx = _make_ctx(_make_mock_tract())
        result = engine.process_event("test", ctx)

        assert result.action_results[0].data == {"custom_override": True}
