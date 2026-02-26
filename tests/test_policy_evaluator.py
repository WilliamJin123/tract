"""Tests for Policy ABC, PolicyEvaluator, and Tract integration.

Covers:
- Policy ABC subclassing and instantiation
- PolicyEvaluator priority sorting, trigger filtering, recursion guard
- Autonomous mode: immediate execution
- Collaborative mode: PendingPolicy creation via hook system
- Manual mode: action skipped
- Audit log entries for every triggered evaluation
- Cooldown: rapid evaluations within cooldown_seconds are skipped
- Tract.configure_policies() and register_policy()
- Tract.pause_all_policies() and resume_all_policies()
- Tract.compile() triggers compile-triggered policies
- Tract.commit() triggers commit-triggered policies
- save_policy_config() and load_policy_config() roundtrip
- Error handling: exception in policy.evaluate() is caught and logged
- _execute_action dispatches to correct Tract method
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from tract import (
    DialogueContent,
    EvaluationResult,
    InstructionContent,
    Policy,
    PolicyAction,
    PolicyEvaluator,
    PolicyExecutionError,
    Priority,
    Tract,
)


# ---------------------------------------------------------------------------
# DummyPolicy -- configurable test policy
# ---------------------------------------------------------------------------


class DummyPolicy(Policy):
    """Configurable test policy.

    By default, default_handler does nothing (leaves pending unresolved)
    so tests can verify the proposal lifecycle. Set auto_approve_default=True
    to use the ABC's auto-approve behavior.
    """

    def __init__(
        self,
        name: str = "dummy",
        priority: int = 100,
        trigger: str = "compile",
        action: PolicyAction | None = None,
        should_raise: Exception | None = None,
        auto_approve_default: bool = False,
    ):
        self._name = name
        self._priority = priority
        self._trigger = trigger
        self._action = action
        self._should_raise = should_raise
        self._auto_approve_default = auto_approve_default
        self.evaluate_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def trigger(self) -> str:
        return self._trigger

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        self.evaluate_count += 1
        if self._should_raise:
            raise self._should_raise
        return self._action

    def default_handler(self, pending) -> None:
        """Override: leave pending unresolved by default for test control."""
        if self._auto_approve_default:
            pending.approve()


class RecursivePolicy(Policy):
    """Policy that tries to call tract.compile() during evaluation (tests recursion guard)."""

    @property
    def name(self) -> str:
        return "recursive"

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        # This should NOT cause infinite recursion due to the recursion guard
        tract.compile()
        return None


# ---------------------------------------------------------------------------
# 1. Policy ABC Tests
# ---------------------------------------------------------------------------


class TestPolicyABC:
    """Test Policy ABC can be subclassed and instantiated."""

    def test_subclass_and_instantiate(self):
        """Policy ABC can be subclassed with required methods."""
        p = DummyPolicy(name="test-policy", priority=50, trigger="commit")
        assert p.name == "test-policy"
        assert p.priority == 50
        assert p.trigger == "commit"

    def test_default_priority(self):
        """Default priority is 100."""
        p = DummyPolicy(name="default-prio")
        assert p.priority == 100

    def test_default_trigger(self):
        """Default trigger is 'compile'."""
        p = DummyPolicy(name="default-trigger")
        assert p.trigger == "compile"

    def test_evaluate_returns_none(self):
        """Policy that doesn't fire returns None."""
        p = DummyPolicy(name="no-fire")
        t = Tract.open(":memory:")
        try:
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_evaluate_returns_action(self):
        """Policy that fires returns PolicyAction."""
        action = PolicyAction(action_type="annotate", params={"target_hash": "abc"})
        p = DummyPolicy(name="fire", action=action)
        t = Tract.open(":memory:")
        try:
            result = p.evaluate(t)
            assert result is action
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 2. PolicyEvaluator Tests
# ---------------------------------------------------------------------------


class TestPolicyEvaluator:
    """Test PolicyEvaluator core functionality."""

    def test_sorts_by_priority(self):
        """Policies are sorted by priority on init."""
        p_high = DummyPolicy(name="high", priority=10)
        p_low = DummyPolicy(name="low", priority=200)
        p_mid = DummyPolicy(name="mid", priority=100)

        t = Tract.open(":memory:")
        try:
            ev = PolicyEvaluator(t, policies=[p_low, p_high, p_mid])
            assert [p.name for p in ev._policies] == ["high", "mid", "low"]
        finally:
            t.close()

    def test_filters_by_trigger(self):
        """evaluate() only runs policies matching the trigger."""
        compile_policy = DummyPolicy(name="compile-p", trigger="compile")
        commit_policy = DummyPolicy(name="commit-p", trigger="commit")

        t = Tract.open(":memory:")
        try:
            ev = PolicyEvaluator(t, policies=[compile_policy, commit_policy])

            ev.evaluate(trigger="compile")
            assert compile_policy.evaluate_count == 1
            assert commit_policy.evaluate_count == 0

            ev.evaluate(trigger="commit")
            assert compile_policy.evaluate_count == 1
            assert commit_policy.evaluate_count == 1
        finally:
            t.close()

    def test_recursion_guard(self):
        """Nested evaluate() calls return empty list (recursion guard)."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            recursive = RecursivePolicy()
            ev = PolicyEvaluator(t, policies=[recursive])
            # This should NOT recurse infinitely
            results = ev.evaluate(trigger="compile")
            assert len(results) == 1  # The recursive policy ran once
        finally:
            t.close()

    def test_pause_resume(self):
        """Paused evaluator returns empty list."""
        p = DummyPolicy(name="p")
        t = Tract.open(":memory:")
        try:
            ev = PolicyEvaluator(t, policies=[p])

            ev.pause()
            assert ev.is_paused
            results = ev.evaluate()
            assert results == []
            assert p.evaluate_count == 0

            ev.resume()
            assert not ev.is_paused
            results = ev.evaluate()
            assert len(results) == 1
            assert p.evaluate_count == 1
        finally:
            t.close()

    def test_register_maintains_priority_order(self):
        """register() adds policy and re-sorts by priority."""
        p1 = DummyPolicy(name="p1", priority=100)
        p2 = DummyPolicy(name="p2", priority=50)

        t = Tract.open(":memory:")
        try:
            ev = PolicyEvaluator(t, policies=[p1])
            ev.register(p2)
            assert [p.name for p in ev._policies] == ["p2", "p1"]
        finally:
            t.close()

    def test_unregister(self):
        """unregister() removes policy by name."""
        p1 = DummyPolicy(name="p1")
        p2 = DummyPolicy(name="p2")

        t = Tract.open(":memory:")
        try:
            ev = PolicyEvaluator(t, policies=[p1, p2])
            ev.unregister("p1")
            assert [p.name for p in ev._policies] == ["p2"]
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 3. Autonomous Mode
# ---------------------------------------------------------------------------


class TestAutonomousMode:
    """Test autonomous mode: action executed immediately."""

    def test_annotate_action(self):
        """Autonomous annotate action is executed immediately."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            action = PolicyAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                    "reason": "important",
                },
                reason="Auto-pin important commit",
                autonomy="autonomous",
            )
            p = DummyPolicy(name="auto-pin", action=action)
            ev = PolicyEvaluator(t, policies=[p])

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "executed"

            # Verify the annotation was actually created
            annotations = t.get_annotations(info.commit_hash)
            assert len(annotations) >= 1
            assert annotations[-1].priority == Priority.PINNED
        finally:
            t.close()

    def test_compress_action(self):
        """Autonomous compress action dispatches to Tract.compress()."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="one"))
            t.commit(DialogueContent(role="user", text="two"))
            t.commit(DialogueContent(role="assistant", text="three"))

            action = PolicyAction(
                action_type="compress",
                params={"content": "Summary of conversation"},
                reason="Auto-compress",
                autonomy="autonomous",
            )
            p = DummyPolicy(name="auto-compress", action=action)
            ev = PolicyEvaluator(t, policies=[p])

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "executed"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 4. Collaborative Mode
# ---------------------------------------------------------------------------


class TestCollaborativeMode:
    """Test collaborative mode: creates PendingPolicy via hook system."""

    def test_creates_pending_policy(self):
        """Collaborative mode creates a PendingPolicy and returns 'proposed'."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = PolicyAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                reason="Propose pinning",
                autonomy="collaborative",
            )
            p = DummyPolicy(name="collab-pin", action=action)
            ev = PolicyEvaluator(
                t, policies=[p],
                policy_repo=t._policy_repo,
            )

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "proposed"
        finally:
            t.close()

    def test_collaborative_with_hook_auto_approves(self):
        """Collaborative mode executes when a user hook auto-approves."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = PolicyAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                reason="Propose pinning",
                autonomy="collaborative",
            )
            p = DummyPolicy(name="collab-pin", action=action)

            # Register a hook that auto-approves
            t.on("policy", lambda pending: pending.approve())

            ev = PolicyEvaluator(
                t, policies=[p], policy_repo=t._policy_repo,
            )

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "executed"

            # Verify annotation was created
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()

    def test_collaborative_with_hook_rejects(self):
        """Collaborative mode returns 'proposed' when hook rejects."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = PolicyAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                autonomy="collaborative",
            )
            p = DummyPolicy(name="collab-pin", action=action)

            # Register a hook that rejects
            t.on("policy", lambda pending: pending.reject("Not needed"))

            ev = PolicyEvaluator(
                t, policies=[p], policy_repo=t._policy_repo,
            )

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "proposed"
        finally:
            t.close()

    def test_collaborative_auto_approve_via_default_handler(self):
        """Collaborative mode executes when default_handler approves."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = PolicyAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                reason="Propose pinning",
                autonomy="collaborative",
            )
            p = DummyPolicy(
                name="collab-pin", action=action, auto_approve_default=True,
            )
            ev = PolicyEvaluator(
                t, policies=[p], policy_repo=t._policy_repo,
            )

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "executed"

            # Verify annotation was created
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 5. Audit Log
# ---------------------------------------------------------------------------


class TestAuditLog:
    """Test audit log entries for policy evaluations."""

    def test_audit_log_created(self):
        """Evaluation creates PolicyLogRow entries."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = PolicyAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "normal",
                },
                reason="Test log",
                autonomy="autonomous",
            )
            p = DummyPolicy(name="logged-policy", action=action)
            ev = PolicyEvaluator(
                t, policies=[p], policy_repo=t._policy_repo,
            )

            ev.evaluate()

            # Check audit log
            log_entries = t._policy_repo.get_log(t.tract_id)
            assert len(log_entries) >= 1
            entry = log_entries[0]
            assert entry.policy_name == "logged-policy"
            assert entry.outcome == "executed"
            assert entry.trigger == "compile"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 6. Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    """Test cooldown prevents rapid re-firing."""

    def test_cooldown_skips_rapid_evaluations(self):
        """Within cooldown_seconds, re-evaluations are skipped."""
        action = PolicyAction(
            action_type="annotate",
            params={"target_hash": "abc", "priority": "normal"},
            autonomy="autonomous",
        )
        # Use a policy that always fires
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = PolicyAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "normal",
                },
                autonomy="autonomous",
            )
            p = DummyPolicy(name="cooldown-test", action=action)
            ev = PolicyEvaluator(t, policies=[p], cooldown_seconds=10.0)

            # First evaluation fires
            results1 = ev.evaluate()
            assert results1[0].outcome == "executed"

            # Second evaluation is cooldown-skipped
            results2 = ev.evaluate()
            assert results2[0].outcome == "skipped"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 7. Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in policy evaluation."""

    def test_exception_caught_and_logged(self):
        """Exception in policy.evaluate() is caught and logged as error."""
        p = DummyPolicy(
            name="error-policy",
            should_raise=ValueError("test error"),
        )

        t = Tract.open(":memory:")
        try:
            ev = PolicyEvaluator(
                t, policies=[p], policy_repo=t._policy_repo,
            )
            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "error"
            assert "test error" in results[0].error

            # Check audit log has error entry
            log_entries = t._policy_repo.get_log(t.tract_id)
            assert len(log_entries) >= 1
            assert log_entries[0].outcome == "error"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 8. Tract Integration Tests
# ---------------------------------------------------------------------------


class TestTractIntegration:
    """Test Tract facade policy methods."""

    def test_configure_policies(self):
        """Tract.configure_policies() creates evaluator."""
        t = Tract.open(":memory:")
        try:
            assert t.policy_evaluator is None
            t.configure_policies()
            assert t.policy_evaluator is not None
        finally:
            t.close()

    def test_register_policy_auto_creates_evaluator(self):
        """Tract.register_policy() auto-creates evaluator if needed."""
        t = Tract.open(":memory:")
        try:
            assert t.policy_evaluator is None
            p = DummyPolicy(name="auto-created")
            t.register_policy(p)
            assert t.policy_evaluator is not None
            assert len(t.policy_evaluator._policies) == 1
        finally:
            t.close()

    def test_unregister_policy(self):
        """Tract.unregister_policy() removes policy."""
        t = Tract.open(":memory:")
        try:
            p = DummyPolicy(name="to-remove")
            t.configure_policies(policies=[p])
            assert len(t.policy_evaluator._policies) == 1
            t.unregister_policy("to-remove")
            assert len(t.policy_evaluator._policies) == 0
        finally:
            t.close()

    def test_pause_resume(self):
        """Tract.pause_all_policies() and resume_all_policies()."""
        t = Tract.open(":memory:")
        try:
            p = DummyPolicy(name="pausable")
            t.configure_policies(policies=[p])

            t.pause_all_policies()
            assert t.policy_evaluator.is_paused

            t.resume_all_policies()
            assert not t.policy_evaluator.is_paused
        finally:
            t.close()

    def test_compile_triggers_compile_policies(self):
        """Tract.compile() triggers compile-triggered policies."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = DummyPolicy(name="compile-trigger", trigger="compile")
            t.configure_policies(policies=[p])

            t.compile()
            assert p.evaluate_count == 1
        finally:
            t.close()

    def test_commit_triggers_commit_policies(self):
        """Tract.commit() triggers commit-triggered policies."""
        t = Tract.open(":memory:")
        try:
            # First commit without policies
            t.commit(InstructionContent(text="first"))

            p = DummyPolicy(name="commit-trigger", trigger="commit")
            t.configure_policies(policies=[p])

            # Second commit triggers the policy
            t.commit(DialogueContent(role="user", text="second"))
            assert p.evaluate_count == 1
        finally:
            t.close()

    def test_compile_does_not_trigger_commit_policies(self):
        """Tract.compile() does NOT trigger commit-triggered policies."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = DummyPolicy(name="commit-only", trigger="commit")
            t.configure_policies(policies=[p])

            t.compile()
            assert p.evaluate_count == 0
        finally:
            t.close()

    def test_commit_does_not_trigger_compile_policies(self):
        """Tract.commit() does NOT trigger compile-triggered policies."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="first"))
            p = DummyPolicy(name="compile-only", trigger="compile")
            t.configure_policies(policies=[p])

            t.commit(DialogueContent(role="user", text="second"))
            assert p.evaluate_count == 0
        finally:
            t.close()

    def test_batch_does_not_trigger_policies(self):
        """Policies are not triggered during batch()."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="first"))

            p_compile = DummyPolicy(name="batch-compile", trigger="compile")
            p_commit = DummyPolicy(name="batch-commit", trigger="commit")
            t.configure_policies(policies=[p_compile, p_commit])

            with t.batch():
                t.commit(DialogueContent(role="user", text="a"))
                t.commit(DialogueContent(role="assistant", text="b"))

            assert p_commit.evaluate_count == 0
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 9. Config Persistence
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    """Test save/load policy config via _trace_meta."""

    def test_roundtrip(self):
        """save_policy_config() and load_policy_config() roundtrip."""
        t = Tract.open(":memory:")
        try:
            config = {
                "policies": ["auto-compress"],
                "cooldown_seconds": 30,
                "enabled": True,
            }
            t.save_policy_config(config)

            loaded = t.load_policy_config()
            assert loaded == config
        finally:
            t.close()

    def test_load_returns_none_when_not_set(self):
        """load_policy_config() returns None when no config saved."""
        t = Tract.open(":memory:")
        try:
            assert t.load_policy_config() is None
        finally:
            t.close()

    def test_update_existing_config(self):
        """save_policy_config() updates existing config."""
        t = Tract.open(":memory:")
        try:
            t.save_policy_config({"v": 1})
            t.save_policy_config({"v": 2, "new_key": "value"})

            loaded = t.load_policy_config()
            assert loaded == {"v": 2, "new_key": "value"}
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 10. Dispatch Tests
# ---------------------------------------------------------------------------


class TestDispatch:
    """Test _execute_action dispatches to correct Tract method."""

    def test_branch_action(self):
        """Branch action dispatches to Tract.branch()."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))

            action = PolicyAction(
                action_type="branch",
                params={"name": "policy-branch", "switch": False},
                autonomy="autonomous",
            )
            p = DummyPolicy(name="branch-policy", action=action)
            ev = PolicyEvaluator(t, policies=[p])

            results = ev.evaluate()
            assert results[0].outcome == "executed"

            # Verify branch was created
            branches = t.list_branches()
            branch_names = [b.name for b in branches]
            assert "policy-branch" in branch_names
        finally:
            t.close()

    def test_unknown_action_type_errors(self):
        """Unknown action_type raises ValueError caught as error."""
        t = Tract.open(":memory:")
        try:
            action = PolicyAction(
                action_type="unknown_type",
                params={},
                autonomy="autonomous",
            )
            p = DummyPolicy(name="unknown-action", action=action)
            ev = PolicyEvaluator(t, policies=[p])

            results = ev.evaluate()
            assert results[0].outcome == "error"
            assert "Unknown action_type" in results[0].error
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 11. Manual/Supervised Mode
# ---------------------------------------------------------------------------


class TestManualMode:
    """Test manual mode: action skipped."""

    def test_manual_mode_skips(self):
        """Manual mode creates result with outcome='skipped'."""
        t = Tract.open(":memory:")
        try:
            action = PolicyAction(
                action_type="annotate",
                params={"target_hash": "abc", "priority": "pinned"},
                autonomy="manual",
            )
            p = DummyPolicy(name="manual-policy", action=action)
            ev = PolicyEvaluator(t, policies=[p])

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "skipped"
            assert results[0].triggered is True
        finally:
            t.close()
