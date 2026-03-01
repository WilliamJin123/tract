"""Tests for Trigger ABC, TriggerEvaluator, and Tract integration.

Covers:
- Trigger ABC subclassing and instantiation
- TriggerEvaluator priority sorting, trigger filtering, recursion guard
- Autonomous mode: immediate execution
- Collaborative mode: PendingTrigger creation via hook system
- Manual mode: action skipped
- Audit log entries for every triggered evaluation
- Cooldown: rapid evaluations within cooldown_seconds are skipped
- Tract.configure_triggers() and register_trigger()
- Tract.pause_all_triggers() and resume_all_triggers()
- Tract.compile() triggers compile-triggered triggers
- Tract.commit() triggers commit-triggered triggers
- save_trigger_config() and load_trigger_config() roundtrip
- Error handling: exception in trigger_obj.evaluate() is caught and logged
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
    Trigger,
    TriggerAction,
    TriggerEvaluator,
    TriggerExecutionError,
    Priority,
    Tract,
)


# ---------------------------------------------------------------------------
# DummyTrigger -- configurable test trigger
# ---------------------------------------------------------------------------


class DummyTrigger(Trigger):
    """Configurable test trigger.

    By default, default_handler does nothing (leaves pending unresolved)
    so tests can verify the proposal lifecycle. Set auto_approve_default=True
    to use the ABC's auto-approve behavior.
    """

    def __init__(
        self,
        name: str = "dummy",
        priority: int = 100,
        trigger: str = "compile",
        action: TriggerAction | None = None,
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
    def fires_on(self) -> str:
        return self._trigger

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        self.evaluate_count += 1
        if self._should_raise:
            raise self._should_raise
        return self._action

    def default_handler(self, pending) -> None:
        """Override: leave pending unresolved by default for test control."""
        if self._auto_approve_default:
            pending.approve()


class RecursiveTrigger(Trigger):
    """Trigger that tries to call tract.compile() during evaluation (tests recursion guard)."""

    @property
    def name(self) -> str:
        return "recursive"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        # This should NOT cause infinite recursion due to the recursion guard
        tract.compile()
        return None


# ---------------------------------------------------------------------------
# 1. Trigger ABC Tests
# ---------------------------------------------------------------------------


class TestTriggerABC:
    """Test Trigger ABC can be subclassed and instantiated."""

    def test_subclass_and_instantiate(self):
        """Trigger ABC can be subclassed with required methods."""
        p = DummyTrigger(name="test-trigger", priority=50, trigger="commit")
        assert p.name == "test-trigger"
        assert p.priority == 50
        assert p.fires_on == "commit"

    def test_default_priority(self):
        """Default priority is 100."""
        p = DummyTrigger(name="default-prio")
        assert p.priority == 100

    def test_default_trigger(self):
        """Default trigger is 'compile'."""
        p = DummyTrigger(name="default-trigger")
        assert p.fires_on == "compile"

    def test_evaluate_returns_none(self):
        """Trigger that doesn't fire returns None."""
        p = DummyTrigger(name="no-fire")
        t = Tract.open(":memory:")
        try:
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_evaluate_returns_action(self):
        """Trigger that fires returns TriggerAction."""
        action = TriggerAction(action_type="annotate", params={"target_hash": "abc"})
        p = DummyTrigger(name="fire", action=action)
        t = Tract.open(":memory:")
        try:
            result = p.evaluate(t)
            assert result is action
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 2. TriggerEvaluator Tests
# ---------------------------------------------------------------------------


class TestTriggerEvaluator:
    """Test TriggerEvaluator core functionality."""

    def test_sorts_by_priority(self):
        """Triggers are sorted by priority on init."""
        p_high = DummyTrigger(name="high", priority=10)
        p_low = DummyTrigger(name="low", priority=200)
        p_mid = DummyTrigger(name="mid", priority=100)

        t = Tract.open(":memory:")
        try:
            ev = TriggerEvaluator(t, triggers=[p_low, p_high, p_mid])
            assert [p.name for p in ev._triggers] == ["high", "mid", "low"]
        finally:
            t.close()

    def test_filters_by_trigger(self):
        """evaluate() only runs triggers matching the trigger."""
        compile_trigger = DummyTrigger(name="compile-p", trigger="compile")
        commit_trigger = DummyTrigger(name="commit-p", trigger="commit")

        t = Tract.open(":memory:")
        try:
            ev = TriggerEvaluator(t, triggers=[compile_trigger, commit_trigger])

            ev.evaluate(trigger="compile")
            assert compile_trigger.evaluate_count == 1
            assert commit_trigger.evaluate_count == 0

            ev.evaluate(trigger="commit")
            assert compile_trigger.evaluate_count == 1
            assert commit_trigger.evaluate_count == 1
        finally:
            t.close()

    def test_recursion_guard(self):
        """Nested evaluate() calls return empty list (recursion guard)."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            recursive = RecursiveTrigger()
            ev = TriggerEvaluator(t, triggers=[recursive])
            # This should NOT recurse infinitely
            results = ev.evaluate(trigger="compile")
            assert len(results) == 1  # The recursive trigger ran once
        finally:
            t.close()

    def test_pause_resume(self):
        """Paused evaluator returns empty list."""
        p = DummyTrigger(name="p")
        t = Tract.open(":memory:")
        try:
            ev = TriggerEvaluator(t, triggers=[p])

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
        """register() adds trigger and re-sorts by priority."""
        p1 = DummyTrigger(name="p1", priority=100)
        p2 = DummyTrigger(name="p2", priority=50)

        t = Tract.open(":memory:")
        try:
            ev = TriggerEvaluator(t, triggers=[p1])
            ev.register(p2)
            assert [p.name for p in ev._triggers] == ["p2", "p1"]
        finally:
            t.close()

    def test_unregister(self):
        """unregister() removes trigger by name."""
        p1 = DummyTrigger(name="p1")
        p2 = DummyTrigger(name="p2")

        t = Tract.open(":memory:")
        try:
            ev = TriggerEvaluator(t, triggers=[p1, p2])
            ev.unregister("p1")
            assert [p.name for p in ev._triggers] == ["p2"]
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

            action = TriggerAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                    "reason": "important",
                },
                reason="Auto-pin important commit",
                autonomy="autonomous",
            )
            p = DummyTrigger(name="auto-pin", action=action)
            ev = TriggerEvaluator(t, triggers=[p])

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

            action = TriggerAction(
                action_type="compress",
                params={"content": "Summary of conversation"},
                reason="Auto-compress",
                autonomy="autonomous",
            )
            p = DummyTrigger(name="auto-compress", action=action)
            ev = TriggerEvaluator(t, triggers=[p])

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "executed"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 4. Collaborative Mode
# ---------------------------------------------------------------------------


class TestCollaborativeMode:
    """Test collaborative mode: creates PendingTrigger via hook system."""

    def test_creates_pending_trigger(self):
        """Collaborative mode creates a PendingTrigger and returns 'proposed'."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = TriggerAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                reason="Propose pinning",
                autonomy="collaborative",
            )
            p = DummyTrigger(name="collab-pin", action=action)
            ev = TriggerEvaluator(
                t, triggers=[p],
                trigger_repo=t._trigger_repo,
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
            action = TriggerAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                reason="Propose pinning",
                autonomy="collaborative",
            )
            p = DummyTrigger(name="collab-pin", action=action)

            # Register a hook that auto-approves
            t.on("trigger", lambda pending: pending.approve())

            ev = TriggerEvaluator(
                t, triggers=[p], trigger_repo=t._trigger_repo,
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
            action = TriggerAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                autonomy="collaborative",
            )
            p = DummyTrigger(name="collab-pin", action=action)

            # Register a hook that rejects
            t.on("trigger", lambda pending: pending.reject("Not needed"))

            ev = TriggerEvaluator(
                t, triggers=[p], trigger_repo=t._trigger_repo,
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
            action = TriggerAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "pinned",
                },
                reason="Propose pinning",
                autonomy="collaborative",
            )
            p = DummyTrigger(
                name="collab-pin", action=action, auto_approve_default=True,
            )
            ev = TriggerEvaluator(
                t, triggers=[p], trigger_repo=t._trigger_repo,
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
    """Test audit log entries for trigger evaluations."""

    def test_audit_log_created(self):
        """Evaluation creates TriggerLogRow entries."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = TriggerAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "normal",
                },
                reason="Test log",
                autonomy="autonomous",
            )
            p = DummyTrigger(name="logged-trigger", action=action)
            ev = TriggerEvaluator(
                t, triggers=[p], trigger_repo=t._trigger_repo,
            )

            ev.evaluate()

            # Check audit log
            log_entries = t._trigger_repo.get_log(t.tract_id)
            assert len(log_entries) >= 1
            entry = log_entries[0]
            assert entry.trigger_name == "logged-trigger"
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
        action = TriggerAction(
            action_type="annotate",
            params={"target_hash": "abc", "priority": "normal"},
            autonomy="autonomous",
        )
        # Use a trigger that always fires
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            action = TriggerAction(
                action_type="annotate",
                params={
                    "target_hash": info.commit_hash,
                    "priority": "normal",
                },
                autonomy="autonomous",
            )
            p = DummyTrigger(name="cooldown-test", action=action)
            ev = TriggerEvaluator(t, triggers=[p], cooldown_seconds=10.0)

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
    """Test error handling in trigger evaluation."""

    def test_exception_caught_and_logged(self):
        """Exception in trigger_obj.evaluate() is caught and logged as error."""
        p = DummyTrigger(
            name="error-trigger",
            should_raise=ValueError("test error"),
        )

        t = Tract.open(":memory:")
        try:
            ev = TriggerEvaluator(
                t, triggers=[p], trigger_repo=t._trigger_repo,
            )
            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "error"
            assert "test error" in results[0].error

            # Check audit log has error entry
            log_entries = t._trigger_repo.get_log(t.tract_id)
            assert len(log_entries) >= 1
            assert log_entries[0].outcome == "error"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 8. Tract Integration Tests
# ---------------------------------------------------------------------------


class TestTractIntegration:
    """Test Tract facade trigger methods."""

    def test_configure_triggers(self):
        """Tract.configure_triggers() creates evaluator."""
        t = Tract.open(":memory:")
        try:
            assert t.trigger_evaluator is None
            t.configure_triggers()
            assert t.trigger_evaluator is not None
        finally:
            t.close()

    def test_register_trigger_auto_creates_evaluator(self):
        """Tract.register_trigger() auto-creates evaluator if needed."""
        t = Tract.open(":memory:")
        try:
            assert t.trigger_evaluator is None
            p = DummyTrigger(name="auto-created")
            t.register_trigger(p)
            assert t.trigger_evaluator is not None
            assert len(t.trigger_evaluator._triggers) == 1
        finally:
            t.close()

    def test_unregister_trigger(self):
        """Tract.unregister_trigger() removes trigger."""
        t = Tract.open(":memory:")
        try:
            p = DummyTrigger(name="to-remove")
            t.configure_triggers(triggers=[p])
            assert len(t.trigger_evaluator._triggers) == 1
            t.unregister_trigger("to-remove")
            assert len(t.trigger_evaluator._triggers) == 0
        finally:
            t.close()

    def test_pause_resume(self):
        """Tract.pause_all_triggers() and resume_all_triggers()."""
        t = Tract.open(":memory:")
        try:
            p = DummyTrigger(name="pausable")
            t.configure_triggers(triggers=[p])

            t.pause_all_triggers()
            assert t.trigger_evaluator.is_paused

            t.resume_all_triggers()
            assert not t.trigger_evaluator.is_paused
        finally:
            t.close()

    def test_compile_triggers_compile_triggers(self):
        """Tract.compile() triggers compile-triggered triggers."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = DummyTrigger(name="compile-trigger", trigger="compile")
            t.configure_triggers(triggers=[p])

            t.compile()
            assert p.evaluate_count == 1
        finally:
            t.close()

    def test_commit_triggers_commit_triggers(self):
        """Tract.commit() triggers commit-triggered triggers."""
        t = Tract.open(":memory:")
        try:
            # First commit without triggers
            t.commit(InstructionContent(text="first"))

            p = DummyTrigger(name="commit-trigger", trigger="commit")
            t.configure_triggers(triggers=[p])

            # Second commit triggers the trigger
            t.commit(DialogueContent(role="user", text="second"))
            assert p.evaluate_count == 1
        finally:
            t.close()

    def test_compile_does_not_trigger_commit_triggers(self):
        """Tract.compile() does NOT trigger commit-triggered triggers."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = DummyTrigger(name="commit-only", trigger="commit")
            t.configure_triggers(triggers=[p])

            t.compile()
            assert p.evaluate_count == 0
        finally:
            t.close()

    def test_commit_does_not_trigger_compile_triggers(self):
        """Tract.commit() does NOT trigger compile-triggered triggers."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="first"))
            p = DummyTrigger(name="compile-only", trigger="compile")
            t.configure_triggers(triggers=[p])

            t.commit(DialogueContent(role="user", text="second"))
            assert p.evaluate_count == 0
        finally:
            t.close()

    def test_batch_does_not_trigger_triggers(self):
        """Triggers are not triggered during batch()."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="first"))

            p_compile = DummyTrigger(name="batch-compile", trigger="compile")
            p_commit = DummyTrigger(name="batch-commit", trigger="commit")
            t.configure_triggers(triggers=[p_compile, p_commit])

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
    """Test save/load trigger config via _trace_meta."""

    def test_roundtrip(self):
        """save_trigger_config() and load_trigger_config() roundtrip."""
        t = Tract.open(":memory:")
        try:
            config = {
                "triggers": ["auto-compress"],
                "cooldown_seconds": 30,
                "enabled": True,
            }
            t.save_trigger_config(config)

            loaded = t.load_trigger_config()
            assert loaded == config
        finally:
            t.close()

    def test_load_returns_none_when_not_set(self):
        """load_trigger_config() returns None when no config saved."""
        t = Tract.open(":memory:")
        try:
            assert t.load_trigger_config() is None
        finally:
            t.close()

    def test_update_existing_config(self):
        """save_trigger_config() updates existing config."""
        t = Tract.open(":memory:")
        try:
            t.save_trigger_config({"v": 1})
            t.save_trigger_config({"v": 2, "new_key": "value"})

            loaded = t.load_trigger_config()
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

            action = TriggerAction(
                action_type="branch",
                params={"name": "trigger-branch", "switch": False},
                autonomy="autonomous",
            )
            p = DummyTrigger(name="branch-trigger", action=action)
            ev = TriggerEvaluator(t, triggers=[p])

            results = ev.evaluate()
            assert results[0].outcome == "executed"

            # Verify branch was created
            branches = t.list_branches()
            branch_names = [b.name for b in branches]
            assert "trigger-branch" in branch_names
        finally:
            t.close()

    def test_unknown_action_type_errors(self):
        """Unknown action_type raises ValueError caught as error."""
        t = Tract.open(":memory:")
        try:
            action = TriggerAction(
                action_type="unknown_type",
                params={},
                autonomy="autonomous",
            )
            p = DummyTrigger(name="unknown-action", action=action)
            ev = TriggerEvaluator(t, triggers=[p])

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
            action = TriggerAction(
                action_type="annotate",
                params={"target_hash": "abc", "priority": "pinned"},
                autonomy="manual",
            )
            p = DummyTrigger(name="manual-trigger", action=action)
            ev = TriggerEvaluator(t, triggers=[p])

            results = ev.evaluate()
            assert len(results) == 1
            assert results[0].outcome == "skipped"
            assert results[0].triggered is True
        finally:
            t.close()
