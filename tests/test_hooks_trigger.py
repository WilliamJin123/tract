"""Tests for trigger hook integration (Phase 2).

Tests the three-tier handler precedence for collaborative trigger actions:
user hook > trigger_obj.default_handler() > auto-approve.
Also tests on_rejection/on_success callbacks and PendingTrigger lifecycle.
"""

from __future__ import annotations

import pytest

from tract import (
    InstructionContent,
    DialogueContent,
    Trigger,
    TriggerAction,
    TriggerEvaluator,
    Tract,
)
from tract.hooks.trigger import PendingTrigger
from tract.hooks.validation import HookRejection


# ---------------------------------------------------------------------------
# Test Triggers
# ---------------------------------------------------------------------------


class SimpleCollabTrigger(Trigger):
    """Collaborative trigger that always fires."""

    def __init__(self, name: str = "simple-collab", action_type: str = "annotate"):
        self._name = name
        self._action_type = action_type
        self.rejection_count = 0
        self.success_count = 0
        self.last_rejection: HookRejection | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def fires_on(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        head = tract.head
        if head is None:
            return None
        return TriggerAction(
            action_type=self._action_type,
            params={"target_hash": head, "priority": "pinned"},
            reason="Test collaborative action",
            autonomy="collaborative",
        )

    def on_rejection(self, rejection: HookRejection) -> None:
        self.rejection_count += 1
        self.last_rejection = rejection

    def on_success(self, result: object) -> None:
        self.success_count += 1


class RejectingDefaultTrigger(Trigger):
    """Trigger whose default_handler rejects."""

    @property
    def name(self) -> str:
        return "rejecting-default"

    @property
    def fires_on(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        head = tract.head
        if head is None:
            return None
        return TriggerAction(
            action_type="annotate",
            params={"target_hash": head, "priority": "pinned"},
            reason="Should be rejected by default_handler",
            autonomy="collaborative",
        )

    def default_handler(self, pending: PendingTrigger) -> None:
        """Override default_handler to reject."""
        pending.reject("Rejected by trigger default_handler")


class AutoApproveDefaultTrigger(Trigger):
    """Trigger whose default_handler approves (same as base ABC default)."""

    @property
    def name(self) -> str:
        return "auto-approve-default"

    @property
    def fires_on(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        head = tract.head
        if head is None:
            return None
        return TriggerAction(
            action_type="annotate",
            params={"target_hash": head, "priority": "pinned"},
            reason="Should be approved by default_handler",
            autonomy="collaborative",
        )


# ===========================================================================
# 1. Three-tier handler precedence
# ===========================================================================


class TestTriggerThreeTier:
    """Test user hook > default_handler > auto-approve precedence."""

    def test_user_hook_takes_precedence(self):
        """User hook (t.on('trigger', handler)) beats default_handler."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            hook_calls = []
            def user_hook(pending):
                hook_calls.append(pending)
                pending.reject("User says no")

            t.on("trigger", user_hook)

            trigger_obj = AutoApproveDefaultTrigger()
            ev = TriggerEvaluator(t, triggers=[trigger_obj])
            results = ev.evaluate()

            # User hook should have fired, not default_handler
            assert len(hook_calls) == 1
            assert isinstance(hook_calls[0], PendingTrigger)
            assert hook_calls[0].status == "rejected"
        finally:
            t.close()

    def test_default_handler_fires_when_no_user_hook(self):
        """Trigger.default_handler() fires when no user hook registered."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            trigger_obj = RejectingDefaultTrigger()
            ev = TriggerEvaluator(t, triggers=[trigger_obj])
            results = ev.evaluate()

            # default_handler should have rejected
            assert len(results) == 1
            assert results[0].outcome == "proposed"  # Rejected by handler
        finally:
            t.close()

    def test_auto_approve_when_default_handler_approves(self):
        """default_handler that calls approve() executes the action."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            trigger_obj = AutoApproveDefaultTrigger()
            ev = TriggerEvaluator(t, triggers=[trigger_obj])
            results = ev.evaluate()

            # default_handler should have approved and executed
            assert len(results) == 1
            assert results[0].outcome == "executed"
        finally:
            t.close()


# ===========================================================================
# 2. PendingTrigger lifecycle
# ===========================================================================


class TestPendingTriggerLifecycle:
    """Test PendingTrigger creation and methods."""

    def test_pending_trigger_creation(self):
        """PendingTrigger is created with correct fields."""
        t = Tract.open(":memory:")
        try:
            pending = PendingTrigger(
                operation="trigger",
                tract=t,
                trigger_name="test-trigger",
                action_type="annotate",
                action_params={"target_hash": "abc", "priority": "pinned"},
                reason="Test reason",
            )
            assert pending.operation == "trigger"
            assert pending.trigger_name == "test-trigger"
            assert pending.action_type == "annotate"
            assert pending.action_params["target_hash"] == "abc"
            assert pending.reason == "Test reason"
            assert pending.status == "pending"
        finally:
            t.close()

    def test_modify_params(self):
        """modify_params() updates action_params."""
        t = Tract.open(":memory:")
        try:
            pending = PendingTrigger(
                operation="trigger",
                tract=t,
                trigger_name="test",
                action_type="annotate",
                action_params={"target_hash": "abc"},
            )
            pending.modify_params({"priority": "normal"})
            assert pending.action_params["priority"] == "normal"
            assert pending.action_params["target_hash"] == "abc"
        finally:
            t.close()

    def test_modify_params_after_approve_raises(self):
        """modify_params() after approve raises RuntimeError."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            pending = PendingTrigger(
                operation="trigger",
                tract=t,
                trigger_name="test",
                action_type="annotate",
                action_params={"target_hash": info.commit_hash, "priority": "pinned"},
            )
            # Need an execute function
            pending._execute_fn = lambda p: None
            pending.approve()
            with pytest.raises(RuntimeError):
                pending.modify_params({"priority": "normal"})
        finally:
            t.close()

    def test_pending_trigger_reject(self):
        """reject() sets status and reason."""
        t = Tract.open(":memory:")
        try:
            pending = PendingTrigger(
                operation="trigger",
                tract=t,
                trigger_name="test",
                action_type="annotate",
            )
            pending.reject("Not appropriate")
            assert pending.status == "rejected"
            assert pending.rejection_reason == "Not appropriate"
        finally:
            t.close()


# ===========================================================================
# 3. on_rejection / on_success callbacks
# ===========================================================================


class TestTriggerFeedback:
    """Test on_rejection and on_success trigger callbacks."""

    def test_on_success_called_on_approve(self):
        """Trigger.on_success() called when action is approved and executed."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            trigger_obj = SimpleCollabTrigger()
            ev = TriggerEvaluator(t, triggers=[trigger_obj])
            ev.evaluate()

            # default_handler (from ABC) approves -> on_success called
            assert trigger_obj.success_count >= 1
        finally:
            t.close()

    def test_on_rejection_called_on_reject(self):
        """Trigger.on_rejection() called when action is rejected by hook."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            def user_hook(pending):
                pending.reject("User rejects")

            t.on("trigger", user_hook)

            trigger_obj = SimpleCollabTrigger()
            ev = TriggerEvaluator(t, triggers=[trigger_obj])
            ev.evaluate()

            assert trigger_obj.rejection_count == 1
            assert trigger_obj.last_rejection is not None
            assert trigger_obj.last_rejection.reason == "User rejects"
        finally:
            t.close()

    def test_on_rejection_called_when_default_handler_rejects(self):
        """Trigger.on_rejection() called when default_handler rejects."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            trigger_obj = RejectingDefaultTrigger()
            # Patch on_rejection to track
            rejections = []
            original_on_rejection = trigger_obj.on_rejection
            def tracking_on_rejection(rejection):
                rejections.append(rejection)
            trigger_obj.on_rejection = tracking_on_rejection

            ev = TriggerEvaluator(t, triggers=[trigger_obj])
            ev.evaluate()

            assert len(rejections) == 1
            assert "Rejected by trigger default_handler" in rejections[0].reason
        finally:
            t.close()


# ===========================================================================
# 4. Trigger hook with Tract.on()
# ===========================================================================


class TestTriggerHookRegistration:
    """Test t.on('trigger', handler) registration and invocation."""

    def test_register_trigger_hook(self):
        """t.on('trigger', handler) registers successfully."""
        t = Tract.open(":memory:")
        try:
            t.on("trigger", lambda p: p.approve())
            assert "trigger" in t.hooks
        finally:
            t.close()

    def test_off_removes_trigger_hook(self):
        """t.off('trigger') removes the hook."""
        t = Tract.open(":memory:")
        try:
            t.on("trigger", lambda p: p.approve())
            assert "trigger" in t.hooks
            t.off("trigger")
            assert "trigger" not in t.hooks
        finally:
            t.close()

    def test_trigger_hook_receives_pending_trigger(self):
        """Trigger hook handler receives PendingTrigger instance."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            received = []
            def handler(pending):
                received.append(pending)
                pending.approve()

            t.on("trigger", handler)

            trigger_obj = SimpleCollabTrigger()
            ev = TriggerEvaluator(t, triggers=[trigger_obj])
            ev.evaluate()

            assert len(received) == 1
            assert isinstance(received[0], PendingTrigger)
            assert received[0].trigger_name == "simple-collab"
            assert received[0].action_type == "annotate"
        finally:
            t.close()


# ===========================================================================
# 5. Trigger hook replaces on_proposal callback
# ===========================================================================


class TestTriggerHookReplacesOnProposal:
    """Trigger hook captures collaborative proposals (on_proposal removed)."""

    def test_trigger_hook_captures_collaborative(self):
        """Trigger hook fires for collaborative proposals."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            proposals = []
            trigger_obj = SimpleCollabTrigger()

            # Use hook system (new API)
            t.on("trigger", lambda p: proposals.append(p))

            ev = TriggerEvaluator(
                t, triggers=[trigger_obj],
            )
            ev.evaluate()

            # Hook should have been called
            assert len(proposals) >= 1
            assert proposals[0].status in ("pending", "approved")
        finally:
            t.close()


# ===========================================================================
# 6. Trigger ABC new methods
# ===========================================================================


class TestTriggerABCNewMethods:
    """Test the three new optional methods on the Trigger ABC."""

    def test_default_handler_auto_approves(self):
        """Base Trigger.default_handler() auto-approves."""
        t = Tract.open(":memory:")
        try:
            pending = PendingTrigger(
                operation="trigger",
                tract=t,
                trigger_name="test",
                action_type="annotate",
            )
            pending._execute_fn = lambda p: "executed"

            # Use a trigger with default ABC implementation
            trigger_obj = AutoApproveDefaultTrigger()
            trigger_obj.default_handler(pending)

            assert pending.status == "approved"
        finally:
            t.close()

    def test_on_rejection_is_noop_by_default(self):
        """Base Trigger.on_rejection() is a no-op."""
        t = Tract.open(":memory:")
        try:
            pending = PendingTrigger(
                operation="trigger",
                tract=t,
                trigger_name="test",
            )
            rejection = HookRejection(
                reason="test",
                pending=pending,
                rejection_source="hook",
            )
            trigger_obj = AutoApproveDefaultTrigger()
            # Should not raise
            trigger_obj.on_rejection(rejection)
        finally:
            t.close()

    def test_on_success_is_noop_by_default(self):
        """Base Trigger.on_success() is a no-op."""
        t = Tract.open(":memory:")
        try:
            trigger_obj = AutoApproveDefaultTrigger()
            # Should not raise
            trigger_obj.on_success("some_result")
        finally:
            t.close()
