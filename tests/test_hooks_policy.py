"""Tests for policy hook integration (Phase 2).

Tests the three-tier handler precedence for collaborative policy actions:
user hook > policy.default_handler() > auto-approve.
Also tests on_rejection/on_success callbacks and PendingPolicy lifecycle.
"""

from __future__ import annotations

import pytest

from tract import (
    InstructionContent,
    DialogueContent,
    Policy,
    PolicyAction,
    PolicyEvaluator,
    Tract,
)
from tract.hooks.policy import PendingPolicy
from tract.hooks.validation import HookRejection


# ---------------------------------------------------------------------------
# Test Policies
# ---------------------------------------------------------------------------


class SimpleCollabPolicy(Policy):
    """Collaborative policy that always fires."""

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
    def trigger(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        head = tract.head
        if head is None:
            return None
        return PolicyAction(
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


class RejectingDefaultPolicy(Policy):
    """Policy whose default_handler rejects."""

    @property
    def name(self) -> str:
        return "rejecting-default"

    @property
    def trigger(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        head = tract.head
        if head is None:
            return None
        return PolicyAction(
            action_type="annotate",
            params={"target_hash": head, "priority": "pinned"},
            reason="Should be rejected by default_handler",
            autonomy="collaborative",
        )

    def default_handler(self, pending: PendingPolicy) -> None:
        """Override default_handler to reject."""
        pending.reject("Rejected by policy default_handler")


class AutoApproveDefaultPolicy(Policy):
    """Policy whose default_handler approves (same as base ABC default)."""

    @property
    def name(self) -> str:
        return "auto-approve-default"

    @property
    def trigger(self) -> str:
        return "compile"

    def evaluate(self, tract: Tract) -> PolicyAction | None:
        head = tract.head
        if head is None:
            return None
        return PolicyAction(
            action_type="annotate",
            params={"target_hash": head, "priority": "pinned"},
            reason="Should be approved by default_handler",
            autonomy="collaborative",
        )


# ===========================================================================
# 1. Three-tier handler precedence
# ===========================================================================


class TestPolicyThreeTier:
    """Test user hook > default_handler > auto-approve precedence."""

    def test_user_hook_takes_precedence(self):
        """User hook (t.on('policy', handler)) beats default_handler."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            hook_calls = []
            def user_hook(pending):
                hook_calls.append(pending)
                pending.reject("User says no")

            t.on("policy", user_hook)

            policy = AutoApproveDefaultPolicy()
            ev = PolicyEvaluator(t, policies=[policy])
            results = ev.evaluate()

            # User hook should have fired, not default_handler
            assert len(hook_calls) == 1
            assert isinstance(hook_calls[0], PendingPolicy)
            assert hook_calls[0].status == "rejected"
        finally:
            t.close()

    def test_default_handler_fires_when_no_user_hook(self):
        """Policy.default_handler() fires when no user hook registered."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            policy = RejectingDefaultPolicy()
            ev = PolicyEvaluator(t, policies=[policy])
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

            policy = AutoApproveDefaultPolicy()
            ev = PolicyEvaluator(t, policies=[policy])
            results = ev.evaluate()

            # default_handler should have approved and executed
            assert len(results) == 1
            assert results[0].outcome == "executed"
        finally:
            t.close()


# ===========================================================================
# 2. PendingPolicy lifecycle
# ===========================================================================


class TestPendingPolicyLifecycle:
    """Test PendingPolicy creation and methods."""

    def test_pending_policy_creation(self):
        """PendingPolicy is created with correct fields."""
        t = Tract.open(":memory:")
        try:
            pending = PendingPolicy(
                operation="policy",
                tract=t,
                policy_name="test-policy",
                action_type="annotate",
                action_params={"target_hash": "abc", "priority": "pinned"},
                reason="Test reason",
            )
            assert pending.operation == "policy"
            assert pending.policy_name == "test-policy"
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
            pending = PendingPolicy(
                operation="policy",
                tract=t,
                policy_name="test",
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
            pending = PendingPolicy(
                operation="policy",
                tract=t,
                policy_name="test",
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

    def test_pending_policy_reject(self):
        """reject() sets status and reason."""
        t = Tract.open(":memory:")
        try:
            pending = PendingPolicy(
                operation="policy",
                tract=t,
                policy_name="test",
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


class TestPolicyFeedback:
    """Test on_rejection and on_success policy callbacks."""

    def test_on_success_called_on_approve(self):
        """Policy.on_success() called when action is approved and executed."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            policy = SimpleCollabPolicy()
            ev = PolicyEvaluator(t, policies=[policy])
            ev.evaluate()

            # default_handler (from ABC) approves -> on_success called
            assert policy.success_count >= 1
        finally:
            t.close()

    def test_on_rejection_called_on_reject(self):
        """Policy.on_rejection() called when action is rejected by hook."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            def user_hook(pending):
                pending.reject("User rejects")

            t.on("policy", user_hook)

            policy = SimpleCollabPolicy()
            ev = PolicyEvaluator(t, policies=[policy])
            ev.evaluate()

            assert policy.rejection_count == 1
            assert policy.last_rejection is not None
            assert policy.last_rejection.reason == "User rejects"
        finally:
            t.close()

    def test_on_rejection_called_when_default_handler_rejects(self):
        """Policy.on_rejection() called when default_handler rejects."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            policy = RejectingDefaultPolicy()
            # Patch on_rejection to track
            rejections = []
            original_on_rejection = policy.on_rejection
            def tracking_on_rejection(rejection):
                rejections.append(rejection)
            policy.on_rejection = tracking_on_rejection

            ev = PolicyEvaluator(t, policies=[policy])
            ev.evaluate()

            assert len(rejections) == 1
            assert "Rejected by policy default_handler" in rejections[0].reason
        finally:
            t.close()


# ===========================================================================
# 4. Policy hook with Tract.on()
# ===========================================================================


class TestPolicyHookRegistration:
    """Test t.on('policy', handler) registration and invocation."""

    def test_register_policy_hook(self):
        """t.on('policy', handler) registers successfully."""
        t = Tract.open(":memory:")
        try:
            t.on("policy", lambda p: p.approve())
            assert "policy" in t.hooks
        finally:
            t.close()

    def test_off_removes_policy_hook(self):
        """t.off('policy') removes the hook."""
        t = Tract.open(":memory:")
        try:
            t.on("policy", lambda p: p.approve())
            assert "policy" in t.hooks
            t.off("policy")
            assert "policy" not in t.hooks
        finally:
            t.close()

    def test_policy_hook_receives_pending_policy(self):
        """Policy hook handler receives PendingPolicy instance."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            received = []
            def handler(pending):
                received.append(pending)
                pending.approve()

            t.on("policy", handler)

            policy = SimpleCollabPolicy()
            ev = PolicyEvaluator(t, policies=[policy])
            ev.evaluate()

            assert len(received) == 1
            assert isinstance(received[0], PendingPolicy)
            assert received[0].policy_name == "simple-collab"
            assert received[0].action_type == "annotate"
        finally:
            t.close()


# ===========================================================================
# 5. Policy hook replaces on_proposal callback
# ===========================================================================


class TestPolicyHookReplacesOnProposal:
    """Policy hook captures collaborative proposals (on_proposal removed)."""

    def test_policy_hook_captures_collaborative(self):
        """Policy hook fires for collaborative proposals."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            proposals = []
            policy = SimpleCollabPolicy()

            # Use hook system (new API)
            t.on("policy", lambda p: proposals.append(p))

            ev = PolicyEvaluator(
                t, policies=[policy],
            )
            ev.evaluate()

            # Hook should have been called
            assert len(proposals) >= 1
            assert proposals[0].status in ("pending", "approved")
        finally:
            t.close()


# ===========================================================================
# 6. Policy ABC new methods
# ===========================================================================


class TestPolicyABCNewMethods:
    """Test the three new optional methods on the Policy ABC."""

    def test_default_handler_auto_approves(self):
        """Base Policy.default_handler() auto-approves."""
        t = Tract.open(":memory:")
        try:
            pending = PendingPolicy(
                operation="policy",
                tract=t,
                policy_name="test",
                action_type="annotate",
            )
            pending._execute_fn = lambda p: "executed"

            # Use a policy with default ABC implementation
            policy = AutoApproveDefaultPolicy()
            policy.default_handler(pending)

            assert pending.status == "approved"
        finally:
            t.close()

    def test_on_rejection_is_noop_by_default(self):
        """Base Policy.on_rejection() is a no-op."""
        t = Tract.open(":memory:")
        try:
            pending = PendingPolicy(
                operation="policy",
                tract=t,
                policy_name="test",
            )
            rejection = HookRejection(
                reason="test",
                pending=pending,
                rejection_source="hook",
            )
            policy = AutoApproveDefaultPolicy()
            # Should not raise
            policy.on_rejection(rejection)
        finally:
            t.close()

    def test_on_success_is_noop_by_default(self):
        """Base Policy.on_success() is a no-op."""
        t = Tract.open(":memory:")
        try:
            policy = AutoApproveDefaultPolicy()
            # Should not raise
            policy.on_success("some_result")
        finally:
            t.close()
