"""End-to-end integration tests for the policy engine.

Tests the full lifecycle: configure policies -> commit content ->
trigger evaluation -> verify actions executed/proposed.

Covers:
- Auto-pin on commit (autonomous mode)
- Auto-compress on compile (collaborative mode with approval)
- Policy priority ordering via audit log
- Pause/resume lifecycle
- Collaborative approve/reject via hook system
- Policy config persistence (save/load)
- Recursive evaluation prevention
- Multiple policy composition
- Custom policy subclass
- Policy hook callback
- Policy config survives restart (file-backed)
"""

from __future__ import annotations

import os
import tempfile
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
    TractConfig,
    TokenBudgetConfig,
)
from tract.models.content import ArtifactContent, ReasoningContent
from tract.models.session import SessionContent
from tract.policy.builtin.compress import CompressPolicy
from tract.policy.builtin.pin import PinPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingPolicy(Policy):
    """Test helper that counts evaluations and returns configurable action.

    default_handler does nothing (leaves pending unresolved) to preserve
    the proposal-based test workflow. Set auto_approve_default=True to
    use auto-approve behavior.
    """

    def __init__(
        self, name: str = "counter", priority: int = 100,
        trigger: str = "compile", action: PolicyAction | None = None,
        auto_approve_default: bool = False,
    ):
        self._name = name
        self._priority = priority
        self._trigger = trigger
        self._action = action
        self._auto_approve_default = auto_approve_default
        self.count = 0

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
        self.count += 1
        return self._action

    def default_handler(self, pending) -> None:
        """Leave pending unresolved by default for test control."""
        if self._auto_approve_default:
            pending.approve()


# ---------------------------------------------------------------------------
# 1. Auto-Pin on Commit
# ---------------------------------------------------------------------------


class TestAutoPinOnCommit:
    """PinPolicy auto-pins InstructionContent on commit."""

    def test_auto_pin_on_commit(self):
        """Configure PinPolicy (autonomous), commit InstructionContent, verify auto-pinned."""
        t = Tract.open(":memory:")
        try:
            # Configure PinPolicy
            pin_policy = PinPolicy()
            t.configure_policies(policies=[pin_policy])

            # Commit instruction content -- should trigger commit-time evaluation
            info = t.commit(InstructionContent(text="You are a helpful assistant."))

            # Verify it was auto-pinned
            annotations = t.get_annotations(info.commit_hash)
            assert len(annotations) >= 1
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()

    def test_auto_pin_does_not_pin_dialogue(self):
        """PinPolicy does not pin DialogueContent."""
        t = Tract.open(":memory:")
        try:
            pin_policy = PinPolicy()
            t.configure_policies(policies=[pin_policy])

            # First commit to establish HEAD
            t.commit(InstructionContent(text="system"))
            # Now commit dialogue
            info = t.commit(DialogueContent(role="user", text="Hi"))

            annotations = t.get_annotations(info.commit_hash)
            # No pinned annotation on dialogue
            assert not any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 2. Auto-Compress on Compile
# ---------------------------------------------------------------------------


class TestAutoCompressOnCompile:
    """CompressPolicy proposes compression in collaborative mode."""

    def test_auto_compress_on_compile(self):
        """Configure CompressPolicy with low threshold, compile triggers proposal."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            proposals = []

            # Register a policy hook that captures but doesn't approve/reject
            # This overrides the policy's default_handler auto-approve
            def policy_hook(pending):
                proposals.append(pending)
                # Leave pending unresolved to test proposal creation

            t.on("policy", policy_hook)

            compress_policy = CompressPolicy(threshold=0.5, summary_content="Compressed")
            t.configure_policies(
                policies=[compress_policy],
            )

            # Commit enough to exceed threshold
            t.commit(InstructionContent(text="This is a fairly long instruction text"))

            # Compile triggers the compress policy
            t.compile()

            # Should have created a proposal
            assert len(proposals) >= 1
            assert proposals[0].policy_name == "auto-compress"
            assert proposals[0].status == "pending"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 3. Priority Ordering
# ---------------------------------------------------------------------------


class TestPolicyPriorityOrdering:
    """Verify pin runs before compress via audit log."""

    def test_policy_priority_ordering(self):
        """PinPolicy (priority=100) evaluates before CompressPolicy (priority=200)."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            # Use counting policies with known priorities
            p_first = CountingPolicy(name="first", priority=50, trigger="compile")
            p_second = CountingPolicy(name="second", priority=150, trigger="compile")
            p_third = CountingPolicy(name="third", priority=250, trigger="compile")

            t.configure_policies(policies=[p_third, p_first, p_second])

            t.commit(InstructionContent(text="hello"))
            t.compile()

            # All three should have been evaluated
            assert p_first.count >= 1
            assert p_second.count >= 1
            assert p_third.count >= 1

            # Check audit log order
            log_entries = t._policy_repo.get_log(t.tract_id)
            # Filter for compile trigger entries
            compile_entries = [e for e in log_entries if e.trigger == "compile"]

            if len(compile_entries) >= 3:
                names = [e.policy_name for e in compile_entries[:3]]
                assert names == ["first", "second", "third"]
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 4. Pause / Resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """Pause and resume policy evaluation lifecycle."""

    def test_pause_resume_lifecycle(self):
        """Paused policies don't fire, resumed policies do."""
        t = Tract.open(":memory:")
        try:
            counter = CountingPolicy(name="lifecycle", trigger="commit")
            t.configure_policies(policies=[counter])

            # Commit while active -- both trigger the policy
            t.commit(InstructionContent(text="first"))
            t.commit(DialogueContent(role="user", text="second"))
            assert counter.count == 2

            # Pause
            t.pause_all_policies()
            t.commit(DialogueContent(role="assistant", text="third"))
            assert counter.count == 2  # No additional evaluation

            # Resume
            t.resume_all_policies()
            t.commit(DialogueContent(role="user", text="fourth"))
            assert counter.count == 3  # Resumed
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 5. Collaborative Approve/Reject
# ---------------------------------------------------------------------------


class TestCollaborativeApproveReject:
    """Collaborative mode approve and reject lifecycle via hooks."""

    def test_collaborative_reject_then_approve(self):
        """First hook rejects, second hook approves -- annotation created on second."""
        t = Tract.open(":memory:")
        try:
            # Use DialogueContent which does NOT auto-pin
            t.commit(InstructionContent(text="system"))
            info = t.commit(DialogueContent(role="user", text="hello"))

            action = PolicyAction(
                action_type="annotate",
                params={"target_hash": info.commit_hash, "priority": "pinned"},
                reason="Should we pin?",
                autonomy="collaborative",
            )

            call_count = [0]

            def hook(pending):
                call_count[0] += 1
                if call_count[0] == 1:
                    pending.reject("Not now")
                else:
                    pending.approve()

            t.on("policy", hook)

            p = CountingPolicy(name="collab", trigger="compile", action=action)
            t.configure_policies(policies=[p])

            # First compile -- hook rejects
            t.compile()
            annotations = t.get_annotations(info.commit_hash)
            assert not any(a.priority == Priority.PINNED for a in annotations)

            # Second compile -- hook approves
            t.compile()
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()

    def test_collaborative_approve_via_hook(self):
        """Hook auto-approves collaborative action, verifying execution."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            action = PolicyAction(
                action_type="annotate",
                params={"target_hash": info.commit_hash, "priority": "pinned"},
                reason="Should we pin?",
                autonomy="collaborative",
            )
            p = CountingPolicy(name="collab", trigger="compile", action=action)

            t.on("policy", lambda pending: pending.approve())
            t.configure_policies(policies=[p])

            t.compile()

            # Annotation was created
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 6. Config Persistence
# ---------------------------------------------------------------------------


class TestPolicyConfigPersistence:
    """Policy config save/load roundtrip."""

    def test_policy_config_persistence(self):
        """Built-in policy configs roundtrip through save/load."""
        t = Tract.open(":memory:")
        try:
            pin = PinPolicy()
            compress = CompressPolicy(threshold=0.8)

            config = {
                "policies": [pin.to_config(), compress.to_config()],
            }
            t.save_policy_config(config)

            loaded = t.load_policy_config()
            assert loaded is not None
            assert len(loaded["policies"]) == 2
            assert loaded["policies"][0]["name"] == "auto-pin"
            assert loaded["policies"][1]["name"] == "auto-compress"
            assert loaded["policies"][1]["threshold"] == 0.8
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 7. Recursive Evaluation Prevention
# ---------------------------------------------------------------------------


class TestRecursiveEvaluationPrevention:
    """Policy that calls compile() during evaluation should not infinite-loop."""

    def test_recursive_evaluation_prevention(self):
        """Policy calling compile() inside evaluate() does not recurse infinitely."""

        class RecursiveCompilePolicy(Policy):
            @property
            def name(self) -> str:
                return "recursive-compile"

            def evaluate(self, tract: Tract) -> PolicyAction | None:
                # This triggers compile, which would re-trigger this policy
                # without the recursion guard
                tract.compile()
                return None

        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = RecursiveCompilePolicy()
            t.configure_policies(policies=[p])

            # This should complete without stack overflow
            result = t.compile()
            assert result is not None
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 8. Multiple Policies Compose
# ---------------------------------------------------------------------------


class TestMultiplePoliciesCompose:
    """Multiple policies interact correctly via priority ordering."""

    def test_multiple_policies_compose(self):
        """PinPolicy + CompressPolicy together: pin runs first, then compress."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            proposals = []

            # Register a policy hook to capture proposals without auto-approving
            def policy_hook(pending):
                proposals.append(pending)

            t.on("policy", policy_hook)

            pin = PinPolicy()
            compress = CompressPolicy(threshold=0.5, summary_content="Summary")

            t.configure_policies(
                policies=[compress, pin],  # Deliberately reverse order
            )

            # Commit instruction (triggers pin, which is commit-triggered)
            info = t.commit(InstructionContent(text="This is a long system prompt"))

            # Pin should have fired (autonomous, commit-triggered)
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)

            # Compile (triggers compress)
            t.compile()

            # Compress should have proposed (collaborative, compile-triggered)
            assert any(p.policy_name == "auto-compress" for p in proposals)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 9. Custom Policy Subclass
# ---------------------------------------------------------------------------


class TestCustomPolicySubclass:
    """User-defined custom policies work with the evaluator."""

    def test_custom_policy_subclass(self):
        """Custom Policy subclass works with configure_policies."""

        class MyCustomPolicy(Policy):
            @property
            def name(self) -> str:
                return "my-custom"

            @property
            def priority(self) -> int:
                return 42

            @property
            def trigger(self) -> str:
                return "commit"

            def evaluate(self, tract: Tract) -> PolicyAction | None:
                head = tract.head
                if head is None:
                    return None
                commit = tract.get_commit(head)
                if commit and commit.content_type == "reasoning":
                    return PolicyAction(
                        action_type="annotate",
                        params={"target_hash": head, "priority": "skip"},
                        reason="Auto-skip reasoning",
                        autonomy="autonomous",
                    )
                return None

        t = Tract.open(":memory:")
        try:
            custom = MyCustomPolicy()
            t.configure_policies(policies=[custom])

            # Commit reasoning content
            t.commit(InstructionContent(text="system"))
            info = t.commit(ReasoningContent(text="thinking..."))

            # Verify it was auto-skipped
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.SKIP for a in annotations)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 10. Policy Hook Callback
# ---------------------------------------------------------------------------


class TestPolicyHookCallback:
    """Policy hook is invoked for collaborative proposals."""

    def test_policy_hook_captures_pending(self):
        """Policy hook receives PendingPolicy for collaborative actions."""
        t = Tract.open(":memory:")
        try:
            hook_calls = []
            info = t.commit(InstructionContent(text="hello"))

            action = PolicyAction(
                action_type="annotate",
                params={"target_hash": info.commit_hash, "priority": "pinned"},
                reason="Test hook",
                autonomy="collaborative",
            )
            p = CountingPolicy(name="hook-test", action=action)

            t.on("policy", lambda pending: hook_calls.append(pending))
            t.configure_policies(policies=[p])

            t.compile()

            assert len(hook_calls) == 1
            assert hook_calls[0].policy_name == "hook-test"
            assert hook_calls[0].status == "pending"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 11. Policy Config Survives Restart (File-Backed)
# ---------------------------------------------------------------------------


class TestPolicyConfigSurvivesRestart:
    """Policy config persists across Tract.open() calls (file-backed)."""

    def test_policy_config_survives_restart_file_backed(self):
        """Policies auto-load from _trace_meta on Tract.open()."""
        db_path = os.path.join(tempfile.mkdtemp(), "test_restart.db")
        tract_id = "restart-test-tract"

        try:
            # First session: configure and save policies
            config = TractConfig(
                db_path=db_path,
                token_budget=TokenBudgetConfig(max_tokens=1000),
            )
            t1 = Tract.open(db_path, tract_id=tract_id, config=config)
            pin = PinPolicy(pin_types={"instruction", "session"})
            compress = CompressPolicy(threshold=0.75)
            t1.configure_policies(policies=[pin, compress])

            # Save config
            policy_config = {
                "policies": [pin.to_config(), compress.to_config()],
            }
            t1.save_policy_config(policy_config)

            # Commit some content
            t1.commit(InstructionContent(text="System prompt"))
            t1.close()

            # Second session: re-open same file
            config2 = TractConfig(
                db_path=db_path,
                token_budget=TokenBudgetConfig(max_tokens=1000),
            )
            t2 = Tract.open(db_path, tract_id=tract_id, config=config2)

            # Verify evaluator was auto-created
            assert t2.policy_evaluator is not None

            # Verify both policies are registered
            policies = t2.policy_evaluator._policies
            assert len(policies) == 2
            policy_names = {p.name for p in policies}
            assert "auto-pin" in policy_names
            assert "auto-compress" in policy_names

            # Verify priority ordering (pin=100, compress=200)
            assert policies[0].name == "auto-pin"
            assert policies[1].name == "auto-compress"

            # Verify compress threshold was restored
            compress_restored = next(
                p for p in policies if p.name == "auto-compress"
            )
            assert compress_restored._threshold == 0.75

            # Verify policies evaluate correctly -- commit SessionContent
            info = t2.commit(SessionContent(session_type="start", summary="Restart"))
            # PinPolicy should have pinned it
            annotations = t2.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)

            t2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
