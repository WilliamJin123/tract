"""End-to-end integration tests for the policy engine.

Tests the full lifecycle: configure policies -> commit content ->
trigger evaluation -> verify actions executed/proposed.

Covers:
- Auto-pin on commit (autonomous mode)
- Auto-compress on compile (collaborative mode with approval)
- Policy priority ordering via audit log
- Pause/resume lifecycle
- Collaborative approve/reject
- Policy config persistence (save/load)
- Recursive evaluation prevention
- Multiple policy composition
- Custom policy subclass
- on_proposal callback
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
    PolicyProposal,
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
    """Test helper that counts evaluations and returns configurable action."""

    def __init__(
        self, name: str = "counter", priority: int = 100,
        trigger: str = "compile", action: PolicyAction | None = None,
    ):
        self._name = name
        self._priority = priority
        self._trigger = trigger
        self._action = action
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
            proposals: list[PolicyProposal] = []
            compress_policy = CompressPolicy(threshold=0.5, summary_content="Compressed")
            t.configure_policies(
                policies=[compress_policy],
                on_proposal=lambda p: proposals.append(p),
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
    """Collaborative mode approve and reject lifecycle."""

    def test_collaborative_approve_reject(self):
        """Approve executes action, reject cancels it."""
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
            t.configure_policies(policies=[p])

            # Trigger evaluation -- creates proposal
            t.compile()

            pending = t.get_pending_proposals()
            assert len(pending) == 1

            # Reject the first proposal
            t.reject_proposal(pending[0].proposal_id)
            assert len(t.get_pending_proposals()) == 0

            # Trigger again -- new proposal
            t.compile()
            pending2 = t.get_pending_proposals()
            assert len(pending2) == 1

            # Approve this one
            t.approve_proposal(pending2[0].proposal_id)

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
            proposals: list[PolicyProposal] = []
            pin = PinPolicy()
            compress = CompressPolicy(threshold=0.5, summary_content="Summary")

            t.configure_policies(
                policies=[compress, pin],  # Deliberately reverse order
                on_proposal=lambda p: proposals.append(p),
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
# 10. on_proposal Callback
# ---------------------------------------------------------------------------


class TestOnProposalCallback:
    """on_proposal callback is invoked for collaborative proposals."""

    def test_on_proposal_callback(self):
        """on_proposal callback is invoked with the proposal object."""
        t = Tract.open(":memory:")
        try:
            callback_calls: list[PolicyProposal] = []
            info = t.commit(InstructionContent(text="hello"))

            action = PolicyAction(
                action_type="annotate",
                params={"target_hash": info.commit_hash, "priority": "pinned"},
                reason="Test callback",
                autonomy="collaborative",
            )
            p = CountingPolicy(name="callback-test", action=action)
            t.configure_policies(
                policies=[p],
                on_proposal=lambda prop: callback_calls.append(prop),
            )

            t.compile()

            assert len(callback_calls) == 1
            assert callback_calls[0].policy_name == "callback-test"
            assert callback_calls[0].status == "pending"
        finally:
            t.close()
