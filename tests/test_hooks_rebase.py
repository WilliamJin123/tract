"""Tests for rebase() hook integration (Phase 2).

Tests three-tier routing, PendingRebase lifecycle, exclude(), and hook
handler interaction for rebase operations.
"""

from __future__ import annotations

import pytest

from tract import Tract
from tract.hooks.rebase import PendingRebase
from tract.models.merge import RebaseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_divergent_tract() -> tuple[Tract, str, str]:
    """Create a tract with two divergent branches for rebase testing.

    Returns (tract, main_head, feature_head) where feature is current.
    Creates:
    - main: system + user + assistant
    - feature (from system): user2 + assistant2
    """
    t = Tract.open(":memory:")
    t.system("System prompt")
    main_after_system = t.head

    # Add more commits to main
    t.user("Main user message")
    t.assistant("Main assistant reply")
    main_head = t.head

    # Create feature branch from the system prompt
    t.branch("feature", source=main_after_system)
    t.switch("feature")
    t.user("Feature user message")
    t.assistant("Feature assistant reply")
    feature_head = t.head

    return t, main_head, feature_head


# ===========================================================================
# 1. No hook: auto-approves
# ===========================================================================


class TestRebaseAutoApprove:
    """rebase() auto-approves when no hook is registered."""

    def test_rebase_returns_rebase_result(self):
        """rebase() returns RebaseResult when no hook."""
        t, main_head, feature_head = _make_divergent_tract()
        try:
            result = t.rebase("main")
            assert isinstance(result, RebaseResult)
            assert result.new_head is not None
        finally:
            t.close()

    def test_rebase_noop_returns_result(self):
        """rebase() on same branch returns RebaseResult (no-op)."""
        t = Tract.open(":memory:")
        try:
            t.system("Hello")
            t.branch("feature")
            t.switch("feature")
            t.user("On feature")
            # feature is ahead of main, rebase onto main is noop
            result = t.rebase("main")
            assert isinstance(result, RebaseResult)
        finally:
            t.close()


# ===========================================================================
# 2. review=True returns PendingRebase
# ===========================================================================


class TestRebaseReview:
    """rebase(review=True) returns PendingRebase for manual inspection."""

    def test_review_returns_pending_rebase(self):
        """rebase(review=True) returns PendingRebase instance."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            assert isinstance(pending, PendingRebase)
            assert pending.status == "pending"
            assert pending.operation == "rebase"
        finally:
            t.close()

    def test_review_pending_has_replay_plan(self):
        """PendingRebase has replay_plan populated."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            assert isinstance(pending.replay_plan, list)
            assert len(pending.replay_plan) > 0
        finally:
            t.close()

    def test_review_pending_has_target_base(self):
        """PendingRebase has target_base set."""
        t, main_head, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            assert pending.target_base != ""
        finally:
            t.close()

    def test_review_pending_approve_executes(self):
        """Approving a PendingRebase executes the rebase."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            result = pending.approve()
            assert pending.status == "approved"
            assert isinstance(result, RebaseResult)
            assert result.new_head is not None
        finally:
            t.close()

    def test_review_pending_reject_preserves_history(self):
        """Rejecting a PendingRebase leaves history unchanged."""
        t, _, feature_head = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            pending.reject("Too risky")
            assert pending.status == "rejected"
            assert pending.rejection_reason == "Too risky"
            # HEAD should still be on feature at original position
            assert t.head == feature_head
        finally:
            t.close()

    def test_review_pending_has_triggered_by(self):
        """PendingRebase carries triggered_by."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase(
                "main", review=True, triggered_by="policy:auto_rebase"
            )
            assert pending.triggered_by == "policy:auto_rebase"
        finally:
            t.close()


# ===========================================================================
# 3. Hook fires handler
# ===========================================================================


class TestRebaseHook:
    """rebase() fires registered hook handler."""

    def test_hook_fires_on_rebase(self):
        """Registered rebase hook handler is called."""
        t, _, _ = _make_divergent_tract()
        try:
            hook_calls = []

            def handler(pending):
                hook_calls.append(pending)
                pending.approve()

            t.on("rebase", handler)
            result = t.rebase("main")

            assert len(hook_calls) == 1
            assert isinstance(hook_calls[0], PendingRebase)
            assert isinstance(result, RebaseResult)
        finally:
            t.close()

    def test_hook_reject_returns_pending(self):
        """Hook handler that rejects returns PendingRebase."""
        t, _, _ = _make_divergent_tract()
        try:
            def handler(pending):
                pending.reject("Not now")

            t.on("rebase", handler)
            result = t.rebase("main")

            assert isinstance(result, PendingRebase)
            assert result.status == "rejected"
        finally:
            t.close()


# ===========================================================================
# 4. PendingRebase.exclude()
# ===========================================================================


class TestRebaseExclude:
    """PendingRebase.exclude() removes commits from the replay plan."""

    def test_exclude_removes_from_replay_plan(self):
        """exclude() removes a commit from replay_plan."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            if len(pending.replay_plan) > 1:
                excluded = pending.replay_plan[0]
                original_count = len(pending.replay_plan)
                pending.exclude(excluded)
                assert len(pending.replay_plan) == original_count - 1
                assert excluded not in pending.replay_plan
        finally:
            t.close()

    def test_exclude_invalid_hash_raises(self):
        """exclude() with a hash not in the replay plan raises ValueError."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            with pytest.raises(ValueError, match="not in the replay plan"):
                pending.exclude("nonexistent_hash")
        finally:
            t.close()

    def test_exclude_then_approve_replays_remaining(self):
        """After exclude(), approve() replays only remaining commits."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            original_count = len(pending.replay_plan)
            if original_count > 1:
                pending.exclude(pending.replay_plan[0])
                result = pending.approve()
                # Should have replayed fewer commits
                assert len(result.replayed_commits) == original_count - 1
        finally:
            t.close()

    def test_exclude_after_approve_raises(self):
        """exclude() after approve() raises RuntimeError."""
        t, _, _ = _make_divergent_tract()
        try:
            pending = t.rebase("main", review=True)
            pending.approve()
            with pytest.raises(RuntimeError):
                pending.exclude("any_hash")
        finally:
            t.close()


# ===========================================================================
# 5. Recursion guard
# ===========================================================================


class TestRebaseRecursionGuard:
    """rebase() inside a hook handler auto-approves."""

    def test_rebase_inside_hook_auto_approves(self):
        """rebase() called inside another hook auto-approves."""
        t = Tract.open(":memory:")
        try:
            t.system("System")
            t.user("User")
            t.assistant("Assistant")

            rebase_hook_calls = []

            def compress_handler(pending):
                pending.approve()

            def rebase_handler(pending):
                rebase_hook_calls.append(pending)
                pending.approve()

            t.on("compress", compress_handler)
            t.on("rebase", rebase_handler)

            # Compress triggers compress hook, not rebase
            t.compress(content="Summary")

            # rebase hook should not have been triggered by compress
            assert len(rebase_hook_calls) == 0
        finally:
            t.close()
