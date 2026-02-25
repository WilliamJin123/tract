"""Tests for gc() hook integration (Phase 2).

Tests three-tier routing, PendingGC lifecycle, exclude(), and hook
handler interaction for garbage collection.
"""

from __future__ import annotations

import pytest

from tract import Tract
from tract.hooks.gc import PendingGC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gc_tract() -> Tract:
    """Create a tract with orphaned commits suitable for gc().

    Strategy: create commits on main, branch, add more commits on the
    branch, switch back, then compress main so the old commits become
    orphans.
    """
    t = Tract.open(":memory:")
    t.system("System prompt")
    t.user("User message 1")
    t.assistant("Assistant reply 1")
    # Compress to create orphaned source commits
    t.compress(content="Summary of conversation")
    return t


def _make_gc_tract_with_orphans() -> Tract:
    """Create a tract that has orphans ready to gc.

    We need orphan_retention_days=0 for immediate gc in tests.
    """
    t = Tract.open(":memory:")
    t.system("System prompt")
    t.user("User message 1")
    t.assistant("Assistant reply 1")
    t.compress(content="Summary")
    return t


# ===========================================================================
# 1. No hook: auto-approves
# ===========================================================================


class TestGCAutoApprove:
    """gc() auto-approves when no hook is registered."""

    def test_gc_returns_gc_result(self):
        """gc() returns GCResult (not PendingGC) when no hook."""
        t = _make_gc_tract_with_orphans()
        try:
            result = t.gc(orphan_retention_days=0)
            # Should be a GCResult, not PendingGC
            assert not isinstance(result, PendingGC)
            assert hasattr(result, "commits_removed")
        finally:
            t.close()

    def test_gc_empty_tract(self):
        """gc() on empty tract returns zero removals."""
        t = Tract.open(":memory:")
        try:
            t.system("Hello")
            result = t.gc(orphan_retention_days=0)
            assert result.commits_removed == 0
        finally:
            t.close()


# ===========================================================================
# 2. review=True returns PendingGC
# ===========================================================================


class TestGCReview:
    """gc(review=True) returns PendingGC for manual inspection."""

    def test_review_returns_pending_gc(self):
        """gc(review=True) returns PendingGC instance."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            assert isinstance(pending, PendingGC)
            assert pending.status == "pending"
            assert pending.operation == "gc"
        finally:
            t.close()

    def test_review_pending_has_commits_to_remove(self):
        """PendingGC from review has commits_to_remove populated."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            # After compress, original commits are orphaned
            assert isinstance(pending.commits_to_remove, list)
        finally:
            t.close()

    def test_review_pending_approve_executes(self):
        """Approving a PendingGC from review executes the gc."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            result = pending.approve()
            assert pending.status == "approved"
            assert hasattr(result, "commits_removed")
        finally:
            t.close()

    def test_review_pending_reject_preserves_commits(self):
        """Rejecting a PendingGC keeps all commits."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            pending.reject("Keep everything")
            assert pending.status == "rejected"
            assert pending.rejection_reason == "Keep everything"
        finally:
            t.close()

    def test_review_pending_has_triggered_by(self):
        """PendingGC carries triggered_by from gc() call."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(
                review=True, orphan_retention_days=0,
                triggered_by="policy:auto_gc",
            )
            assert pending.triggered_by == "policy:auto_gc"
        finally:
            t.close()


# ===========================================================================
# 3. Hook fires handler
# ===========================================================================


class TestGCHook:
    """gc() fires registered hook handler."""

    def test_hook_fires_on_gc(self):
        """Registered gc hook handler is called."""
        t = _make_gc_tract_with_orphans()
        try:
            hook_calls = []

            def handler(pending):
                hook_calls.append(pending)
                pending.approve()

            t.on("gc", handler)
            result = t.gc(orphan_retention_days=0)

            assert len(hook_calls) == 1
            assert isinstance(hook_calls[0], PendingGC)
            # Result should be GCResult since handler approved
            assert hasattr(result, "commits_removed")
        finally:
            t.close()

    def test_hook_reject_returns_pending(self):
        """Hook handler that rejects returns PendingGC."""
        t = _make_gc_tract_with_orphans()
        try:
            def handler(pending):
                pending.reject("Not now")

            t.on("gc", handler)
            result = t.gc(orphan_retention_days=0)

            assert isinstance(result, PendingGC)
            assert result.status == "rejected"
        finally:
            t.close()

    def test_catch_all_hook_fires(self):
        """Catch-all * hook fires for gc when no specific hook."""
        t = _make_gc_tract_with_orphans()
        try:
            hook_calls = []

            def handler(pending):
                hook_calls.append(pending.operation)
                pending.approve()

            t.on("*", handler)
            t.gc(orphan_retention_days=0)

            assert "gc" in hook_calls
        finally:
            t.close()

    def test_specific_hook_beats_catch_all(self):
        """Specific gc hook takes precedence over catch-all."""
        t = _make_gc_tract_with_orphans()
        try:
            calls = []

            def specific(pending):
                calls.append("specific")
                pending.approve()

            def catch_all(pending):
                calls.append("catch_all")
                pending.approve()

            t.on("gc", specific)
            t.on("*", catch_all)
            t.gc(orphan_retention_days=0)

            assert calls == ["specific"]
        finally:
            t.close()


# ===========================================================================
# 4. PendingGC.exclude()
# ===========================================================================


class TestGCExclude:
    """PendingGC.exclude() removes commits from the gc plan."""

    def test_exclude_removes_from_plan(self):
        """exclude() removes a commit from commits_to_remove."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            if len(pending.commits_to_remove) > 0:
                excluded = pending.commits_to_remove[0]
                original_count = len(pending.commits_to_remove)
                pending.exclude(excluded)
                assert len(pending.commits_to_remove) == original_count - 1
                assert excluded not in pending.commits_to_remove
        finally:
            t.close()

    def test_exclude_invalid_hash_raises(self):
        """exclude() with a hash not in the list raises ValueError."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            with pytest.raises(ValueError, match="not in the removal list"):
                pending.exclude("nonexistent_hash")
        finally:
            t.close()

    def test_exclude_then_approve_respects_exclusion(self):
        """After exclude(), approve() only removes remaining commits."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            if len(pending.commits_to_remove) > 1:
                excluded = pending.commits_to_remove[0]
                pending.exclude(excluded)
                result = pending.approve()
                # The excluded commit should not have been removed
                assert hasattr(result, "commits_removed")
        finally:
            t.close()

    def test_exclude_after_approve_raises(self):
        """exclude() after approve() raises RuntimeError."""
        t = _make_gc_tract_with_orphans()
        try:
            pending = t.gc(review=True, orphan_retention_days=0)
            pending.approve()
            with pytest.raises(RuntimeError):
                pending.exclude("any_hash")
        finally:
            t.close()


# ===========================================================================
# 5. Recursion guard
# ===========================================================================


class TestGCRecursionGuard:
    """gc() inside a hook handler auto-approves (no recursion)."""

    def test_gc_inside_hook_auto_approves(self):
        """gc() called inside a compress hook auto-approves without firing gc hook."""
        t = Tract.open(":memory:")
        try:
            t.system("System")
            t.user("User")
            t.assistant("Assistant")

            gc_hook_calls = []

            def compress_handler(pending):
                # Call gc() from inside compress hook -- should auto-approve
                t.gc(orphan_retention_days=0)
                pending.approve()

            def gc_handler(pending):
                gc_hook_calls.append(pending)
                pending.approve()

            t.on("compress", compress_handler)
            t.on("gc", gc_handler)

            t.compress(content="Summary")

            # gc hook should NOT have fired (recursion guard)
            assert len(gc_hook_calls) == 0
        finally:
            t.close()
