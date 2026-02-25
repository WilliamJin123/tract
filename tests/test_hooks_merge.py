"""Tests for merge() conflict hook integration (Phase 2).

Tests three-tier routing for the conflict path, PendingMerge lifecycle,
edit_resolution(), and hook handler interaction. Fast-forward and clean
merges proceed without hooks.
"""

from __future__ import annotations

import pytest

from tract import Tract, DialogueContent
from tract.hooks.merge import PendingMerge
from tract.models.commit import CommitOperation
from tract.models.merge import MergeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conflict_tract() -> tuple[Tract, str]:
    """Create a tract with two branches that will produce a conflict.

    Creates a both_edit conflict: main and feature both EDIT the
    same base commit.

    Returns (tract, source_branch_name).
    """
    from tract import InstructionContent

    t = Tract.open(":memory:")

    # Base commit on main
    base = t.commit(InstructionContent(text="original"))
    base_hash = base.commit_hash

    # Create feature branch (at current HEAD = base)
    t.branch("feature")

    # Feature branch: EDIT the base commit
    t.switch("feature")
    t.commit(
        DialogueContent(role="assistant", text="feature edit"),
        operation=CommitOperation.EDIT,
        edit_target=base_hash,
    )

    # Main branch: EDIT the same base commit
    t.switch("main")
    t.commit(
        DialogueContent(role="assistant", text="main edit"),
        operation=CommitOperation.EDIT,
        edit_target=base_hash,
    )

    return t, "feature"


def _make_clean_merge_tract() -> tuple[Tract, str]:
    """Create a tract where merge produces no conflicts (clean merge)."""
    t = Tract.open(":memory:")
    t.system("System prompt")
    base = t.head

    t.user("Main message")

    t.branch("feature", source=base)
    t.switch("feature")
    t.user("Feature message")

    t.switch("main")
    return t, "feature"


def _make_ff_merge_tract() -> tuple[Tract, str]:
    """Create a tract where merge is fast-forward."""
    t = Tract.open(":memory:")
    t.system("System prompt")
    base = t.head

    t.branch("feature")
    t.switch("feature")
    t.user("Feature message 1")
    t.assistant("Feature reply 1")

    t.switch("main")
    return t, "feature"


# ===========================================================================
# 1. Fast-forward and clean merges: no hook
# ===========================================================================


class TestMergeNoHook:
    """Fast-forward and clean merges proceed without hook interception."""

    def test_ff_merge_returns_merge_result(self):
        """Fast-forward merge returns MergeResult directly."""
        t, source = _make_ff_merge_tract()
        try:
            hook_calls = []
            t.on("merge", lambda p: hook_calls.append(p))

            result = t.merge(source)
            assert isinstance(result, MergeResult)
            assert result.merge_type == "fast_forward"
            # Hook should NOT have fired for ff merge
            assert len(hook_calls) == 0
        finally:
            t.close()

    def test_clean_merge_returns_merge_result(self):
        """Clean merge returns MergeResult directly."""
        t, source = _make_clean_merge_tract()
        try:
            hook_calls = []
            t.on("merge", lambda p: hook_calls.append(p))

            result = t.merge(source)
            assert isinstance(result, MergeResult)
            assert result.merge_type == "clean"
            # Hook should NOT have fired for clean merge
            assert len(hook_calls) == 0
        finally:
            t.close()


# ===========================================================================
# 2. Conflict path: review=True returns PendingMerge
# ===========================================================================


class TestMergeConflictReview:
    """merge(review=True) returns PendingMerge for conflict merges."""

    def test_conflict_review_returns_pending(self):
        """Conflict merge with review=True returns PendingMerge."""
        t, source = _make_conflict_tract()
        try:
            # Need a resolver to generate resolutions
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test resolution",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            assert isinstance(pending, PendingMerge)
            assert pending.status == "pending"
            assert pending.operation == "merge"
            assert pending.source_branch == source
        finally:
            t.close()

    def test_conflict_review_has_resolutions(self):
        """PendingMerge from review has resolutions populated."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test resolution",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            assert isinstance(pending.resolutions, dict)
            assert len(pending.resolutions) > 0
        finally:
            t.close()

    def test_conflict_review_approve_commits(self):
        """Approving PendingMerge commits the merge."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test resolution",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            result = pending.approve()
            assert pending.status == "approved"
            assert isinstance(result, MergeResult)
            assert result.committed is True
        finally:
            t.close()

    def test_conflict_review_reject(self):
        """Rejecting PendingMerge leaves branches unchanged."""
        t, source = _make_conflict_tract()
        try:
            original_head = t.head

            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test resolution",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            pending.reject("Bad merge")
            assert pending.status == "rejected"
            # HEAD should not have changed
            assert t.head == original_head
        finally:
            t.close()


# ===========================================================================
# 3. Hook fires for conflict path
# ===========================================================================


class TestMergeConflictHook:
    """merge() fires hook handler for conflict merges."""

    def test_hook_fires_on_conflict(self):
        """Registered merge hook handler fires for conflict merges."""
        t, source = _make_conflict_tract()
        try:
            hook_calls = []

            def handler(pending):
                hook_calls.append(pending)
                pending.approve()

            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test resolution",
                )

            t.on("merge", handler)
            result = t.merge(source, resolver=resolver)

            assert len(hook_calls) == 1
            assert isinstance(hook_calls[0], PendingMerge)
            assert isinstance(result, MergeResult)
            assert result.committed is True
        finally:
            t.close()

    def test_hook_reject_conflict(self):
        """Hook that rejects conflict merge returns PendingMerge."""
        t, source = _make_conflict_tract()
        try:
            def handler(pending):
                pending.reject("Resolutions need work")

            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test resolution",
                )

            t.on("merge", handler)
            result = t.merge(source, resolver=resolver)

            assert isinstance(result, PendingMerge)
            assert result.status == "rejected"
        finally:
            t.close()


# ===========================================================================
# 4. PendingMerge.edit_resolution()
# ===========================================================================


class TestMergeEditResolution:
    """PendingMerge.edit_resolution() modifies conflict resolutions."""

    def test_edit_resolution_updates_content(self):
        """edit_resolution() updates the resolution for a conflict key."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Original resolution",
                    reasoning="Test",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            if pending.resolutions:
                key = next(iter(pending.resolutions))
                pending.edit_resolution(key, "Better resolution")
                assert pending.resolutions[key] == "Better resolution"
        finally:
            t.close()

    def test_edit_resolution_invalid_key_raises(self):
        """edit_resolution() with invalid key raises KeyError."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Resolution",
                    reasoning="Test",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            with pytest.raises(KeyError):
                pending.edit_resolution("nonexistent_key", "content")
        finally:
            t.close()

    def test_edit_resolution_after_approve_raises(self):
        """edit_resolution() after approve() raises RuntimeError."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Resolution",
                    reasoning="Test",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            pending.approve()
            if pending.resolutions:
                key = next(iter(pending.resolutions))
                with pytest.raises(RuntimeError):
                    pending.edit_resolution(key, "Too late")
        finally:
            t.close()


# ===========================================================================
# 5. Backward compatibility: auto_commit still works
# ===========================================================================


class TestMergeBackwardCompat:
    """auto_commit parameter still works for backward compatibility."""

    def test_auto_commit_true_with_resolver(self):
        """auto_commit=True with resolver auto-commits resolved conflicts."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test resolution",
                )

            result = t.merge(source, resolver=resolver, auto_commit=True)
            assert isinstance(result, MergeResult)
            assert result.committed is True
        finally:
            t.close()
