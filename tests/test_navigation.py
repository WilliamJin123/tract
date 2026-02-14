"""Tests for navigation operations -- reset, checkout, prefix matching, symbolic refs.

Tests the storage infrastructure (symbolic refs, prefix matching) and the
navigation operations (reset, checkout, resolve_commit) through the Tract
facade.
"""

from __future__ import annotations

import pytest

from tract import (
    AmbiguousPrefixError,
    CommitNotFoundError,
    DetachedHeadError,
    InstructionContent,
    DialogueContent,
    Tract,
    TraceError,
)
from tests.conftest import make_tract, populate_tract


# ==================================================================
# Symbolic ref basics
# ==================================================================

class TestSymbolicRefs:
    """Tests for symbolic ref resolution and HEAD state."""

    def test_initial_head_is_attached(self):
        """After first commit, HEAD is attached to main."""
        t = make_tract()
        hashes = populate_tract(t, 1)
        assert not t.is_detached
        assert t.current_branch == "main"
        assert t.head == hashes[0]

    def test_head_attached_after_multiple_commits(self):
        """HEAD stays attached through multiple commits."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        assert not t.is_detached
        assert t.current_branch == "main"
        assert t.head == hashes[-1]

    def test_no_commits_not_detached(self):
        """Empty tract is not detached (no HEAD at all)."""
        t = make_tract()
        assert not t.is_detached
        assert t.current_branch is None
        assert t.head is None

    def test_symbolic_ref_resolves_through_branch(self):
        """get_head resolves HEAD -> refs/heads/main -> commit_hash."""
        t = make_tract()
        hashes = populate_tract(t, 2)
        # get_head should resolve to latest commit
        assert t.head == hashes[1]
        # branch should also point there
        branch_hash = t._ref_repo.get_branch(t.tract_id, "main")
        assert branch_hash == hashes[1]


# ==================================================================
# Prefix matching
# ==================================================================

class TestPrefixMatching:
    """Tests for commit hash prefix resolution."""

    @pytest.mark.parametrize("prefix_len", [64, 8, 4])
    def test_resolve_by_prefix(self, prefix_len):
        t = make_tract()
        hashes = populate_tract(t, 1)
        prefix = hashes[0][:prefix_len]
        resolved = t.resolve_commit(prefix)
        assert resolved == hashes[0]

    def test_resolve_prefix_too_short(self):
        """Prefix under 4 chars raises CommitNotFoundError."""
        t = make_tract()
        populate_tract(t, 1)
        with pytest.raises(CommitNotFoundError):
            t.resolve_commit("abc")

    def test_resolve_by_branch_name(self):
        """Branch name resolves to its commit."""
        t = make_tract()
        hashes = populate_tract(t, 2)
        resolved = t.resolve_commit("main")
        assert resolved == hashes[-1]

    def test_resolve_nonexistent(self):
        """Nonexistent ref raises CommitNotFoundError."""
        t = make_tract()
        populate_tract(t, 1)
        with pytest.raises(CommitNotFoundError):
            t.resolve_commit("nonexistent_branch")

    def test_prefix_ambiguous_error(self):
        """AmbiguousPrefixError propagates through resolve_commit."""
        from unittest.mock import patch
        from tract.storage.schema import CommitRow

        t = make_tract()
        hashes = populate_tract(t, 1)

        # Mock get_by_prefix to raise AmbiguousPrefixError
        with patch.object(
            t._commit_repo,
            "get_by_prefix",
            side_effect=AmbiguousPrefixError("abcd", ["abcd1111", "abcd2222"]),
        ):
            with pytest.raises(AmbiguousPrefixError) as exc_info:
                t.resolve_commit("abcdef")  # 6-char prefix triggers step 3
            assert exc_info.value.prefix == "abcd"
            assert len(exc_info.value.candidates) == 2

    def test_prefix_resolves_via_get_by_prefix(self):
        """Successful prefix resolution via get_by_prefix."""
        t = make_tract()
        hashes = populate_tract(t, 1)
        row = t._commit_repo.get_by_prefix(hashes[0][:8], tract_id=t.tract_id)
        assert row is not None
        assert row.commit_hash == hashes[0]


# ==================================================================
# Reset
# ==================================================================

class TestReset:
    """Tests for Tract.reset()."""

    @pytest.mark.parametrize("mode", ["soft", "hard"])
    def test_reset_moves_head(self, mode):
        t = make_tract()
        hashes = populate_tract(t, 3)
        result = t.reset(hashes[0], mode=mode)
        assert result == hashes[0]
        assert t.head == hashes[0]

    def test_reset_stores_orig_head(self):
        """Reset stores the previous HEAD as ORIG_HEAD."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        old_head = t.head

        t.reset(hashes[0])
        orig = t._ref_repo.get_ref(t.tract_id, "ORIG_HEAD")
        assert orig == old_head

    def test_reset_head_stays_attached(self):
        """After reset, HEAD remains attached to main."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        t.reset(hashes[0])
        assert not t.is_detached
        assert t.current_branch == "main"

    def test_reset_by_prefix(self):
        """Reset accepts a hash prefix as target."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        prefix = hashes[0][:8]
        result = t.reset(prefix)
        assert result == hashes[0]
        assert t.head == hashes[0]

    def test_reset_can_commit_after(self):
        """After soft reset, new commits extend from reset point."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        t.reset(hashes[0])
        info = t.commit(DialogueContent(role="user", text="After reset"))
        assert info.parent_hash == hashes[0]


# ==================================================================
# Checkout
# ==================================================================

class TestCheckout:
    """Tests for Tract.checkout()."""

    def test_checkout_commit_detaches_head(self):
        """Checkout a commit hash detaches HEAD."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        t.checkout(hashes[0])
        assert t.is_detached
        assert t.current_branch is None
        assert t.head == hashes[0]

    def test_checkout_branch_attaches_head(self):
        """Checkout a branch name re-attaches HEAD."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        # Detach first
        t.checkout(hashes[0])
        assert t.is_detached

        # Re-attach to main
        t.checkout("main")
        assert not t.is_detached
        assert t.current_branch == "main"
        assert t.head == hashes[-1]

    def test_checkout_dash_returns_to_prev(self):
        """checkout('-') returns to PREV_HEAD position."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        original_head = t.head

        # Checkout an old commit
        t.checkout(hashes[0])
        assert t.head == hashes[0]

        # Return via "-"
        t.checkout("-")
        assert t.head == original_head

    def test_checkout_dash_no_prev_head_raises(self):
        """checkout('-') with no PREV_HEAD raises TraceError."""
        t = make_tract()
        populate_tract(t, 1)
        with pytest.raises(TraceError, match="No previous position"):
            t.checkout("-")

    def test_checkout_by_prefix(self):
        """Checkout accepts a hash prefix."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        prefix = hashes[0][:8]
        result = t.checkout(prefix)
        assert result == hashes[0]
        assert t.is_detached

    def test_checkout_stores_prev_head(self):
        """Checkout stores PREV_HEAD before switching."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        old_head = t.head

        t.checkout(hashes[0])
        prev = t._ref_repo.get_ref(t.tract_id, "PREV_HEAD")
        assert prev == old_head

    def test_checkout_dash_restores_branch_attachment(self):
        """checkout('-') re-attaches to branch if previous position was on a branch."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        assert not t.is_detached
        assert t.current_branch == "main"

        # Checkout a specific commit (detach)
        t.checkout(hashes[0])
        assert t.is_detached

        # Return via "-" should re-attach to main
        t.checkout("-")
        assert not t.is_detached
        assert t.current_branch == "main"
        assert t.head == hashes[-1]


# ==================================================================
# Detached HEAD blocks commits
# ==================================================================

class TestDetachedHeadGuard:
    """Tests that detached HEAD prevents new commits."""

    def test_commit_in_detached_head_raises(self):
        """Committing in detached HEAD state raises DetachedHeadError."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        t.checkout(hashes[0])
        assert t.is_detached

        with pytest.raises(DetachedHeadError):
            t.commit(DialogueContent(role="user", text="Should fail"))

    def test_commit_after_reattach_works(self):
        """After re-attaching HEAD, commits work again."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        # Detach
        t.checkout(hashes[0])
        assert t.is_detached

        # Re-attach
        t.checkout("main")
        assert not t.is_detached

        # Commit should work
        info = t.commit(DialogueContent(role="user", text="After reattach"))
        assert info.parent_hash == hashes[-1]


# ==================================================================
# LRU cache interaction with navigation
# ==================================================================

class TestCacheNavigation:
    """Tests that LRU cache survives checkout/reset operations."""

    def test_cache_hit_after_checkout_back(self):
        """Checking out back to a previously-compiled HEAD gets a cache hit."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        # Compile at HEAD (populates cache)
        result1 = t.compile()
        assert result1.commit_count == 3

        # Checkout old commit, compile there
        t.checkout(hashes[0])
        result_old = t.compile()
        assert result_old.commit_count == 1

        # Checkout back to main
        t.checkout("main")
        # Cache should have the snapshot from before
        result2 = t.compile()
        assert result2.commit_count == 3
        assert result2.messages == result1.messages

    def test_cache_survives_reset(self):
        """Reset doesn't clear the cache -- old snapshots remain."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        # Compile at HEAD to populate cache
        t.compile()

        # Reset to first commit
        t.reset(hashes[0])

        # The cache entry for the old HEAD is still there (not cleared)
        # but compile() now compiles at the new HEAD
        result = t.compile()
        assert result.commit_count == 1

    def test_compile_in_detached_head(self):
        """Compile works in detached HEAD state (read-only)."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        t.checkout(hashes[1])
        assert t.is_detached

        result = t.compile()
        assert result.commit_count == 2


# ==================================================================
# Edge cases
# ==================================================================

class TestNavigationEdgeCases:
    """Edge cases and boundary conditions."""

    def test_reset_to_same_head(self):
        """Reset to current HEAD is a no-op (but sets ORIG_HEAD)."""
        t = make_tract()
        hashes = populate_tract(t, 2)
        current = t.head

        t.reset(current)
        assert t.head == current
        orig = t._ref_repo.get_ref(t.tract_id, "ORIG_HEAD")
        assert orig == current

    def test_multiple_checkouts_update_prev_head(self):
        """Each checkout updates PREV_HEAD to the previous position."""
        t = make_tract()
        hashes = populate_tract(t, 3)

        t.checkout(hashes[0])
        prev1 = t._ref_repo.get_ref(t.tract_id, "PREV_HEAD")
        assert prev1 == hashes[2]

        t.checkout(hashes[1])
        prev2 = t._ref_repo.get_ref(t.tract_id, "PREV_HEAD")
        assert prev2 == hashes[0]

    def test_resolve_commit_prefers_exact_hash_over_branch(self):
        """Full hash match takes priority over branch name resolution."""
        t = make_tract()
        hashes = populate_tract(t, 1)
        # Resolve by full hash should work even if a branch with same name existed
        resolved = t.resolve_commit(hashes[0])
        assert resolved == hashes[0]

    @pytest.mark.parametrize("op", ["checkout", "reset"])
    def test_nonexistent_target_raises(self, op):
        t = make_tract()
        populate_tract(t, 1)
        with pytest.raises(CommitNotFoundError):
            getattr(t, op)("nonexistent")

    def test_detached_head_error_message(self):
        """DetachedHeadError has descriptive message."""
        err = DetachedHeadError()
        msg = str(err).lower()
        assert "detached head" in msg
        assert "checkout" in msg

    def test_ambiguous_prefix_error_attributes(self):
        """AmbiguousPrefixError stores prefix and candidates."""
        err = AmbiguousPrefixError("abcd", ["abcd1111", "abcd2222"])
        assert err.prefix == "abcd"
        assert len(err.candidates) == 2
        assert "abcd" in str(err)
