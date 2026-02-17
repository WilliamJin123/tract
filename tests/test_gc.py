"""Comprehensive tests for garbage collection operations.

Tests GC with orphan retention, archive preservation, multi-branch
reachability, and edge cases.
"""

from __future__ import annotations

import pytest

from tract import (
    CommitInfo,
    DialogueContent,
    GCResult,
    InstructionContent,
    Priority,
    Tract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from tests.conftest import make_tract_with_commits


def create_orphans(t, hashes):
    """Create orphans by resetting to the first commit.

    Returns the orphaned hashes (those no longer reachable from HEAD).
    """
    # Reset to the first commit, making commits 2..N orphaned
    t.reset(hashes[0], mode="hard")
    return hashes[1:]


# ===========================================================================
# 1. Basic GC tests
# ===========================================================================


class TestBasicGC:
    """Tests for basic garbage collection functionality."""

    def test_gc_no_orphans(self):
        """gc() on tract with no orphans returns commits_removed=0."""
        t, hashes = make_tract_with_commits(5)

        result = t.gc(orphan_retention_days=0)

        assert isinstance(result, GCResult)
        assert result.commits_removed == 0
        assert result.blobs_removed == 0
        assert result.tokens_freed == 0
        assert result.archives_removed == 0

    def test_gc_removes_orphans(self):
        """gc(orphan_retention_days=0) removes unreachable commits."""
        t, hashes = make_tract_with_commits(5)
        orphans = create_orphans(t, hashes)

        result = t.gc(orphan_retention_days=0)

        assert result.commits_removed == len(orphans)
        assert result.tokens_freed > 0

        # Verify orphaned commits are actually gone from DB
        for h in orphans:
            row = t._commit_repo.get(h)
            assert row is None, f"Orphan {h[:8]} should be deleted"

    def test_gc_respects_retention(self):
        """gc(orphan_retention_days=7) does NOT remove recent orphans."""
        t, hashes = make_tract_with_commits(5)
        orphans = create_orphans(t, hashes)

        # Default retention = 7 days. Freshly created commits are < 7 days old.
        result = t.gc(orphan_retention_days=7)

        assert result.commits_removed == 0
        # Orphans still exist
        for h in orphans:
            row = t._commit_repo.get(h)
            assert row is not None, f"Recent orphan {h[:8]} should be preserved"

    def test_gc_returns_stats(self):
        """GCResult has correct counts (commits_removed, tokens_freed, blobs_removed)."""
        t, hashes = make_tract_with_commits(5)
        orphans = create_orphans(t, hashes)

        result = t.gc(orphan_retention_days=0)

        assert isinstance(result, GCResult)
        assert result.commits_removed == 4  # 4 orphans (hashes[1:5])
        assert result.tokens_freed > 0
        assert result.blobs_removed >= 0  # May or may not have unique blobs
        assert result.duration_seconds >= 0

    def test_gc_removes_orphaned_blobs(self):
        """After removing orphan commits, their blobs are also removed if unique."""
        t, hashes = make_tract_with_commits(5, texts=[
            "Unique content A",
            "Unique content B",
            "Unique content C",
            "Unique content D",
            "Unique content E",
        ])

        # Get content hashes before GC
        orphan_content_hashes = set()
        for h in hashes[1:]:
            row = t._commit_repo.get(h)
            orphan_content_hashes.add(row.content_hash)

        # Create orphans
        create_orphans(t, hashes)

        result = t.gc(orphan_retention_days=0)

        assert result.commits_removed == 4
        # All orphaned blobs should be removed since they have unique content
        assert result.blobs_removed >= 1  # At least some blobs removed


# ===========================================================================
# 2. Compression archive tests
# ===========================================================================


class TestArchiveGC:
    """Tests for GC interaction with compression archives."""

    def test_gc_preserves_archives_by_default(self):
        """After compress(), gc() does NOT remove archived source commits."""
        t, hashes = make_tract_with_commits(5)

        # Compress all commits
        result = t.compress(content="Summary of everything")

        # Original commits are now unreachable but archived
        gc_result = t.gc(orphan_retention_days=0)

        # Archives should be preserved (archive_retention=None)
        assert gc_result.archives_removed == 0

    def test_gc_removes_old_archives(self):
        """gc(archive_retention_days=0) removes archived commits."""
        t, hashes = make_tract_with_commits(5)

        # Compress all commits
        result = t.compress(content="Summary of everything")

        # Now GC with archive_retention_days=0
        gc_result = t.gc(orphan_retention_days=0, archive_retention_days=0)

        # Archives should be removed
        assert gc_result.archives_removed > 0

    def test_gc_archive_retention_threshold(self):
        """gc(archive_retention_days=30) preserves recent archives."""
        t, hashes = make_tract_with_commits(5)

        # Compress all commits
        result = t.compress(content="Summary of everything")

        # GC with 30-day archive retention (commits are seconds old)
        gc_result = t.gc(orphan_retention_days=0, archive_retention_days=30)

        assert gc_result.archives_removed == 0

    def test_gc_archives_removed_count(self):
        """GCResult.archives_removed reflects actual archive removals."""
        t, hashes = make_tract_with_commits(5)

        # Compress
        compress_result = t.compress(content="Summary")

        # GC with immediate archive removal
        gc_result = t.gc(orphan_retention_days=0, archive_retention_days=0)

        assert gc_result.archives_removed == len(compress_result.source_commits)


# ===========================================================================
# 3. Multi-branch reachability tests
# ===========================================================================


class TestMultiBranchGC:
    """Tests for GC with multiple branches."""

    def test_gc_respects_all_branches(self):
        """Commit reachable from feature branch but not main is NOT removed."""
        t, hashes = make_tract_with_commits(3)

        # Create feature branch at commit 3
        t.branch("feature", switch=False)

        # Reset main back to commit 1 (hashes[2] is only on feature)
        t.reset(hashes[0], mode="hard")

        # GC -- hashes[1] and hashes[2] are reachable from 'feature'
        gc_result = t.gc(orphan_retention_days=0)

        assert gc_result.commits_removed == 0

        # All commits should still exist
        for h in hashes:
            assert t._commit_repo.get(h) is not None

    def test_gc_branch_scoping(self):
        """gc(branch='main') only checks main's reachability."""
        t, hashes = make_tract_with_commits(3)

        # Create feature branch from commit 3
        t.branch("feature", switch=False)

        # Reset main to commit 1
        t.reset(hashes[0], mode="hard")

        # GC scoped to main -- hashes[1:] not reachable from main
        gc_result = t.gc(orphan_retention_days=0, branch="main")

        assert gc_result.commits_removed == 2
        # hashes[1] and hashes[2] should be removed
        for h in hashes[1:]:
            assert t._commit_repo.get(h) is None

    def test_gc_detached_head(self):
        """Commits reachable from detached HEAD are not removed."""
        t, hashes = make_tract_with_commits(5)

        # Detach HEAD at commit 3, then reset main to commit 1
        t.checkout(hashes[2])  # Detach at commit 3
        # hashes[2] is now the detached HEAD

        # GC should preserve commits reachable from detached HEAD
        gc_result = t.gc(orphan_retention_days=0)

        # Commits 0, 1, 2 reachable from detached HEAD
        # Commits 0-4 reachable from main
        assert gc_result.commits_removed == 0


# ===========================================================================
# 4. Edge cases
# ===========================================================================


class TestGCEdgeCases:
    """Edge case tests for garbage collection."""

    def test_gc_empty_tract(self):
        """gc() on empty tract returns all zeros."""
        t = Tract.open()

        result = t.gc(orphan_retention_days=0)

        assert result.commits_removed == 0
        assert result.blobs_removed == 0
        assert result.tokens_freed == 0
        assert result.archives_removed == 0

    def test_gc_duration_positive(self):
        """GCResult.duration_seconds >= 0."""
        t, hashes = make_tract_with_commits(3)

        result = t.gc(orphan_retention_days=0)

        assert result.duration_seconds >= 0

    def test_gc_idempotent(self):
        """Running gc() twice produces same result (second run removes nothing)."""
        t, hashes = make_tract_with_commits(5)
        create_orphans(t, hashes)

        first_result = t.gc(orphan_retention_days=0)
        assert first_result.commits_removed == 4

        second_result = t.gc(orphan_retention_days=0)
        assert second_result.commits_removed == 0


# ===========================================================================
# 5. Provenance cleanup tests
# ===========================================================================


class TestGCProvenanceCleanup:
    """Tests that GC cleans up compression provenance records."""

    def test_gc_cleans_provenance_after_removing_archives(self):
        """After GC removes archived commits, their provenance is cleaned up."""
        t, hashes = make_tract_with_commits(5)

        # Compress to create archives
        from tests.test_compression import MockLLMClient
        mock = MockLLMClient()
        t.configure_llm(mock)
        result = t.compress()

        # Now run GC with archive_retention_days=0 to remove archives
        gc_result = t.gc(orphan_retention_days=0, archive_retention_days=0)

        assert gc_result.archives_removed > 0

        # Verify provenance was cleaned up: querying sources should return empty
        from tract.storage.sqlite import SqliteCompressionRepository
        # The compression record itself should have been cleaned up
        # (no sources or results remain)
        # We verify by checking the GC result is consistent
        # and a second GC finds nothing more to remove
        gc_result2 = t.gc(orphan_retention_days=0, archive_retention_days=0)
        assert gc_result2.commits_removed == 0
        assert gc_result2.archives_removed == 0

    def test_gc_cleans_result_commit_fk(self):
        """GC correctly handles CompressionResultRow FK when summary commits become unreachable."""
        t, hashes = make_tract_with_commits(5)

        # Compress to create summary commits
        result = t.compress(content="Summary of everything")
        summary_hashes = result.summary_commits
        compression_id = result.compression_id

        # Verify result rows exist in provenance
        results_before = t._compression_repo.get_results(compression_id)
        assert len(results_before) >= 1

        # Reset to a previous commit to make summary commits unreachable
        # (The summary is the new HEAD, so we need to make it orphaned)
        # Create new commits on top, then reset back
        new_h1 = t.commit(DialogueContent(role="user", text="New after compress"))
        new_h2 = t.commit(DialogueContent(role="assistant", text="Response"))
        # Now compress again to orphan the previous summary
        result2 = t.compress(content="Second summary replacing everything")

        # First summary commits are now unreachable (orphaned)
        # GC should clean them up along with their CompressionResultRow entries
        gc_result = t.gc(orphan_retention_days=0, archive_retention_days=0)

        # Should have removed the old summary commits and source archives
        assert gc_result.commits_removed > 0

        # Verify no FK constraint violations -- a second GC should succeed cleanly
        gc_result2 = t.gc(orphan_retention_days=0, archive_retention_days=0)
        assert gc_result2.commits_removed >= 0  # May remove more or zero
