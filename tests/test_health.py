"""Tests for DAG health check and validation operations."""

from __future__ import annotations

import pytest

from tract import (
    DialogueContent,
    HealthReport,
    InstructionContent,
    Tract,
)

from tests.conftest import make_tract_with_commits


# ===========================================================================
# 1. Basic health checks
# ===========================================================================


class TestHealthyTract:
    """Tests that healthy tracts are reported as healthy."""

    def test_empty_tract_is_healthy(self):
        """An empty tract with no commits should be healthy."""
        t = Tract.open()
        report = t.health()

        assert isinstance(report, HealthReport)
        assert report.healthy is True
        assert report.commit_count == 0
        assert report.branch_count == 0  # no branches until first commit
        assert report.orphan_count == 0
        assert report.missing_blobs == []
        assert report.missing_parents == []
        assert report.unreachable_commits == []
        assert report.warnings == []

    def test_single_commit_healthy(self):
        """A tract with one commit is healthy."""
        t = Tract.open()
        t.commit(InstructionContent(text="System prompt"))
        report = t.health()

        assert report.healthy is True
        assert report.commit_count == 1
        assert report.orphan_count == 0

    def test_multiple_commits_healthy(self):
        """A tract with a linear chain of commits is healthy."""
        t, hashes = make_tract_with_commits(5)
        report = t.health()

        assert report.healthy is True
        assert report.commit_count == 5
        assert report.orphan_count == 0
        assert report.missing_blobs == []
        assert report.missing_parents == []

    def test_correct_branch_count(self):
        """Branch count reflects all branches in the tract."""
        t, hashes = make_tract_with_commits(3)
        t.branch("feature-a", switch=False)
        t.branch("feature-b", switch=False)

        report = t.health()

        assert report.healthy is True
        assert report.branch_count == 3  # main + feature-a + feature-b

    def test_correct_commit_count(self):
        """Commit count reflects total commits in the tract."""
        t, hashes = make_tract_with_commits(7)
        report = t.health()
        assert report.commit_count == 7


# ===========================================================================
# 2. Orphan detection
# ===========================================================================


class TestOrphanDetection:
    """Tests for detecting unreachable (orphaned) commits."""

    def test_orphans_after_reset(self):
        """After hard reset, unreachable commits are reported as orphans."""
        t, hashes = make_tract_with_commits(5)

        # Reset to first commit, orphaning commits 2-5
        t.reset(hashes[0], mode="hard")

        report = t.health()

        assert report.healthy is True  # orphans produce warnings, not errors
        assert report.orphan_count == 4
        assert len(report.unreachable_commits) == 4
        assert len(report.warnings) == 1
        assert "unreachable" in report.warnings[0].lower()

    def test_branch_reachability_prevents_orphans(self):
        """Commits reachable from any branch are not orphans."""
        t, hashes = make_tract_with_commits(5)

        # Create branch at tip, then reset main
        t.branch("feature", switch=False)
        t.reset(hashes[0], mode="hard")

        report = t.health()

        # All commits reachable from 'feature' branch
        assert report.orphan_count == 0
        assert report.healthy is True

    def test_orphan_hashes_are_correct(self):
        """Unreachable_commits list contains the correct orphan hashes."""
        t, hashes = make_tract_with_commits(5)
        t.reset(hashes[1], mode="hard")

        report = t.health()

        orphan_set = set(report.unreachable_commits)
        expected_orphans = set(hashes[2:])
        assert orphan_set == expected_orphans


# ===========================================================================
# 3. Blob integrity
# ===========================================================================


class TestBlobIntegrity:
    """Tests for missing blob detection."""

    def test_no_missing_blobs_normal(self):
        """Normal tract has no missing blobs."""
        t, hashes = make_tract_with_commits(3)
        report = t.health()
        assert report.missing_blobs == []

    def test_detect_missing_blob(self):
        """If a blob is manually deleted, health check detects it."""
        t, hashes = make_tract_with_commits(3)

        # Manually corrupt: delete a blob from the DB
        commit_row = t._commit_repo.get(hashes[1])
        content_hash = commit_row.content_hash

        # Must disable FK checks to delete a referenced blob
        from sqlalchemy import text
        t._session.execute(text("PRAGMA foreign_keys = OFF"))
        t._session.execute(
            text("DELETE FROM blobs WHERE content_hash = :h"),
            {"h": content_hash},
        )
        t._session.execute(text("PRAGMA foreign_keys = ON"))
        t._session.flush()

        report = t.health()

        assert report.healthy is False
        assert hashes[1] in report.missing_blobs


# ===========================================================================
# 4. Parent integrity
# ===========================================================================


class TestParentIntegrity:
    """Tests for missing parent detection."""

    def test_no_missing_parents_normal(self):
        """Normal tract has no missing parent references."""
        t, hashes = make_tract_with_commits(5)
        report = t.health()
        assert report.missing_parents == []

    def test_detect_missing_parent(self):
        """If a parent commit is deleted, health check detects it."""
        t, hashes = make_tract_with_commits(3)

        # Manually corrupt: update a commit's parent_hash to a nonexistent hash
        from sqlalchemy import text
        fake_parent = "deadbeef" * 8  # 64 char fake hash
        t._session.execute(text("PRAGMA foreign_keys = OFF"))
        t._session.execute(
            text("UPDATE commits SET parent_hash = :p WHERE commit_hash = :h"),
            {"p": fake_parent, "h": hashes[2]},
        )
        t._session.execute(text("PRAGMA foreign_keys = ON"))
        t._session.flush()
        # Clear SQLAlchemy cache so the repo sees the raw update
        t._session.expire_all()

        report = t.health()

        assert report.healthy is False
        assert len(report.missing_parents) == 1
        assert report.missing_parents[0] == (hashes[2], fake_parent)


# ===========================================================================
# 5. Branch HEAD validity
# ===========================================================================


class TestBranchHeadValidity:
    """Tests for branch HEAD pointing to missing commits."""

    def test_valid_branch_heads(self):
        """Normal branches point to existing commits."""
        t, hashes = make_tract_with_commits(3)
        t.branch("feature", switch=False)
        report = t.health()
        assert report.healthy is True

    def test_detect_invalid_branch_head(self):
        """If a branch HEAD points to a missing commit, health check detects it."""
        t, hashes = make_tract_with_commits(3)

        # Manually corrupt: set a branch to point to nonexistent commit
        fake_hash = "abcdef12" * 8
        from sqlalchemy import text
        t._session.execute(text("PRAGMA foreign_keys = OFF"))
        t._session.execute(
            text(
                "INSERT INTO refs (tract_id, ref_name, commit_hash) "
                "VALUES (:tid, :ref, :h)"
            ),
            {"tid": t._tract_id, "ref": "refs/heads/broken", "h": fake_hash},
        )
        t._session.execute(text("PRAGMA foreign_keys = ON"))
        t._session.flush()

        report = t.health()

        assert report.healthy is False
        found = any("broken" in w and "missing commit" in w for w in report.warnings)
        assert found, f"Expected branch HEAD warning, got: {report.warnings}"


# ===========================================================================
# 6. Summary format
# ===========================================================================


class TestSummaryFormat:
    """Tests for the HealthReport.summary() output."""

    def test_healthy_summary(self):
        """Healthy report summary starts with 'Health: OK'."""
        t, _ = make_tract_with_commits(3)
        report = t.health()
        s = report.summary()
        assert s.startswith("Health: OK")
        assert "Commits: 3" in s

    def test_unhealthy_summary(self):
        """Unhealthy report summary shows 'ISSUES FOUND'."""
        report = HealthReport(healthy=False, missing_blobs=["abc123"])
        s = report.summary()
        assert "ISSUES FOUND" in s
        assert "Missing blobs: 1" in s

    def test_summary_with_orphans(self):
        """Summary includes orphan count when present."""
        report = HealthReport(orphan_count=5, unreachable_commits=["a", "b", "c", "d", "e"])
        report.warnings.append("5 unreachable commits (run gc to clean)")
        s = report.summary()
        assert "Orphan commits: 5" in s
        assert "Warning:" in s

    def test_summary_with_missing_parents(self):
        """Summary includes missing parent count when present."""
        report = HealthReport(
            healthy=False,
            missing_parents=[("abc", "def"), ("ghi", "jkl")],
        )
        s = report.summary()
        assert "Missing parents: 2" in s


# ===========================================================================
# 7. Health after operations
# ===========================================================================


class TestHealthAfterOperations:
    """Tests that health check passes after various operations."""

    def test_health_after_branch_and_merge(self):
        """Health is OK after creating a branch and merging."""
        t, hashes = make_tract_with_commits(3)
        t.branch("feature", switch=True)
        t.commit(DialogueContent(role="user", text="Feature work"))
        t.switch("main")
        t.merge("feature")

        report = t.health()
        assert report.healthy is True
        assert report.commit_count >= 4

    def test_health_after_compress(self):
        """Health is OK after compression."""
        t, hashes = make_tract_with_commits(5)
        t.compress(content="Summary of everything")

        report = t.health()
        assert report.healthy is True

    def test_health_after_gc(self):
        """Health is OK after garbage collection."""
        t, hashes = make_tract_with_commits(5)
        t.reset(hashes[0], mode="hard")
        t.gc(orphan_retention_days=0)

        report = t.health()
        assert report.healthy is True
        assert report.orphan_count == 0

    def test_health_after_compress_then_gc(self):
        """Health is OK after compress followed by gc."""
        t, hashes = make_tract_with_commits(5)
        t.compress(content="Summary of everything")
        t.gc(orphan_retention_days=0, archive_retention_days=0)

        report = t.health()
        assert report.healthy is True

    def test_health_after_edit_commit(self):
        """Health is OK after creating an EDIT commit."""
        from tract import CommitOperation
        t = Tract.open()
        info = t.commit(DialogueContent(role="user", text="Original"))
        t.commit(
            DialogueContent(role="user", text="Revised"),
            operation=CommitOperation.EDIT,
            edit_target=info.commit_hash,
        )

        report = t.health()
        assert report.healthy is True

    def test_health_detached_head(self):
        """Health is OK when HEAD is detached."""
        t, hashes = make_tract_with_commits(3)
        t.checkout(hashes[1])  # Detach HEAD

        report = t.health()
        assert report.healthy is True
        # All commits should still be reachable (from main and detached HEAD)
        assert report.orphan_count == 0


# ===========================================================================
# 8. Import from package
# ===========================================================================


class TestImport:
    """Test that HealthReport is importable from the public API."""

    def test_import_health_report(self):
        """HealthReport is importable from tract."""
        from tract import HealthReport as HR
        assert HR is HealthReport

    def test_health_method_exists(self):
        """Tract has a health() method."""
        assert hasattr(Tract, "health")
        assert callable(getattr(Tract, "health"))
