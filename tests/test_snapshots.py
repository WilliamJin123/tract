"""Tests for the snapshot (restore point) system.

Covers:
- Create snapshot with label
- Create snapshot without label (uses timestamp)
- list_snapshots returns correct data
- restore_snapshot with branch creation
- restore_snapshot with direct reset (create_branch=False)
- Snapshot survives close/reopen (persistence)
- Multiple snapshots in order
- Restore to specific snapshot by tag
- Restore to snapshot by label substring
- Error on unknown snapshot
- Snapshot metadata is correct
"""

import time

import pytest

from tract import Tract, TraceError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tract(**kwargs) -> Tract:
    """Create an in-memory Tract for testing."""
    return Tract.open(":memory:", **kwargs)


# ---------------------------------------------------------------------------
# Basic snapshot creation
# ---------------------------------------------------------------------------

class TestSnapshotCreate:
    """Creating snapshots with and without labels."""

    def test_snapshot_with_label(self):
        """Snapshot with a label returns a tag containing that label."""
        t = make_tract()
        t.user("hello")
        tag = t.snapshot("before-compress")
        assert tag.startswith("snapshot:before-compress:")
        assert len(tag) > len("snapshot:before-compress:")

    def test_snapshot_without_label(self):
        """Snapshot without a label falls back to timestamp-based tag."""
        t = make_tract()
        t.user("hello")
        tag = t.snapshot()
        # Should be "snapshot:<timestamp>:<hash>"
        parts = tag.split(":")
        assert parts[0] == "snapshot"
        assert parts[1].isdigit()  # timestamp
        assert len(parts[2]) == 7  # short hash

    def test_snapshot_on_empty_tract_raises(self):
        """Snapshot on a tract with no commits raises TraceError."""
        t = make_tract()
        with pytest.raises(TraceError, match="no commits"):
            t.snapshot("oops")


# ---------------------------------------------------------------------------
# list_snapshots
# ---------------------------------------------------------------------------

class TestListSnapshots:
    """Listing snapshots returns correct data."""

    def test_list_returns_correct_fields(self):
        """Each snapshot dict has the expected keys."""
        t = make_tract()
        t.user("first")
        tag = t.snapshot("checkpoint-1")

        snaps = t.list_snapshots()
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap["tag"] == tag
        assert snap["label"] == "checkpoint-1"
        assert snap["head"] != ""  # has a commit hash
        assert isinstance(snap["timestamp"], int)
        assert snap["timestamp"] > 0
        assert snap["hash"] != ""  # the snapshot commit's own hash

    def test_list_empty_when_no_snapshots(self):
        """Empty list when no snapshots have been created."""
        t = make_tract()
        t.user("hello")
        assert t.list_snapshots() == []

    def test_list_multiple_in_order(self):
        """Multiple snapshots are returned newest first."""
        t = make_tract()
        t.user("msg1")
        t.snapshot("first")
        t.user("msg2")
        t.snapshot("second")

        snaps = t.list_snapshots()
        assert len(snaps) == 2
        # Newest first (log returns reverse chronological)
        assert snaps[0]["label"] == "second"
        assert snaps[1]["label"] == "first"


# ---------------------------------------------------------------------------
# Snapshot metadata correctness
# ---------------------------------------------------------------------------

class TestSnapshotMetadata:
    """Snapshot metadata records the correct state."""

    def test_metadata_records_head(self):
        """Snapshot records the HEAD hash at the time it was created."""
        t = make_tract()
        t.user("hello")
        head_before = t.head
        t.snapshot("check")

        snaps = t.list_snapshots()
        assert snaps[0]["head"] == head_before

    def test_metadata_records_branch(self):
        """Snapshot records the current branch name."""
        t = make_tract()
        t.user("hello")
        branch = t.current_branch
        t.snapshot("check")

        snaps = t.list_snapshots()
        assert snaps[0]["branch"] == branch

    def test_custom_metadata_stored(self):
        """Custom metadata passed to snapshot() is stored in the commit."""
        t = make_tract()
        t.user("hello")
        t.snapshot("check", metadata={"reason": "pre-merge"})

        # Verify via log that the metadata commit has our custom field
        for entry in t.log(limit=10):
            meta = entry.metadata or {}
            if meta.get("snapshot"):
                assert meta.get("reason") == "pre-merge"
                break
        else:
            pytest.fail("Snapshot commit not found in log")


# ---------------------------------------------------------------------------
# restore_snapshot with branch creation (default)
# ---------------------------------------------------------------------------

class TestRestoreSnapshotWithBranch:
    """Restoring snapshots via branch creation (safe mode)."""

    def test_restore_creates_branch_and_switches(self):
        """Restore with create_branch=True creates a recovery branch."""
        t = make_tract()
        t.user("msg1")
        head_at_snap = t.head
        tag = t.snapshot("safe-point")

        # Add more commits after snapshot
        t.user("msg2")
        t.user("msg3")
        assert t.head != head_at_snap

        # Restore
        restored = t.restore_snapshot(tag)
        assert restored == head_at_snap
        assert t.current_branch == "restore/safe-point"
        assert t.head == head_at_snap

    def test_restore_by_label_substring(self):
        """Restore by matching a substring of the label."""
        t = make_tract()
        t.user("msg1")
        head_at_snap = t.head
        t.snapshot("pre-dangerous-operation")

        t.user("msg2")

        restored = t.restore_snapshot("dangerous")
        assert restored == head_at_snap

    def test_restore_by_full_tag(self):
        """Restore by passing the exact tag name."""
        t = make_tract()
        t.user("msg1")
        head_at_snap = t.head
        tag = t.snapshot("exact")

        t.user("msg2")

        restored = t.restore_snapshot(tag)
        assert restored == head_at_snap


# ---------------------------------------------------------------------------
# restore_snapshot with direct reset
# ---------------------------------------------------------------------------

class TestRestoreSnapshotWithReset:
    """Restoring snapshots via direct HEAD reset."""

    def test_restore_with_reset(self):
        """Restore with create_branch=False resets HEAD directly."""
        t = make_tract()
        t.user("msg1")
        head_at_snap = t.head
        t.snapshot("reset-point")

        t.user("msg2")
        t.user("msg3")

        restored = t.restore_snapshot("reset-point", create_branch=False)
        assert restored == head_at_snap
        assert t.head == head_at_snap


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestRestoreSnapshotErrors:
    """Error cases for restore_snapshot."""

    def test_unknown_snapshot_raises(self):
        """ValueError when no matching snapshot is found."""
        t = make_tract()
        t.user("hello")
        t.snapshot("exists")

        with pytest.raises(ValueError, match="Snapshot not found"):
            t.restore_snapshot("nonexistent-label-xyz")

    def test_unknown_tag_raises(self):
        """ValueError when tag doesn't match any snapshot."""
        t = make_tract()
        t.user("hello")

        with pytest.raises(ValueError, match="Snapshot not found"):
            t.restore_snapshot("snapshot:bogus:0000000")


# ---------------------------------------------------------------------------
# Persistence across close/reopen
# ---------------------------------------------------------------------------

class TestSnapshotPersistence:
    """Snapshots survive tract close and reopen."""

    def test_snapshot_persists(self, tmp_path):
        """Snapshot metadata is readable after closing and reopening."""
        db_path = str(tmp_path / "test.db")

        # Create and snapshot
        t = Tract.open(db_path, tract_id="persist-test")
        t.user("hello")
        head_at_snap = t.head
        tag = t.snapshot("durable")
        t.close()

        # Reopen and verify
        t2 = Tract.open(db_path, tract_id="persist-test")
        snaps = t2.list_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["tag"] == tag
        assert snaps[0]["head"] == head_at_snap
        assert snaps[0]["label"] == "durable"
        t2.close()


# ---------------------------------------------------------------------------
# Multiple snapshots and selective restore
# ---------------------------------------------------------------------------

class TestMultipleSnapshots:
    """Working with multiple snapshots."""

    def test_restore_picks_first_match(self):
        """When multiple snapshots match, the newest one wins (log order)."""
        t = make_tract()
        t.user("msg1")
        head1 = t.head
        t.snapshot("deploy-v1")

        t.user("msg2")
        head2 = t.head
        t.snapshot("deploy-v2")

        t.user("msg3")

        # "deploy" matches both; newest (deploy-v2) should be returned first
        restored = t.restore_snapshot("deploy")
        assert restored == head2

    def test_restore_specific_older_snapshot(self):
        """Can restore to an older snapshot by using its exact tag."""
        t = make_tract()
        t.user("msg1")
        head1 = t.head
        tag1 = t.snapshot("alpha")

        t.user("msg2")
        t.snapshot("beta")

        t.user("msg3")

        # Restore to the older one by exact tag
        restored = t.restore_snapshot(tag1)
        assert restored == head1
