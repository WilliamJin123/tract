"""Tests for SQLAlchemy ORM schema.

Covers:
- All tables are created
- BlobRow round-trip
- CommitRow round-trip with all fields
- RefRow composite PK
- AnnotationRow autoincrement ID
- Foreign key constraints
- Indexes exist on expected columns
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from tract.models.annotations import Priority
from tract.models.commit import CommitOperation
from tract.storage.schema import (
    AnnotationRow,
    Base,
    BlobRow,
    CommitRow,
    RefRow,
    TraceMetaRow,
)


class TestTableCreation:
    def test_all_tables_exist(self, engine):
        """Verify all 5 expected tables are created."""
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        expected = {"blobs", "commits", "refs", "annotations", "commit_parents", "_trace_meta"}
        assert expected <= table_names, f"Missing tables: {expected - table_names}"

    def test_trace_meta_has_schema_version(self, session):
        """Schema version should be set by init_db."""
        row = session.execute(
            select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        ).scalar_one_or_none()
        assert row is not None
        assert row.value == "5"


class TestBlobRow:
    def test_round_trip(self, session):
        """Create, save, and query a blob by PK."""
        now = datetime.now(timezone.utc)
        blob = BlobRow(
            content_hash="abc123" + "0" * 58,
            payload_json='{"content_type":"instruction","text":"hello"}',
            byte_size=45,
            token_count=5,
            created_at=now,
        )
        session.add(blob)
        session.flush()

        result = session.execute(
            select(BlobRow).where(BlobRow.content_hash == blob.content_hash)
        ).scalar_one()

        assert result.content_hash == blob.content_hash
        assert result.payload_json == blob.payload_json
        assert result.byte_size == 45
        assert result.token_count == 5


class TestCommitRow:
    def _make_blob(self, session, content_hash="blob_hash_" + "0" * 54):
        """Helper: create and flush a blob row."""
        now = datetime.now(timezone.utc)
        blob = BlobRow(
            content_hash=content_hash,
            payload_json='{"content_type":"instruction","text":"test"}',
            byte_size=40,
            token_count=4,
            created_at=now,
        )
        session.add(blob)
        session.flush()
        return blob

    def test_round_trip_all_fields(self, session):
        """CommitRow with all fields populated survives save/query."""
        blob = self._make_blob(session)
        now = datetime.now(timezone.utc)

        commit = CommitRow(
            commit_hash="commit_" + "a" * 57,
            tract_id="test-tract",
            parent_hash=None,
            content_hash=blob.content_hash,
            content_type="instruction",
            operation=CommitOperation.APPEND,
            response_to=None,
            message="Initial commit",
            token_count=4,
            metadata_json={"key": "value"},
            created_at=now,
        )
        session.add(commit)
        session.flush()

        result = session.execute(
            select(CommitRow).where(CommitRow.commit_hash == commit.commit_hash)
        ).scalar_one()

        assert result.commit_hash == commit.commit_hash
        assert result.tract_id == "test-tract"
        assert result.operation == CommitOperation.APPEND
        assert result.message == "Initial commit"
        assert result.metadata_json == {"key": "value"}

    def test_edit_operation(self, session):
        """CommitRow with EDIT operation."""
        blob = self._make_blob(session)
        now = datetime.now(timezone.utc)

        # Create original commit
        original = CommitRow(
            commit_hash="original_" + "a" * 55,
            tract_id="test-tract",
            content_hash=blob.content_hash,
            content_type="instruction",
            operation=CommitOperation.APPEND,
            token_count=4,
            created_at=now,
        )
        session.add(original)
        session.flush()

        # Create edit blob and commit
        edit_blob = self._make_blob(session, content_hash="edit_blob_" + "0" * 54)

        edit = CommitRow(
            commit_hash="edit_" + "b" * 60,
            tract_id="test-tract",
            parent_hash=original.commit_hash,
            content_hash=edit_blob.content_hash,
            content_type="instruction",
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
            token_count=4,
            created_at=now,
        )
        session.add(edit)
        session.flush()

        result = session.execute(
            select(CommitRow).where(CommitRow.commit_hash == edit.commit_hash)
        ).scalar_one()
        assert result.operation == CommitOperation.EDIT
        assert result.response_to == original.commit_hash

    def test_fk_invalid_content_hash_fails(self, session):
        """Commit referencing nonexistent blob should fail on flush."""
        now = datetime.now(timezone.utc)
        commit = CommitRow(
            commit_hash="bad_commit_" + "x" * 53,
            tract_id="test-tract",
            content_hash="nonexistent_blob_hash_" + "0" * 42,
            content_type="instruction",
            operation=CommitOperation.APPEND,
            token_count=0,
            created_at=now,
        )
        session.add(commit)
        with pytest.raises(IntegrityError):
            session.flush()


class TestRefRow:
    def _make_commit_with_blob(self, session, commit_hash, tract_id="test-tract"):
        """Helper: create blob + commit for FK satisfaction."""
        now = datetime.now(timezone.utc)
        blob_hash = f"blob_{commit_hash}"[:64]
        blob = BlobRow(
            content_hash=blob_hash,
            payload_json='{}',
            byte_size=2,
            token_count=0,
            created_at=now,
        )
        session.add(blob)
        session.flush()

        commit = CommitRow(
            commit_hash=commit_hash,
            tract_id=tract_id,
            content_hash=blob_hash,
            content_type="instruction",
            operation=CommitOperation.APPEND,
            token_count=0,
            created_at=now,
        )
        session.add(commit)
        session.flush()
        return commit

    def test_composite_pk(self, session):
        """RefRow uses (tract_id, ref_name) as composite PK."""
        commit = self._make_commit_with_blob(session, "ref_commit_" + "a" * 53)

        ref = RefRow(
            tract_id="test-tract",
            ref_name="HEAD",
            commit_hash=commit.commit_hash,
        )
        session.add(ref)
        session.flush()

        result = session.execute(
            select(RefRow).where(
                RefRow.tract_id == "test-tract", RefRow.ref_name == "HEAD"
            )
        ).scalar_one()
        assert result.commit_hash == commit.commit_hash

    def test_different_tracts_same_ref_name(self, session):
        """Different tracts can have refs with the same name."""
        c1 = self._make_commit_with_blob(session, "commit1_" + "a" * 56, "tract-1")
        c2 = self._make_commit_with_blob(session, "commit2_" + "b" * 56, "tract-2")

        session.add(RefRow(tract_id="tract-1", ref_name="HEAD", commit_hash=c1.commit_hash))
        session.add(RefRow(tract_id="tract-2", ref_name="HEAD", commit_hash=c2.commit_hash))
        session.flush()

        r1 = session.execute(
            select(RefRow).where(RefRow.tract_id == "tract-1", RefRow.ref_name == "HEAD")
        ).scalar_one()
        r2 = session.execute(
            select(RefRow).where(RefRow.tract_id == "tract-2", RefRow.ref_name == "HEAD")
        ).scalar_one()
        assert r1.commit_hash != r2.commit_hash


class TestAnnotationRow:
    def _make_commit_with_blob(self, session, commit_hash, tract_id="test-tract"):
        """Helper: create blob + commit for FK satisfaction."""
        now = datetime.now(timezone.utc)
        blob_hash = f"blob_{commit_hash}"[:64]
        blob = BlobRow(
            content_hash=blob_hash,
            payload_json='{}',
            byte_size=2,
            token_count=0,
            created_at=now,
        )
        session.add(blob)
        session.flush()

        commit = CommitRow(
            commit_hash=commit_hash,
            tract_id=tract_id,
            content_hash=blob_hash,
            content_type="instruction",
            operation=CommitOperation.APPEND,
            token_count=0,
            created_at=now,
        )
        session.add(commit)
        session.flush()
        return commit

    def test_autoincrement_id(self, session):
        """AnnotationRow id auto-increments."""
        commit = self._make_commit_with_blob(session, "ann_commit_" + "a" * 53)
        now = datetime.now(timezone.utc)

        a1 = AnnotationRow(
            tract_id="test-tract",
            target_hash=commit.commit_hash,
            priority=Priority.NORMAL,
            created_at=now,
        )
        session.add(a1)
        session.flush()

        a2 = AnnotationRow(
            tract_id="test-tract",
            target_hash=commit.commit_hash,
            priority=Priority.PINNED,
            reason="Important",
            created_at=now,
        )
        session.add(a2)
        session.flush()

        assert a1.id is not None
        assert a2.id is not None
        assert a2.id > a1.id


class TestIndexes:
    def test_expected_indexes_exist(self, engine):
        """Verify key indexes are created."""
        inspector = inspect(engine)

        # Check commits table indexes
        commit_indexes = inspector.get_indexes("commits")
        commit_index_names = {idx["name"] for idx in commit_indexes}
        assert "ix_commits_tract_time" in commit_index_names
        assert "ix_commits_tract_type" in commit_index_names
        assert "ix_commits_response_to" in commit_index_names

        # Check annotations table indexes
        ann_indexes = inspector.get_indexes("annotations")
        ann_index_names = {idx["name"] for idx in ann_indexes}
        assert "ix_annotations_target_time" in ann_index_names
