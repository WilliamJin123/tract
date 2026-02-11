"""Tests for repository implementations.

Covers:
- SqliteBlobRepository deduplication
- SqliteCommitRepository CRUD and ancestor chain
- SqliteRefRepository HEAD and branch operations
- SqliteAnnotationRepository latest, history, and batch operations
"""

from datetime import datetime, timezone, timedelta

import pytest

from tract.models.annotations import Priority
from tract.models.commit import CommitOperation
from tract.storage.schema import (
    AnnotationRow,
    BlobRow,
    CommitRow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blob(content_hash: str, payload: str = '{"content_type":"instruction","text":"test"}'):
    """Create a BlobRow instance."""
    return BlobRow(
        content_hash=content_hash,
        payload_json=payload,
        byte_size=len(payload),
        token_count=4,
        created_at=datetime.now(timezone.utc),
    )


def _make_commit(
    commit_hash: str,
    tract_id: str,
    content_hash: str,
    parent_hash: str | None = None,
    content_type: str = "instruction",
    operation: CommitOperation = CommitOperation.APPEND,
    response_to: str | None = None,
    created_at: datetime | None = None,
):
    """Create a CommitRow instance."""
    return CommitRow(
        commit_hash=commit_hash,
        tract_id=tract_id,
        parent_hash=parent_hash,
        content_hash=content_hash,
        content_type=content_type,
        operation=operation,
        response_to=response_to,
        token_count=4,
        created_at=created_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Blob Repository
# ---------------------------------------------------------------------------


class TestSqliteBlobRepository:
    def test_save_and_get(self, blob_repo):
        blob = _make_blob("hash_save_get_" + "0" * 50)
        blob_repo.save_if_absent(blob)
        result = blob_repo.get(blob.content_hash)
        assert result is not None
        assert result.content_hash == blob.content_hash

    def test_get_nonexistent(self, blob_repo):
        result = blob_repo.get("nonexistent_" + "0" * 52)
        assert result is None

    def test_deduplication(self, blob_repo, session):
        """Saving the same blob twice results in only one row."""
        hash_val = "dedup_hash_" + "0" * 53
        blob1 = _make_blob(hash_val)
        blob2 = _make_blob(hash_val)

        blob_repo.save_if_absent(blob1)
        blob_repo.save_if_absent(blob2)  # Should be a no-op

        # Count rows with this hash
        from sqlalchemy import select, func
        from tract.storage.schema import BlobRow as BR
        count = session.execute(
            select(func.count()).where(BR.content_hash == hash_val)
        ).scalar()
        assert count == 1


# ---------------------------------------------------------------------------
# Commit Repository
# ---------------------------------------------------------------------------


class TestSqliteCommitRepository:
    def _setup_blob(self, blob_repo, content_hash="test_blob_" + "0" * 54):
        blob = _make_blob(content_hash)
        blob_repo.save_if_absent(blob)
        return blob

    def test_get_nonexistent(self, commit_repo):
        result = commit_repo.get("nonexistent_" + "0" * 52)
        assert result is None

    def test_save_and_get(self, commit_repo, blob_repo, sample_tract_id):
        blob = self._setup_blob(blob_repo)
        commit = _make_commit(
            commit_hash="commit_1_" + "a" * 55,
            tract_id=sample_tract_id,
            content_hash=blob.content_hash,
        )
        commit_repo.save(commit)

        result = commit_repo.get(commit.commit_hash)
        assert result is not None
        assert result.commit_hash == commit.commit_hash
        assert result.tract_id == sample_tract_id
        assert result.operation == CommitOperation.APPEND

    def test_get_ancestors_chain(self, commit_repo, blob_repo, sample_tract_id):
        """get_ancestors returns correct parent chain."""
        blob = self._setup_blob(blob_repo)
        now = datetime.now(timezone.utc)

        c1 = _make_commit("c1_" + "a" * 61, sample_tract_id, blob.content_hash, created_at=now)
        c2 = _make_commit("c2_" + "b" * 61, sample_tract_id, blob.content_hash,
                          parent_hash=c1.commit_hash, created_at=now + timedelta(seconds=1))
        c3 = _make_commit("c3_" + "c" * 61, sample_tract_id, blob.content_hash,
                          parent_hash=c2.commit_hash, created_at=now + timedelta(seconds=2))

        commit_repo.save(c1)
        commit_repo.save(c2)
        commit_repo.save(c3)

        ancestors = commit_repo.get_ancestors(c3.commit_hash)
        assert len(ancestors) == 3
        assert ancestors[0].commit_hash == c3.commit_hash
        assert ancestors[1].commit_hash == c2.commit_hash
        assert ancestors[2].commit_hash == c1.commit_hash

    def test_get_ancestors_with_limit(self, commit_repo, blob_repo, sample_tract_id):
        blob = self._setup_blob(blob_repo)
        now = datetime.now(timezone.utc)

        c1 = _make_commit("lim1_" + "a" * 59, sample_tract_id, blob.content_hash, created_at=now)
        c2 = _make_commit("lim2_" + "b" * 59, sample_tract_id, blob.content_hash,
                          parent_hash=c1.commit_hash, created_at=now + timedelta(seconds=1))
        c3 = _make_commit("lim3_" + "c" * 59, sample_tract_id, blob.content_hash,
                          parent_hash=c2.commit_hash, created_at=now + timedelta(seconds=2))

        commit_repo.save(c1)
        commit_repo.save(c2)
        commit_repo.save(c3)

        ancestors = commit_repo.get_ancestors(c3.commit_hash, limit=2)
        assert len(ancestors) == 2

    def test_get_by_type(self, commit_repo, blob_repo, sample_tract_id):
        blob = self._setup_blob(blob_repo)
        now = datetime.now(timezone.utc)

        c1 = _make_commit("type1_" + "a" * 58, sample_tract_id, blob.content_hash,
                          content_type="instruction", created_at=now)
        c2 = _make_commit("type2_" + "b" * 58, sample_tract_id, blob.content_hash,
                          content_type="dialogue", created_at=now + timedelta(seconds=1))
        c3 = _make_commit("type3_" + "c" * 58, sample_tract_id, blob.content_hash,
                          content_type="instruction", created_at=now + timedelta(seconds=2))

        commit_repo.save(c1)
        commit_repo.save(c2)
        commit_repo.save(c3)

        instructions = commit_repo.get_by_type("instruction", sample_tract_id)
        assert len(instructions) == 2
        assert all(c.content_type == "instruction" for c in instructions)

    def test_get_children(self, commit_repo, blob_repo, sample_tract_id):
        blob = self._setup_blob(blob_repo)
        now = datetime.now(timezone.utc)

        parent = _make_commit("parent_" + "a" * 57, sample_tract_id, blob.content_hash, created_at=now)
        child1 = _make_commit("child1_" + "b" * 57, sample_tract_id, blob.content_hash,
                              parent_hash=parent.commit_hash, created_at=now + timedelta(seconds=1))
        child2 = _make_commit("child2_" + "c" * 57, sample_tract_id, blob.content_hash,
                              parent_hash=parent.commit_hash, created_at=now + timedelta(seconds=2))

        commit_repo.save(parent)
        commit_repo.save(child1)
        commit_repo.save(child2)

        children = commit_repo.get_children(parent.commit_hash)
        assert len(children) == 2


# ---------------------------------------------------------------------------
# Ref Repository
# ---------------------------------------------------------------------------


class TestSqliteRefRepository:
    def _make_commit_with_blob(self, blob_repo, commit_repo, commit_hash, tract_id):
        """Create a commit with its blob for FK satisfaction."""
        blob = _make_blob(f"blob_{commit_hash}"[:64])
        blob_repo.save_if_absent(blob)
        commit = _make_commit(commit_hash, tract_id, blob.content_hash)
        commit_repo.save(commit)
        return commit

    def test_head_none_initially(self, ref_repo, sample_tract_id):
        assert ref_repo.get_head(sample_tract_id) is None

    def test_update_and_get_head(self, ref_repo, blob_repo, commit_repo, sample_tract_id):
        commit = self._make_commit_with_blob(
            blob_repo, commit_repo, "head_commit_" + "a" * 52, sample_tract_id
        )
        ref_repo.update_head(sample_tract_id, commit.commit_hash)
        assert ref_repo.get_head(sample_tract_id) == commit.commit_hash

    def test_update_head_twice(self, ref_repo, blob_repo, commit_repo, sample_tract_id):
        """Updating HEAD twice should update the existing ref, not create two."""
        c1 = self._make_commit_with_blob(
            blob_repo, commit_repo, "head_v1_" + "a" * 56, sample_tract_id
        )
        c2 = self._make_commit_with_blob(
            blob_repo, commit_repo, "head_v2_" + "b" * 56, sample_tract_id
        )
        ref_repo.update_head(sample_tract_id, c1.commit_hash)
        ref_repo.update_head(sample_tract_id, c2.commit_hash)
        assert ref_repo.get_head(sample_tract_id) == c2.commit_hash

    def test_branch_operations(self, ref_repo, blob_repo, commit_repo, sample_tract_id):
        commit = self._make_commit_with_blob(
            blob_repo, commit_repo, "branch_commit_" + "a" * 50, sample_tract_id
        )
        # Set branch
        ref_repo.set_branch(sample_tract_id, "main", commit.commit_hash)
        assert ref_repo.get_branch(sample_tract_id, "main") == commit.commit_hash

        # List branches
        branches = ref_repo.list_branches(sample_tract_id)
        assert "main" in branches

    def test_get_nonexistent_branch(self, ref_repo, sample_tract_id):
        assert ref_repo.get_branch(sample_tract_id, "nonexistent") is None

    def test_multiple_branches(self, ref_repo, blob_repo, commit_repo, sample_tract_id):
        c1 = self._make_commit_with_blob(
            blob_repo, commit_repo, "br_main_" + "a" * 56, sample_tract_id
        )
        c2 = self._make_commit_with_blob(
            blob_repo, commit_repo, "br_dev_" + "b" * 57, sample_tract_id
        )
        ref_repo.set_branch(sample_tract_id, "main", c1.commit_hash)
        ref_repo.set_branch(sample_tract_id, "dev", c2.commit_hash)

        branches = ref_repo.list_branches(sample_tract_id)
        assert set(branches) == {"main", "dev"}
        assert ref_repo.get_branch(sample_tract_id, "main") == c1.commit_hash
        assert ref_repo.get_branch(sample_tract_id, "dev") == c2.commit_hash


# ---------------------------------------------------------------------------
# Annotation Repository
# ---------------------------------------------------------------------------


class TestSqliteAnnotationRepository:
    def _make_commit_with_blob(self, blob_repo, commit_repo, commit_hash, tract_id):
        blob = _make_blob(f"blob_{commit_hash}"[:64])
        blob_repo.save_if_absent(blob)
        commit = _make_commit(commit_hash, tract_id, blob.content_hash)
        commit_repo.save(commit)
        return commit

    def test_get_latest_none(self, annotation_repo):
        result = annotation_repo.get_latest("nonexistent_" + "0" * 52)
        assert result is None

    def test_save_and_get_latest(self, annotation_repo, blob_repo, commit_repo, sample_tract_id):
        commit = self._make_commit_with_blob(
            blob_repo, commit_repo, "ann_commit_" + "a" * 53, sample_tract_id
        )
        now = datetime.now(timezone.utc)

        a1 = AnnotationRow(
            tract_id=sample_tract_id,
            target_hash=commit.commit_hash,
            priority=Priority.NORMAL,
            created_at=now,
        )
        annotation_repo.save(a1)

        a2 = AnnotationRow(
            tract_id=sample_tract_id,
            target_hash=commit.commit_hash,
            priority=Priority.PINNED,
            reason="Important content",
            created_at=now + timedelta(seconds=1),
        )
        annotation_repo.save(a2)

        latest = annotation_repo.get_latest(commit.commit_hash)
        assert latest is not None
        assert latest.priority == Priority.PINNED
        assert latest.reason == "Important content"

    def test_get_history(self, annotation_repo, blob_repo, commit_repo, sample_tract_id):
        commit = self._make_commit_with_blob(
            blob_repo, commit_repo, "hist_commit_" + "a" * 52, sample_tract_id
        )
        now = datetime.now(timezone.utc)

        for i, priority in enumerate([Priority.NORMAL, Priority.PINNED, Priority.SKIP]):
            ann = AnnotationRow(
                tract_id=sample_tract_id,
                target_hash=commit.commit_hash,
                priority=priority,
                created_at=now + timedelta(seconds=i),
            )
            annotation_repo.save(ann)

        history = annotation_repo.get_history(commit.commit_hash)
        assert len(history) == 3
        assert history[0].priority == Priority.NORMAL
        assert history[1].priority == Priority.PINNED
        assert history[2].priority == Priority.SKIP

    def test_batch_get_latest(self, annotation_repo, blob_repo, commit_repo, sample_tract_id):
        """batch_get_latest returns latest annotation per target."""
        now = datetime.now(timezone.utc)

        # Create two commits
        c1 = self._make_commit_with_blob(
            blob_repo, commit_repo, "batch_c1_" + "a" * 55, sample_tract_id
        )
        c2 = self._make_commit_with_blob(
            blob_repo, commit_repo, "batch_c2_" + "b" * 55, sample_tract_id
        )

        # Annotate c1 twice
        annotation_repo.save(AnnotationRow(
            tract_id=sample_tract_id,
            target_hash=c1.commit_hash,
            priority=Priority.NORMAL,
            created_at=now,
        ))
        annotation_repo.save(AnnotationRow(
            tract_id=sample_tract_id,
            target_hash=c1.commit_hash,
            priority=Priority.PINNED,
            created_at=now + timedelta(seconds=1),
        ))

        # Annotate c2 once
        annotation_repo.save(AnnotationRow(
            tract_id=sample_tract_id,
            target_hash=c2.commit_hash,
            priority=Priority.SKIP,
            created_at=now,
        ))

        result = annotation_repo.batch_get_latest([c1.commit_hash, c2.commit_hash])
        assert len(result) == 2
        assert result[c1.commit_hash].priority == Priority.PINNED
        assert result[c2.commit_hash].priority == Priority.SKIP

    def test_batch_get_latest_empty(self, annotation_repo):
        result = annotation_repo.batch_get_latest([])
        assert result == {}

    def test_batch_get_latest_no_annotations(self, annotation_repo):
        """Commits with no annotations are omitted from result."""
        result = annotation_repo.batch_get_latest(["nonexistent_" + "0" * 52])
        assert result == {}


# ---------------------------------------------------------------------------
# Commit Repository: get_by_config
# ---------------------------------------------------------------------------


class TestSqliteCommitRepositoryGetByConfig:
    """Unit tests for SqliteCommitRepository.get_by_config()."""

    def _setup_blob(self, blob_repo, content_hash="cfg_blob_" + "0" * 55):
        blob = _make_blob(content_hash)
        blob_repo.save_if_absent(blob)
        return blob

    def test_get_by_config_equality(self, commit_repo, blob_repo, session, sample_tract_id):
        blob = self._setup_blob(blob_repo)
        now = datetime.now(timezone.utc)

        c1 = _make_commit("cfg_eq1_" + "a" * 56, sample_tract_id, blob.content_hash, created_at=now)
        c1.generation_config_json = {"model": "gpt-4o", "temperature": 0.7}
        c2 = _make_commit("cfg_eq2_" + "b" * 56, sample_tract_id, blob.content_hash,
                          parent_hash=c1.commit_hash, created_at=now + timedelta(seconds=1))
        c2.generation_config_json = {"model": "claude-3", "temperature": 0.5}

        commit_repo.save(c1)
        commit_repo.save(c2)

        results = commit_repo.get_by_config(sample_tract_id, "model", "=", "gpt-4o")
        assert len(results) == 1
        assert results[0].commit_hash == c1.commit_hash

    def test_get_by_config_greater_than(self, commit_repo, blob_repo, session, sample_tract_id):
        blob = self._setup_blob(blob_repo, "cfg_gt_blob_" + "0" * 52)
        now = datetime.now(timezone.utc)

        c1 = _make_commit("cfg_gt1_" + "a" * 56, sample_tract_id, blob.content_hash, created_at=now)
        c1.generation_config_json = {"temperature": 0.3}
        c2 = _make_commit("cfg_gt2_" + "b" * 56, sample_tract_id, blob.content_hash,
                          parent_hash=c1.commit_hash, created_at=now + timedelta(seconds=1))
        c2.generation_config_json = {"temperature": 0.9}

        commit_repo.save(c1)
        commit_repo.save(c2)

        results = commit_repo.get_by_config(sample_tract_id, "temperature", ">", 0.5)
        assert len(results) == 1
        assert results[0].commit_hash == c2.commit_hash

    def test_get_by_config_no_matches(self, commit_repo, blob_repo, session, sample_tract_id):
        blob = self._setup_blob(blob_repo, "cfg_nm_blob_" + "0" * 52)
        now = datetime.now(timezone.utc)

        c1 = _make_commit("cfg_nm1_" + "a" * 56, sample_tract_id, blob.content_hash, created_at=now)
        c1.generation_config_json = {"temperature": 0.3}
        commit_repo.save(c1)

        results = commit_repo.get_by_config(sample_tract_id, "temperature", ">", 0.9)
        assert len(results) == 0

    def test_get_by_config_invalid_operator_raises(self, commit_repo, blob_repo, session, sample_tract_id):
        with pytest.raises(ValueError, match="Unsupported operator"):
            commit_repo.get_by_config(sample_tract_id, "temperature", "LIKE", 0.5)

    def test_get_by_config_null_config_excluded(self, commit_repo, blob_repo, session, sample_tract_id):
        """Commits with NULL generation_config_json are not returned."""
        blob = self._setup_blob(blob_repo, "cfg_null_blob" + "0" * 51)
        now = datetime.now(timezone.utc)

        c1 = _make_commit("cfg_null1" + "a" * 55, sample_tract_id, blob.content_hash, created_at=now)
        # No generation_config_json set (NULL)
        c2 = _make_commit("cfg_null2" + "b" * 55, sample_tract_id, blob.content_hash,
                          parent_hash=c1.commit_hash, created_at=now + timedelta(seconds=1))
        c2.generation_config_json = {"temperature": 0.7}

        commit_repo.save(c1)
        commit_repo.save(c2)

        results = commit_repo.get_by_config(sample_tract_id, "temperature", "=", 0.7)
        assert len(results) == 1
        assert results[0].commit_hash == c2.commit_hash
