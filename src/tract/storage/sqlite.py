"""SQLite implementations of repository interfaces.

All repositories use SQLAlchemy 2.0-style queries (select() + session.execute()).
Each repository takes a Session in its constructor.
"""

from __future__ import annotations

from datetime import datetime
from collections.abc import Sequence

from sqlalchemy import delete, func, select, update, and_, or_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from tract.storage.repositories import (
    AnnotationRepository,
    BlobRepository,
    CommitParentRepository,
    CommitRepository,
    CompileRecordRepository,
    OperationEventRepository,
    PersistenceRepository,
    TagAnnotationRepository,
    TagRegistryRepository,
    RefRepository,
    SpawnPointerRepository,
    ToolSchemaRepository,
)
from tract.storage.schema import (
    AnnotationRow,
    BlobRow,
    CommitParentRow,
    CommitRow,
    CommitToolRow,
    CompileEffectiveRow,
    CompileRecordRow,
    ConfigChangeRow,
    OperationCommitRow,
    OperationConfigRow,
    OperationEventRow,
    TagAnnotationRow,
    TagRegistryRow,
    RefRow,
    SpawnPointerRow,
    ToolSchemaRow,
)


class SqliteBlobRepository(BlobRepository):
    """SQLite implementation of blob repository.

    Content-addressable: save_if_absent checks existence before insert.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, content_hash: str) -> BlobRow | None:
        stmt = select(BlobRow).where(BlobRow.content_hash == content_hash)
        return self._session.execute(stmt).scalar_one_or_none()

    def save_if_absent(self, blob: BlobRow) -> None:
        """Store blob only if content_hash not already present (dedup).

        Uses INSERT OR IGNORE to avoid race conditions when multiple
        threads attempt to insert the same content-addressed blob
        concurrently.
        """
        stmt = sqlite_insert(BlobRow).values(
            content_hash=blob.content_hash,
            payload_json=blob.payload_json,
            byte_size=blob.byte_size,
            token_count=blob.token_count,
            created_at=blob.created_at,
        ).on_conflict_do_nothing(index_elements=["content_hash"])
        self._session.execute(stmt)
        self._session.flush()

    def batch_get(self, content_hashes: list[str]) -> dict[str, BlobRow]:
        """Get multiple blobs by content hash in a single query."""
        if not content_hashes:
            return {}
        stmt = select(BlobRow).where(BlobRow.content_hash.in_(content_hashes))
        rows = self._session.execute(stmt).scalars().all()
        return {row.content_hash: row for row in rows}

    def delete_if_orphaned(self, content_hash: str) -> bool:
        """Delete a blob if no commit still references it.

        Returns True if deleted, False if still referenced.
        """
        # Check if any commit still references this content_hash
        ref_stmt = select(CommitRow).where(
            CommitRow.content_hash == content_hash
        ).limit(1)
        if self._session.execute(ref_stmt).first() is not None:
            return False

        blob = self.get(content_hash)
        if blob is not None:
            self._session.delete(blob)
            self._session.flush()
            return True
        return False


class SqliteCommitRepository(CommitRepository):
    """SQLite implementation of commit repository."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, commit_hash: str) -> CommitRow | None:
        stmt = select(CommitRow).where(CommitRow.commit_hash == commit_hash)
        return self._session.execute(stmt).scalar_one_or_none()

    def save(self, commit: CommitRow) -> None:
        self._session.add(commit)
        self._session.flush()

    def get_ancestors(
        self,
        commit_hash: str,
        limit: int | None = None,
        *,
        op_filter: object | None = None,
    ) -> Sequence[CommitRow]:
        """Walk parent chain from commit to root.

        Args:
            commit_hash: Starting commit hash.
            limit: Maximum number of matching commits to return.
            op_filter: If set (a CommitOperation value), only include commits
                whose operation matches.  The chain walk continues through
                non-matching commits so all matching ancestors are found.

        Returns commits in reverse chronological order (newest first).

        Implementation: walks the parent chain one commit at a time via
        primary-key lookups (O(1) each). Only fetches chain-length rows
        instead of all commits in the tract.
        """
        ancestors: list[CommitRow] = []
        current_hash: str | None = commit_hash

        while current_hash is not None:
            if limit is not None and len(ancestors) >= limit:
                break
            commit = self.get(current_hash)
            if commit is None:
                break
            if op_filter is None or commit.operation == op_filter:
                ancestors.append(commit)
            current_hash = commit.parent_hash

        return ancestors

    def sum_ancestor_tokens(self, commit_hash: str) -> int:
        """Sum token_count for the ancestor chain using a recursive CTE."""
        from sqlalchemy import text

        result = self._session.execute(
            text(
                "WITH RECURSIVE ancestors(commit_hash, parent_hash, token_count) AS ("
                "  SELECT commit_hash, parent_hash, token_count"
                "  FROM commits WHERE commit_hash = :start_hash"
                "  UNION ALL"
                "  SELECT c.commit_hash, c.parent_hash, c.token_count"
                "  FROM commits c"
                "  JOIN ancestors a ON c.commit_hash = a.parent_hash"
                ") SELECT COALESCE(SUM(token_count), 0) FROM ancestors"
            ),
            {"start_hash": commit_hash},
        ).scalar()
        return int(result)  # type: ignore[arg-type]

    def get_by_type(self, content_type: str, tract_id: str) -> Sequence[CommitRow]:
        stmt = (
            select(CommitRow)
            .where(CommitRow.tract_id == tract_id, CommitRow.content_type == content_type)
            .order_by(CommitRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_children(self, commit_hash: str) -> Sequence[CommitRow]:
        stmt = select(CommitRow).where(CommitRow.parent_hash == commit_hash)
        return list(self._session.execute(stmt).scalars().all())

    def get_by_prefix(self, prefix: str, tract_id: str | None = None) -> CommitRow | None:
        """Find commit by hash prefix (min 4 chars).

        Raises AmbiguousPrefixError if multiple matches.
        Returns None if no match.
        """
        from tract.exceptions import AmbiguousPrefixError

        if len(prefix) < 4:
            raise ValueError("Commit hash prefix must be at least 4 characters")

        conditions = [CommitRow.commit_hash.startswith(prefix)]
        if tract_id is not None:
            conditions.append(CommitRow.tract_id == tract_id)

        stmt = select(CommitRow).where(and_(*conditions))
        results = list(self._session.execute(stmt).scalars().all())

        if len(results) == 0:
            return None
        if len(results) == 1:
            return results[0]
        raise AmbiguousPrefixError(prefix, [r.commit_hash for r in results])

    def get_by_config(
        self, tract_id: str, json_path: str, operator: str, value: object
    ) -> Sequence[CommitRow]:
        return self.get_by_config_multi(tract_id, [(json_path, operator, value)])

    def get_by_config_multi(
        self, tract_id: str, conditions: list[tuple[str, str, object]]
    ) -> Sequence[CommitRow]:
        where_clauses = [CommitRow.tract_id == tract_id]
        ops = {
            "=": lambda e, v: e == v,
            "!=": lambda e, v: e != v,
            ">": lambda e, v: e > v,
            "<": lambda e, v: e < v,
            ">=": lambda e, v: e >= v,
            "<=": lambda e, v: e <= v,
            "in": lambda e, v: e.in_(v),
            "not in": lambda e, v: e.not_in(v),
            "between": lambda e, v: and_(e >= v[0], e <= v[1]),
            "not between": lambda e, v: or_(e < v[0], e > v[1]),
        }
        for json_path, operator, value in conditions:
            if operator not in ops:
                raise ValueError(
                    f"Unsupported operator: {operator}. "
                    f"Use one of: {list(ops.keys())}"
                )
            extracted = CommitRow.generation_config_json[json_path]
            # Cast to the appropriate scalar type for cross-dialect comparison.
            # Without this, the JSON-typed result causes type mismatches on
            # some backends (e.g. SQLite wraps bind values in JSON()).
            # For list-valued operators (in, not in, between, not between),
            # sample the first element to determine the type.
            _sample = value[0] if isinstance(value, (list, tuple)) and value else value
            if isinstance(_sample, bool):
                extracted = extracted.as_boolean()
            elif isinstance(_sample, int):
                extracted = extracted.as_integer()
            elif isinstance(_sample, float):
                extracted = extracted.as_float()
            elif isinstance(_sample, str):
                extracted = extracted.as_string()
            where_clauses.append(ops[operator](extracted, value))
        stmt = (
            select(CommitRow)
            .where(and_(*where_clauses))
            .order_by(CommitRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_all(self, tract_id: str) -> Sequence[CommitRow]:
        """Get all commits for a tract, ordered by created_at ascending."""
        stmt = (
            select(CommitRow)
            .where(CommitRow.tract_id == tract_id)
            .order_by(CommitRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_edits_for(self, commit_hash: str, tract_id: str) -> Sequence[CommitRow]:
        """Get original commit and all its edits, ordered by created_at."""
        stmt = (
            select(CommitRow)
            .where(
                CommitRow.tract_id == tract_id,
                or_(
                    CommitRow.commit_hash == commit_hash,
                    CommitRow.edit_target == commit_hash,
                ),
            )
            .order_by(CommitRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def update_metadata(self, commit_hash: str, metadata: dict) -> None:
        row = self.get(commit_hash)
        if row is not None:
            row.metadata_json = metadata
            self._session.flush()

    def delete(self, commit_hash: str) -> None:
        """Delete a commit by hash. Also cleans up related rows.

        Removes all rows that reference this commit via FK, and nullifies
        parent_hash/edit_target references from other commits, before
        deleting the commit itself.
        """
        # Bulk delete CommitParentRow entries where this commit is child or parent
        self._session.execute(
            delete(CommitParentRow).where(
                (CommitParentRow.commit_hash == commit_hash)
                | (CommitParentRow.parent_hash == commit_hash)
            )
        )

        # Bulk delete AnnotationRow entries referencing this commit
        self._session.execute(
            delete(AnnotationRow).where(AnnotationRow.target_hash == commit_hash)
        )

        # Bulk delete RefRow entries pointing to this commit (e.g., ORIG_HEAD)
        self._session.execute(
            delete(RefRow).where(RefRow.commit_hash == commit_hash)
        )

        # Bulk delete CompileEffectiveRow entries referencing this commit
        self._session.execute(
            delete(CompileEffectiveRow).where(
                CompileEffectiveRow.commit_hash == commit_hash
            )
        )

        # Bulk delete CommitToolRow entries referencing this commit
        self._session.execute(
            delete(CommitToolRow).where(CommitToolRow.commit_hash == commit_hash)
        )

        # Bulk nullify parent_hash on children (SET NULL semantics)
        self._session.execute(
            update(CommitRow)
            .where(CommitRow.parent_hash == commit_hash)
            .values(parent_hash=None)
        )

        # Bulk nullify edit_target references (SET NULL semantics)
        self._session.execute(
            update(CommitRow)
            .where(CommitRow.edit_target == commit_hash)
            .values(edit_target=None)
        )

        # Now delete the commit itself
        self._session.execute(
            delete(CommitRow).where(CommitRow.commit_hash == commit_hash)
        )

        # Expire all to sync identity map with bulk changes above
        self._session.expire_all()
        self._session.flush()


class SqliteRefRepository(RefRepository):
    """SQLite implementation of ref repository.

    HEAD is stored as ref_name="HEAD".  When attached, HEAD has a
    symbolic_target (e.g. "refs/heads/main") and the branch ref stores
    the actual commit hash.  When detached, HEAD stores commit_hash
    directly with symbolic_target=None.

    Branches are stored as ref_name="refs/heads/{name}".
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _get_ref_row(self, tract_id: str, ref_name: str) -> RefRow | None:
        """Get a ref row by tract_id and ref_name."""
        stmt = select(RefRow).where(
            RefRow.tract_id == tract_id, RefRow.ref_name == ref_name
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_head(self, tract_id: str) -> str | None:
        """Get the HEAD commit hash, resolving symbolic refs."""
        head_ref = self._get_ref_row(tract_id, "HEAD")
        if head_ref is None:
            return None

        # If HEAD is symbolic (attached), resolve through branch ref
        if head_ref.symbolic_target:
            branch_ref = self._get_ref_row(tract_id, head_ref.symbolic_target)
            return branch_ref.commit_hash if branch_ref else None

        # Detached HEAD: commit_hash stored directly
        return head_ref.commit_hash

    def update_head(self, tract_id: str, commit_hash: str) -> None:
        """Update HEAD to point at a new commit.

        Backward compatible: existing callers (CommitEngine) call this
        when creating commits.  Behavior:
        - No HEAD exists: create symbolic HEAD -> refs/heads/main + branch ref
        - Attached HEAD: update the target branch ref
        - Detached HEAD: update commit_hash directly
        """
        head_ref = self._get_ref_row(tract_id, "HEAD")

        if head_ref is None:
            # First commit: create symbolic HEAD -> refs/heads/main
            self._session.add(
                RefRow(
                    tract_id=tract_id,
                    ref_name="HEAD",
                    commit_hash=None,
                    symbolic_target="refs/heads/main",
                )
            )
            # Also create the branch ref
            self._session.add(
                RefRow(
                    tract_id=tract_id,
                    ref_name="refs/heads/main",
                    commit_hash=commit_hash,
                )
            )
        elif head_ref.symbolic_target:
            # Attached HEAD: update the branch ref
            branch_ref = self._get_ref_row(tract_id, head_ref.symbolic_target)
            if branch_ref is None:
                self._session.add(
                    RefRow(
                        tract_id=tract_id,
                        ref_name=head_ref.symbolic_target,
                        commit_hash=commit_hash,
                    )
                )
            else:
                branch_ref.commit_hash = commit_hash
        else:
            # Detached HEAD: update commit_hash directly
            head_ref.commit_hash = commit_hash

        self._session.flush()

    def get_branch(self, tract_id: str, branch_name: str) -> str | None:
        ref_name = f"refs/heads/{branch_name}"
        ref = self._get_ref_row(tract_id, ref_name)
        return ref.commit_hash if ref else None

    def set_branch(self, tract_id: str, branch_name: str, commit_hash: str) -> None:
        ref_name = f"refs/heads/{branch_name}"
        ref = self._get_ref_row(tract_id, ref_name)
        if ref is None:
            self._session.add(
                RefRow(tract_id=tract_id, ref_name=ref_name, commit_hash=commit_hash)
            )
        else:
            ref.commit_hash = commit_hash
        self._session.flush()

    def list_branches(self, tract_id: str) -> list[str]:
        prefix = "refs/heads/"
        stmt = select(RefRow).where(
            RefRow.tract_id == tract_id,
            RefRow.ref_name.startswith(prefix),
        )
        refs = self._session.execute(stmt).scalars().all()
        return [ref.ref_name[len(prefix):] for ref in refs]

    def is_detached(self, tract_id: str) -> bool:
        """Check if HEAD is in detached state."""
        head_ref = self._get_ref_row(tract_id, "HEAD")
        if head_ref is None:
            return False  # No HEAD yet = not detached
        return head_ref.symbolic_target is None

    def attach_head(self, tract_id: str, branch_name: str) -> None:
        """Attach HEAD to a branch (symbolic ref)."""
        ref_name = f"refs/heads/{branch_name}"
        head_ref = self._get_ref_row(tract_id, "HEAD")
        if head_ref is None:
            self._session.add(
                RefRow(
                    tract_id=tract_id,
                    ref_name="HEAD",
                    commit_hash=None,
                    symbolic_target=ref_name,
                )
            )
        else:
            head_ref.symbolic_target = ref_name
            head_ref.commit_hash = None
        self._session.flush()

    def detach_head(self, tract_id: str, commit_hash: str) -> None:
        """Detach HEAD to point directly at a commit hash."""
        head_ref = self._get_ref_row(tract_id, "HEAD")
        if head_ref is None:
            self._session.add(
                RefRow(
                    tract_id=tract_id,
                    ref_name="HEAD",
                    commit_hash=commit_hash,
                    symbolic_target=None,
                )
            )
        else:
            head_ref.commit_hash = commit_hash
            head_ref.symbolic_target = None
        self._session.flush()

    def get_ref(self, tract_id: str, ref_name: str) -> str | None:
        """Get the commit hash for a named ref."""
        ref = self._get_ref_row(tract_id, ref_name)
        return ref.commit_hash if ref else None

    def set_ref(self, tract_id: str, ref_name: str, commit_hash: str) -> None:
        """Set or update a named ref to point at a commit hash."""
        ref = self._get_ref_row(tract_id, ref_name)
        if ref is None:
            self._session.add(
                RefRow(tract_id=tract_id, ref_name=ref_name, commit_hash=commit_hash)
            )
        else:
            ref.commit_hash = commit_hash
        self._session.flush()

    def delete_ref(self, tract_id: str, ref_name: str) -> None:
        """Delete a named ref. No-op if ref doesn't exist."""
        ref = self._get_ref_row(tract_id, ref_name)
        if ref is not None:
            self._session.delete(ref)
            self._session.flush()

    def set_symbolic_ref(self, tract_id: str, ref_name: str, symbolic_target: str) -> None:
        """Set a ref to point symbolically (commit_hash=None, symbolic_target set)."""
        ref = self._get_ref_row(tract_id, ref_name)
        if ref is None:
            self._session.add(
                RefRow(
                    tract_id=tract_id,
                    ref_name=ref_name,
                    commit_hash=None,
                    symbolic_target=symbolic_target,
                )
            )
        else:
            ref.commit_hash = None
            ref.symbolic_target = symbolic_target
        self._session.flush()

    def get_symbolic_ref(self, tract_id: str, ref_name: str) -> str | None:
        """Get the symbolic target of a ref. Returns None if not found or not symbolic."""
        ref = self._get_ref_row(tract_id, ref_name)
        return ref.symbolic_target if ref else None

    def get_current_branch(self, tract_id: str) -> str | None:
        """Get the current branch name if HEAD is attached."""
        head_ref = self._get_ref_row(tract_id, "HEAD")
        if head_ref is None or head_ref.symbolic_target is None:
            return None
        prefix = "refs/heads/"
        if head_ref.symbolic_target.startswith(prefix):
            return head_ref.symbolic_target[len(prefix):]
        return None


class SqliteCommitParentRepository(CommitParentRepository):
    """SQLite implementation of multi-parent commit storage."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_parent(self, commit_hash: str, parent_hash: str, position: int) -> None:
        self._session.add(
            CommitParentRow(
                commit_hash=commit_hash,
                parent_hash=parent_hash,
                position=position,
            )
        )
        self._session.flush()

    def get_parents(self, commit_hash: str) -> list[str]:
        stmt = (
            select(CommitParentRow.parent_hash)
            .where(CommitParentRow.commit_hash == commit_hash)
            .order_by(CommitParentRow.position)
        )
        return list(self._session.execute(stmt).scalars().all())

    def add_parents(self, commit_hash: str, parent_hashes: list[str]) -> None:
        for i, ph in enumerate(parent_hashes):
            self._session.add(
                CommitParentRow(
                    commit_hash=commit_hash,
                    parent_hash=ph,
                    position=i,
                )
            )
        self._session.flush()


class SqliteAnnotationRepository(AnnotationRepository):
    """SQLite implementation of annotation repository.

    Annotations are append-only. The latest annotation for a given
    target_hash (by created_at) is the current priority.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_latest(self, target_hash: str) -> AnnotationRow | None:
        stmt = (
            select(AnnotationRow)
            .where(AnnotationRow.target_hash == target_hash)
            .order_by(AnnotationRow.created_at.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def save(self, annotation: AnnotationRow) -> None:
        self._session.add(annotation)
        self._session.flush()

    def get_history(self, target_hash: str) -> Sequence[AnnotationRow]:
        stmt = (
            select(AnnotationRow)
            .where(AnnotationRow.target_hash == target_hash)
            .order_by(AnnotationRow.created_at.asc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def batch_get_latest(self, target_hashes: list[str]) -> dict[str, AnnotationRow]:
        """Get latest annotation per target using a single query with subquery."""
        if not target_hashes:
            return {}

        # Subquery: max created_at per target_hash
        max_time_subq = (
            select(
                AnnotationRow.target_hash,
                func.max(AnnotationRow.created_at).label("max_created_at"),
            )
            .where(AnnotationRow.target_hash.in_(target_hashes))
            .group_by(AnnotationRow.target_hash)
            .subquery()
        )

        # Join to get full rows
        stmt = (
            select(AnnotationRow)
            .join(
                max_time_subq,
                (AnnotationRow.target_hash == max_time_subq.c.target_hash)
                & (AnnotationRow.created_at == max_time_subq.c.max_created_at),
            )
        )

        rows = self._session.execute(stmt).scalars().all()
        return {row.target_hash: row for row in rows}


class SqliteOperationEventRepository(OperationEventRepository):
    """SQLite implementation of unified operation event storage.

    Tracks operation events (compress, reorganize, import) and their
    associated source/result commits.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_event(
        self,
        event_id: str,
        tract_id: str,
        event_type: str,
        branch_name: str | None,
        created_at: datetime,
        original_tokens: int,
        compressed_tokens: int,
        params_json: dict | None,
    ) -> None:
        row = OperationEventRow(
            event_id=event_id,
            tract_id=tract_id,
            event_type=event_type,
            branch_name=branch_name,
            created_at=created_at,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            params_json=params_json,
        )
        self._session.add(row)
        self._session.flush()

    def add_commit(
        self, event_id: str, commit_hash: str, role: str, position: int
    ) -> None:
        row = OperationCommitRow(
            event_id=event_id,
            commit_hash=commit_hash,
            role=role,
            position=position,
        )
        self._session.add(row)
        self._session.flush()

    def get_event(self, event_id: str) -> OperationEventRow | None:
        stmt = select(OperationEventRow).where(
            OperationEventRow.event_id == event_id
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_commits(
        self, event_id: str, role: str | None = None
    ) -> list[OperationCommitRow]:
        conditions = [OperationCommitRow.event_id == event_id]
        if role is not None:
            conditions.append(OperationCommitRow.role == role)
        stmt = (
            select(OperationCommitRow)
            .where(and_(*conditions))
            .order_by(OperationCommitRow.position)
        )
        return list(self._session.execute(stmt).scalars().all())

    def is_source_of(self, commit_hash: str) -> bool:
        stmt = select(OperationCommitRow).where(
            OperationCommitRow.commit_hash == commit_hash,
            OperationCommitRow.role == "source",
        )
        return self._session.execute(stmt).first() is not None

    def get_all_source_hashes(self, tract_id: str) -> set[str]:
        stmt = (
            select(OperationCommitRow.commit_hash)
            .join(
                OperationEventRow,
                OperationCommitRow.event_id == OperationEventRow.event_id,
            )
            .where(
                OperationEventRow.tract_id == tract_id,
                OperationCommitRow.role == "source",
            )
        )
        return set(self._session.execute(stmt).scalars().all())

    def get_all_ids(self, tract_id: str) -> list[str]:
        stmt = select(OperationEventRow.event_id).where(
            OperationEventRow.tract_id == tract_id
        )
        return list(self._session.execute(stmt).scalars().all())

    def delete_commit(self, commit_hash: str) -> None:
        """Delete all OperationCommitRow entries for a commit hash."""
        self._session.execute(
            delete(OperationCommitRow).where(
                OperationCommitRow.commit_hash == commit_hash
            )
        )
        self._session.expire_all()
        self._session.flush()

    def delete_event(self, event_id: str) -> None:
        """Delete an event and all its commit associations."""
        # Bulk delete commit associations first
        self._session.execute(
            delete(OperationCommitRow).where(
                OperationCommitRow.event_id == event_id
            )
        )
        # Bulk delete the event itself
        self._session.execute(
            delete(OperationEventRow).where(
                OperationEventRow.event_id == event_id
            )
        )
        self._session.expire_all()
        self._session.flush()


class SqliteCompileRecordRepository(CompileRecordRepository):
    """SQLite implementation of compile record storage.

    Tracks compile operations and their effective commits.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_record(
        self,
        record_id: str,
        tract_id: str,
        head_hash: str,
        token_count: int,
        commit_count: int,
        token_source: str,
        params_json: dict | None,
        created_at: datetime,
    ) -> None:
        row = CompileRecordRow(
            record_id=record_id,
            tract_id=tract_id,
            head_hash=head_hash,
            token_count=token_count,
            commit_count=commit_count,
            token_source=token_source,
            params_json=params_json,
            created_at=created_at,
        )
        self._session.add(row)
        self._session.flush()

    def add_effective(
        self, record_id: str, commit_hash: str, position: int
    ) -> None:
        row = CompileEffectiveRow(
            record_id=record_id,
            commit_hash=commit_hash,
            position=position,
        )
        self._session.add(row)
        self._session.flush()

    def get_record(self, record_id: str) -> CompileRecordRow | None:
        stmt = select(CompileRecordRow).where(
            CompileRecordRow.record_id == record_id
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_all(self, tract_id: str) -> list[CompileRecordRow]:
        stmt = (
            select(CompileRecordRow)
            .where(CompileRecordRow.tract_id == tract_id)
            .order_by(CompileRecordRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_effectives(self, record_id: str) -> list[CompileEffectiveRow]:
        stmt = (
            select(CompileEffectiveRow)
            .where(CompileEffectiveRow.record_id == record_id)
            .order_by(CompileEffectiveRow.position)
        )
        return list(self._session.execute(stmt).scalars().all())


class SqliteSpawnPointerRepository(SpawnPointerRepository):
    """SQLite implementation of spawn pointer storage.

    Tracks parent-child relationships between tracts in the spawn tree.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self,
        parent_tract_id: str,
        parent_commit_hash: str | None,
        child_tract_id: str,
        purpose: str,
        inheritance_mode: str,
        display_name: str | None,
        created_at: datetime,
    ) -> SpawnPointerRow:
        row = SpawnPointerRow(
            parent_tract_id=parent_tract_id,
            parent_commit_hash=parent_commit_hash,
            child_tract_id=child_tract_id,
            purpose=purpose,
            inheritance_mode=inheritance_mode,
            display_name=display_name,
            created_at=created_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def get(self, id: int) -> SpawnPointerRow | None:
        stmt = select(SpawnPointerRow).where(SpawnPointerRow.id == id)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_child(self, child_tract_id: str) -> SpawnPointerRow | None:
        stmt = select(SpawnPointerRow).where(
            SpawnPointerRow.child_tract_id == child_tract_id
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_children(self, parent_tract_id: str) -> list[SpawnPointerRow]:
        stmt = (
            select(SpawnPointerRow)
            .where(SpawnPointerRow.parent_tract_id == parent_tract_id)
            .order_by(SpawnPointerRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_all(self, tract_id: str) -> list[SpawnPointerRow]:
        stmt = select(SpawnPointerRow).where(
            (SpawnPointerRow.parent_tract_id == tract_id)
            | (SpawnPointerRow.child_tract_id == tract_id)
        )
        return list(self._session.execute(stmt).scalars().all())

    def has_ancestor(self, child_tract_id: str, potential_ancestor: str) -> bool:
        """Walk up the spawn tree to check ancestry.

        Iteratively follows parent pointers from child_tract_id.
        Returns True if potential_ancestor is found at any level.
        Terminates when root is reached (no parent pointer) or cycle detected.
        """
        visited: set[str] = set()
        current = child_tract_id

        while True:
            if current in visited:
                # Cycle detected -- stop walking
                return False
            visited.add(current)

            pointer = self.get_by_child(current)
            if pointer is None:
                # Reached root of spawn tree
                return False

            if pointer.parent_tract_id == potential_ancestor:
                return True

            current = pointer.parent_tract_id


class SqliteToolSchemaRepository(ToolSchemaRepository):
    """SQLite implementation of tool schema storage.

    Content-addressed: store() checks existence before insert.
    get_for_commit() joins through CommitToolRow ordered by position.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def store(
        self, content_hash: str, name: str, schema: dict, created_at: datetime
    ) -> ToolSchemaRow:
        """Store a tool schema (idempotent)."""
        existing = self.get(content_hash)
        if existing is not None:
            return existing
        row = ToolSchemaRow(
            content_hash=content_hash,
            name=name,
            schema_json=schema,
            created_at=created_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def get(self, content_hash: str) -> ToolSchemaRow | None:
        stmt = select(ToolSchemaRow).where(
            ToolSchemaRow.content_hash == content_hash
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_name(self, name: str) -> Sequence[ToolSchemaRow]:
        stmt = (
            select(ToolSchemaRow)
            .where(ToolSchemaRow.name == name)
            .order_by(ToolSchemaRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_for_commit(self, commit_hash: str) -> Sequence[ToolSchemaRow]:
        """Get tool schemas for a commit, ordered by position."""
        stmt = (
            select(ToolSchemaRow)
            .join(
                CommitToolRow,
                ToolSchemaRow.content_hash == CommitToolRow.tool_hash,
            )
            .where(CommitToolRow.commit_hash == commit_hash)
            .order_by(CommitToolRow.position)
        )
        return list(self._session.execute(stmt).scalars().all())

    def link_to_commit(
        self, commit_hash: str, tool_hash: str, position: int
    ) -> None:
        row = CommitToolRow(
            commit_hash=commit_hash,
            tool_hash=tool_hash,
            position=position,
        )
        self._session.add(row)
        self._session.flush()

    def get_commit_tool_hashes(self, commit_hash: str) -> Sequence[str]:
        stmt = (
            select(CommitToolRow.tool_hash)
            .where(CommitToolRow.commit_hash == commit_hash)
            .order_by(CommitToolRow.position)
        )
        return list(self._session.execute(stmt).scalars().all())


class SqliteTagAnnotationRepository(TagAnnotationRepository):
    """SQLite implementation of mutable tag annotation storage."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_tag(
        self, tract_id: str, target_hash: str, tag: str, created_at: datetime
    ) -> TagAnnotationRow:
        row = TagAnnotationRow(
            tract_id=tract_id,
            target_hash=target_hash,
            tag=tag,
            created_at=created_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def remove_tag(self, tract_id: str, target_hash: str, tag: str) -> bool:
        stmt = select(TagAnnotationRow).where(
            and_(
                TagAnnotationRow.tract_id == tract_id,
                TagAnnotationRow.target_hash == target_hash,
                TagAnnotationRow.tag == tag,
            )
        )
        rows = list(self._session.execute(stmt).scalars().all())
        if not rows:
            return False
        for row in rows:
            self._session.delete(row)
        self._session.flush()
        return True

    def get_tags(self, target_hash: str) -> list[str]:
        stmt = (
            select(TagAnnotationRow.tag)
            .where(TagAnnotationRow.target_hash == target_hash)
            .order_by(TagAnnotationRow.created_at)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_commits_by_tag(self, tract_id: str, tag: str) -> list[str]:
        stmt = (
            select(TagAnnotationRow.target_hash)
            .where(
                and_(
                    TagAnnotationRow.tract_id == tract_id,
                    TagAnnotationRow.tag == tag,
                )
            )
            .distinct()
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_commits_by_tags(
        self, tract_id: str, tags: list[str], match: str = "any"
    ) -> list[str]:
        if not tags:
            return []

        if match == "any":
            stmt = (
                select(TagAnnotationRow.target_hash)
                .where(
                    and_(
                        TagAnnotationRow.tract_id == tract_id,
                        TagAnnotationRow.tag.in_(tags),
                    )
                )
                .distinct()
            )
            return list(self._session.execute(stmt).scalars().all())
        else:
            # "all" -- must have every tag
            stmt = (
                select(TagAnnotationRow.target_hash)
                .where(
                    and_(
                        TagAnnotationRow.tract_id == tract_id,
                        TagAnnotationRow.tag.in_(tags),
                    )
                )
                .group_by(TagAnnotationRow.target_hash)
                .having(func.count(func.distinct(TagAnnotationRow.tag)) == len(tags))
            )
            return list(self._session.execute(stmt).scalars().all())


class SqliteTagRegistryRepository(TagRegistryRepository):
    """SQLite implementation of tag registry storage."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def register(
        self,
        tract_id: str,
        tag_name: str,
        description: str | None,
        auto_created: bool,
        created_at: datetime,
    ) -> TagRegistryRow:
        # Check if already registered (idempotent)
        existing = self.get(tract_id, tag_name)
        if existing is not None:
            # Update description if changed
            if description is not None and existing.description != description:
                existing.description = description
                self._session.flush()
            return existing
        row = TagRegistryRow(
            tract_id=tract_id,
            tag_name=tag_name,
            description=description,
            auto_created=auto_created,
            created_at=created_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def get(self, tract_id: str, tag_name: str) -> TagRegistryRow | None:
        stmt = select(TagRegistryRow).where(
            and_(
                TagRegistryRow.tract_id == tract_id,
                TagRegistryRow.tag_name == tag_name,
            )
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list_all(self, tract_id: str) -> list[TagRegistryRow]:
        stmt = (
            select(TagRegistryRow)
            .where(TagRegistryRow.tract_id == tract_id)
            .order_by(TagRegistryRow.tag_name)
        )
        return list(self._session.execute(stmt).scalars().all())

    def is_registered(self, tract_id: str, tag_name: str) -> bool:
        return self.get(tract_id, tag_name) is not None

    def delete(self, tract_id: str, tag_name: str) -> bool:
        row = self.get(tract_id, tag_name)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


class SqlitePersistenceRepository(PersistenceRepository):
    """SQLite implementation of persistence repository.

    Provides CRUD for operation configs and config change log.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- Operation configs --

    def save_operation_config(self, config_row: OperationConfigRow) -> OperationConfigRow:
        # Upsert: if config_key already exists for this tract_id, update it
        existing = self.get_operation_config(config_row.tract_id, config_row.config_key)
        if existing is not None:
            existing.config_json = config_row.config_json
            existing.created_at = config_row.created_at
            self._session.flush()
            return existing
        self._session.add(config_row)
        self._session.flush()
        return config_row

    def get_operation_configs(self, tract_id: str) -> list[OperationConfigRow]:
        stmt = (
            select(OperationConfigRow)
            .where(OperationConfigRow.tract_id == tract_id)
            .order_by(OperationConfigRow.id)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_operation_config(
        self, tract_id: str, config_key: str
    ) -> OperationConfigRow | None:
        stmt = select(OperationConfigRow).where(
            and_(
                OperationConfigRow.tract_id == tract_id,
                OperationConfigRow.config_key == config_key,
            )
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def delete_operation_config(self, tract_id: str, config_key: str) -> bool:
        row = self.get_operation_config(tract_id, config_key)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    # -- Config change log --

    def save_config_change(self, change: ConfigChangeRow) -> ConfigChangeRow:
        self._session.add(change)
        self._session.flush()
        return change

    def get_config_changes(
        self,
        tract_id: str,
        *,
        change_type: str | None = None,
        limit: int = 100,
    ) -> list[ConfigChangeRow]:
        stmt = (
            select(ConfigChangeRow)
            .where(ConfigChangeRow.tract_id == tract_id)
        )
        if change_type is not None:
            stmt = stmt.where(ConfigChangeRow.change_type == change_type)
        stmt = stmt.order_by(ConfigChangeRow.created_at.desc()).limit(limit)
        return list(self._session.execute(stmt).scalars().all())
