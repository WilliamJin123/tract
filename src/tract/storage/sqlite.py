"""SQLite implementations of repository interfaces.

All repositories use SQLAlchemy 2.0-style queries (select() + session.execute()).
Each repository takes a Session in its constructor.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import func, select, and_
from sqlalchemy.orm import Session

from tract.storage.repositories import (
    AnnotationRepository,
    BlobRepository,
    CommitParentRepository,
    CommitRepository,
    CompressionRepository,
    RefRepository,
)
from tract.storage.schema import (
    AnnotationRow,
    BlobRow,
    CommitParentRow,
    CommitRow,
    CompressionResultRow,
    CompressionRow,
    CompressionSourceRow,
    RefRow,
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
        """Store blob only if content_hash not already present (dedup)."""
        existing = self.get(blob.content_hash)
        if existing is None:
            self._session.add(blob)
            self._session.flush()

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

        Implementation note: instead of issuing one SQL query per ancestor
        (N+1 pattern), we fetch the starting commit to learn its tract_id,
        then batch-load all commits for that tract into an in-memory dict
        and walk the parent chain there.  This trades 2 queries for N+1.
        """
        # 1. Fetch starting commit to learn its tract_id
        start_commit = self.get(commit_hash)
        if start_commit is None:
            return []

        # 2. Batch-load all commits for this tract into a dict
        stmt = select(CommitRow).where(CommitRow.tract_id == start_commit.tract_id)
        all_commits = self._session.execute(stmt).scalars().all()
        commits_by_hash: dict[str, CommitRow] = {c.commit_hash: c for c in all_commits}

        # 3. Walk parent chain in-memory
        ancestors: list[CommitRow] = []
        current_hash: str | None = commit_hash

        while current_hash is not None:
            if limit is not None and len(ancestors) >= limit:
                break
            commit = commits_by_hash.get(current_hash)
            if commit is None:
                break
            if op_filter is None or commit.operation == op_filter:
                ancestors.append(commit)
            current_hash = commit.parent_hash

        return ancestors

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
        extracted = func.json_extract(
            CommitRow.generation_config_json, f'$.{json_path}'
        )
        ops = {
            "=": lambda e, v: e == v,
            "!=": lambda e, v: e != v,
            ">": lambda e, v: e > v,
            "<": lambda e, v: e < v,
            ">=": lambda e, v: e >= v,
            "<=": lambda e, v: e <= v,
        }
        if operator not in ops:
            raise ValueError(f"Unsupported operator: {operator}. Use one of: {list(ops.keys())}")
        condition = ops[operator](extracted, value)
        stmt = (
            select(CommitRow)
            .where(CommitRow.tract_id == tract_id, condition)
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

    def delete(self, commit_hash: str) -> None:
        """Delete a commit by hash. Also cleans up related rows.

        Removes CommitParentRow, AnnotationRow, RefRow pointing to this
        commit, and nullifies parent_hash/response_to references from
        other commits before deleting the commit itself.
        """
        # Delete CommitParentRow entries where this commit is child or parent
        parent_stmts = select(CommitParentRow).where(
            (CommitParentRow.commit_hash == commit_hash)
            | (CommitParentRow.parent_hash == commit_hash)
        )
        for row in self._session.execute(parent_stmts).scalars().all():
            self._session.delete(row)

        # Delete AnnotationRow entries referencing this commit
        annotation_stmts = select(AnnotationRow).where(
            AnnotationRow.target_hash == commit_hash
        )
        for row in self._session.execute(annotation_stmts).scalars().all():
            self._session.delete(row)

        # Delete RefRow entries pointing to this commit (e.g., ORIG_HEAD)
        ref_stmts = select(RefRow).where(
            RefRow.commit_hash == commit_hash
        )
        for row in self._session.execute(ref_stmts).scalars().all():
            self._session.delete(row)

        # Nullify parent_hash on children (SET NULL semantics)
        child_stmts = select(CommitRow).where(
            CommitRow.parent_hash == commit_hash
        )
        for child in self._session.execute(child_stmts).scalars().all():
            child.parent_hash = None

        # Nullify response_to references (SET NULL semantics)
        resp_stmts = select(CommitRow).where(
            CommitRow.response_to == commit_hash
        )
        for resp in self._session.execute(resp_stmts).scalars().all():
            resp.response_to = None

        # Flush all modifications BEFORE deleting the commit itself
        self._session.flush()

        # Now delete the commit
        commit = self.get(commit_hash)
        if commit is not None:
            self._session.delete(commit)
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


class SqliteCompressionRepository(CompressionRepository):
    """SQLite implementation of compression provenance storage.

    Tracks compression records and their source/result commit associations.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_record(
        self,
        compression_id: str,
        tract_id: str,
        branch_name: str | None,
        created_at: datetime,
        original_tokens: int,
        compressed_tokens: int,
        target_tokens: int | None,
        instructions: str | None,
    ) -> None:
        row = CompressionRow(
            compression_id=compression_id,
            tract_id=tract_id,
            branch_name=branch_name,
            created_at=created_at,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            target_tokens=target_tokens,
            instructions=instructions,
        )
        self._session.add(row)
        self._session.flush()

    def add_source(self, compression_id: str, commit_hash: str, position: int) -> None:
        row = CompressionSourceRow(
            compression_id=compression_id,
            commit_hash=commit_hash,
            position=position,
        )
        self._session.add(row)
        self._session.flush()

    def add_result(self, compression_id: str, commit_hash: str, position: int) -> None:
        row = CompressionResultRow(
            compression_id=compression_id,
            commit_hash=commit_hash,
            position=position,
        )
        self._session.add(row)
        self._session.flush()

    def get_record(self, compression_id: str) -> CompressionRow | None:
        stmt = select(CompressionRow).where(
            CompressionRow.compression_id == compression_id
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_sources(self, compression_id: str) -> list[CompressionSourceRow]:
        stmt = (
            select(CompressionSourceRow)
            .where(CompressionSourceRow.compression_id == compression_id)
            .order_by(CompressionSourceRow.position)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_results(self, compression_id: str) -> list[CompressionResultRow]:
        stmt = (
            select(CompressionResultRow)
            .where(CompressionResultRow.compression_id == compression_id)
            .order_by(CompressionResultRow.position)
        )
        return list(self._session.execute(stmt).scalars().all())

    def is_source_of(self, commit_hash: str) -> bool:
        stmt = select(CompressionSourceRow).where(
            CompressionSourceRow.commit_hash == commit_hash
        )
        return self._session.execute(stmt).first() is not None

    def get_all_source_hashes(self, tract_id: str) -> set[str]:
        stmt = (
            select(CompressionSourceRow.commit_hash)
            .join(
                CompressionRow,
                CompressionSourceRow.compression_id == CompressionRow.compression_id,
            )
            .where(CompressionRow.tract_id == tract_id)
        )
        return set(self._session.execute(stmt).scalars().all())

    def delete_source(self, commit_hash: str) -> None:
        """Delete CompressionSourceRow entries for a commit hash."""
        stmt = select(CompressionSourceRow).where(
            CompressionSourceRow.commit_hash == commit_hash
        )
        for row in self._session.execute(stmt).scalars().all():
            self._session.delete(row)
        self._session.flush()

    def delete_result(self, commit_hash: str) -> None:
        """Delete CompressionResultRow entries for a commit hash."""
        stmt = select(CompressionResultRow).where(
            CompressionResultRow.commit_hash == commit_hash
        )
        for row in self._session.execute(stmt).scalars().all():
            self._session.delete(row)
        self._session.flush()
