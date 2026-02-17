"""Cross-repo query operations for multi-agent sessions.

Pure functions that operate on SQLAlchemy sessions and repositories.
Used by Session class to provide timeline, search, compile_at, resume,
and list_tracts functionality.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from tract.models.commit import CommitInfo, CommitOperation
from tract.storage.schema import BlobRow, CommitRow, SpawnPointerRow

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from tract.protocols import CompiledContext
    from tract.storage.sqlite import SqliteSpawnPointerRepository


def _is_tract_ended(session: Session, tract_id: str) -> bool:
    """Check if a tract has a session_type='end' commit.

    Queries session-type commits for the tract and inspects their blob
    payloads to find an 'end' marker.
    """
    end_stmt = (
        select(CommitRow)
        .where(
            CommitRow.tract_id == tract_id,
            CommitRow.content_type == "session",
        )
    )
    end_commits = session.execute(end_stmt).scalars().all()
    for ec in end_commits:
        blob = session.execute(
            select(BlobRow).where(BlobRow.content_hash == ec.content_hash)
        ).scalar_one_or_none()
        if blob is not None:
            payload = json.loads(blob.payload_json)
            if payload.get("session_type") == "end":
                return True
    return False


def list_tracts(
    session: Session,
    spawn_repo: SqliteSpawnPointerRepository,
) -> list[dict]:
    """List all tracts in the database with metadata.

    Args:
        session: SQLAlchemy session.
        spawn_repo: Repository for spawn pointer lookups.

    Returns:
        List of dicts with tract metadata: tract_id, display_name,
        commit_count, latest_commit_at, is_active, parent_tract_id.
    """
    # Get distinct tract_ids from commits
    stmt = (
        select(
            CommitRow.tract_id,
            func.count(CommitRow.commit_hash).label("commit_count"),
            func.max(CommitRow.created_at).label("latest_commit_at"),
        )
        .group_by(CommitRow.tract_id)
    )
    rows = session.execute(stmt).all()

    results = []
    for row in rows:
        tract_id = row[0]
        commit_count = row[1]
        latest_commit_at = row[2]

        is_active = not _is_tract_ended(session, tract_id)

        # Check spawn pointer for parent info and display name
        pointer = spawn_repo.get_by_child(tract_id)
        parent_tract_id = pointer.parent_tract_id if pointer else None
        display_name = pointer.display_name if pointer else None

        results.append({
            "tract_id": tract_id,
            "display_name": display_name,
            "commit_count": commit_count,
            "latest_commit_at": latest_commit_at,
            "is_active": is_active,
            "parent_tract_id": parent_tract_id,
        })

    return results


def timeline(
    session: Session,
    *,
    limit: int | None = None,
) -> list[CommitInfo]:
    """Get all commits across all tracts in chronological order.

    Args:
        session: SQLAlchemy session.
        limit: Maximum number of commits to return.

    Returns:
        List of CommitInfo in chronological order (oldest first).
    """
    stmt = select(CommitRow).order_by(CommitRow.created_at.asc())
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = session.execute(stmt).scalars().all()
    return [_row_to_commit_info(row) for row in rows]


def search(
    session: Session,
    term: str,
    *,
    tract_id: str | None = None,
) -> list[CommitInfo]:
    """Search for commits matching a term in blob content.

    Args:
        session: SQLAlchemy session.
        term: Search term (matched via LIKE on blob payload).
        tract_id: Optional filter to a specific tract.

    Returns:
        List of matching CommitInfo.
    """
    # Escape LIKE wildcards
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    # Query blobs matching the term
    blob_stmt = (
        select(BlobRow.content_hash)
        .where(BlobRow.payload_json.like(f"%{escaped}%", escape="\\"))
    )
    matching_hashes = set(session.execute(blob_stmt).scalars().all())

    if not matching_hashes:
        return []

    # Find commits referencing those blobs
    commit_stmt = select(CommitRow).where(
        CommitRow.content_hash.in_(matching_hashes)
    ).order_by(CommitRow.created_at.asc())

    if tract_id is not None:
        commit_stmt = commit_stmt.where(CommitRow.tract_id == tract_id)

    rows = session.execute(commit_stmt).scalars().all()
    return [_row_to_commit_info(row) for row in rows]


def compile_at(
    session_factory,
    engine,
    tract_id: str,
    *,
    at_time: datetime | None = None,
    at_commit: str | None = None,
) -> CompiledContext:
    """Compile any tract at a historical point-in-time.

    Creates a temporary Tract for the target tract_id and compiles it.

    Args:
        session_factory: SQLAlchemy sessionmaker.
        engine: SQLAlchemy engine.
        tract_id: The tract to compile.
        at_time: Compile as of this datetime.
        at_commit: Compile up to this commit hash.

    Returns:
        CompiledContext for the tract at the specified point.
    """
    from tract.engine.commit import CommitEngine
    from tract.engine.compiler import DefaultContextCompiler
    from tract.engine.tokens import TiktokenCounter
    from tract.models.config import TractConfig
    from tract.storage.sqlite import (
        SqliteAnnotationRepository,
        SqliteBlobRepository,
        SqliteCommitParentRepository,
        SqliteCommitRepository,
        SqliteRefRepository,
    )

    session = session_factory()
    try:
        config = TractConfig()

        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        ref_repo = SqliteRefRepository(session)
        annotation_repo = SqliteAnnotationRepository(session)
        parent_repo = SqliteCommitParentRepository(session)

        token_counter = TiktokenCounter(encoding_name=config.tokenizer_encoding)

        compiler = DefaultContextCompiler(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            parent_repo=parent_repo,
        )

        # Get HEAD for this tract
        head_hash = ref_repo.get_head(tract_id)
        if head_hash is None:
            from tract.protocols import CompiledContext

            return CompiledContext(messages=[], token_count=0, commit_count=0, token_source="")

        return compiler.compile(
            tract_id,
            head_hash,
            at_time=at_time,
            at_commit=at_commit,
        )
    finally:
        session.close()


def resume(
    session: Session,
    spawn_repo: SqliteSpawnPointerRepository,
) -> dict | None:
    """Find the most recent active tract for crash recovery.

    Prefers root tracts (no parent in spawn_pointers) with the latest commit.

    Args:
        session: SQLAlchemy session.
        spawn_repo: Repository for spawn pointer lookups.

    Returns:
        Dict with tract_id and metadata, or None if no active tracts.
    """
    # Get all tract IDs with their latest commit time
    stmt = (
        select(
            CommitRow.tract_id,
            func.max(CommitRow.created_at).label("latest"),
        )
        .group_by(CommitRow.tract_id)
        .order_by(func.max(CommitRow.created_at).desc())
    )
    rows = session.execute(stmt).all()

    if not rows:
        return None

    # Filter out ended tracts and prefer root tracts
    best_root = None
    best_any = None

    for row in rows:
        tract_id = row[0]
        latest = row[1]

        if _is_tract_ended(session, tract_id):
            continue

        is_root = spawn_repo.get_by_child(tract_id) is None

        info = {
            "tract_id": tract_id,
            "latest_commit_at": latest,
            "is_root": is_root,
        }

        if is_root and best_root is None:
            best_root = info
        elif best_any is None:
            best_any = info

    # Prefer root tracts
    return best_root or best_any


def _row_to_commit_info(row: CommitRow) -> CommitInfo:
    """Convert a CommitRow ORM object to a CommitInfo Pydantic model.

    Note: This duplicates conversion logic in CommitEngine._row_to_info().
    Kept separate because session_ops operates on raw sessions without
    a CommitEngine instance.
    """
    return CommitInfo(
        commit_hash=row.commit_hash,
        tract_id=row.tract_id,
        parent_hash=row.parent_hash,
        content_hash=row.content_hash,
        content_type=row.content_type,
        operation=row.operation,
        response_to=row.response_to,
        message=row.message,
        token_count=row.token_count,
        metadata=row.metadata_json,
        generation_config=row.generation_config_json,
        created_at=row.created_at,
    )
