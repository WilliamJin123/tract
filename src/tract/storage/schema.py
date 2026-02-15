"""SQLAlchemy ORM schema for Trace.

Defines all database tables: blobs, commits, refs, annotations, _trace_meta.

IMPORTANT: CommitOperation and Priority enums are imported from the domain
models -- they are NOT redefined here. The ORM uses the same Python enums.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from tract.models.annotations import Priority
from tract.models.commit import CommitOperation


class Base(DeclarativeBase):
    """Base class for all Trace ORM models."""

    pass


class BlobRow(Base):
    """Content-addressable blob storage. Keyed by SHA-256 of content."""

    __tablename__ = "blobs"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class CommitRow(Base):
    """A commit in the context DAG."""

    __tablename__ = "commits"

    commit_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    parent_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash", ondelete="SET NULL"),
        nullable=True,
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("blobs.content_hash"),
        nullable=False,
    )
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    operation: Mapped[CommitOperation] = mapped_column(nullable=False)
    response_to: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash", ondelete="SET NULL"),
        nullable=True,
    )
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    generation_config_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Relationships
    blob: Mapped["BlobRow"] = relationship("BlobRow", lazy="select")
    parent: Mapped[Optional["CommitRow"]] = relationship(
        "CommitRow",
        remote_side="CommitRow.commit_hash",
        foreign_keys=[parent_hash],
    )

    __table_args__ = (
        Index("ix_commits_tract_time", "tract_id", "created_at"),
        Index("ix_commits_tract_type", "tract_id", "content_type"),
        Index("ix_commits_response_to", "response_to"),
    )


class RefRow(Base):
    """Mutable named pointer to a commit (branch, HEAD)."""

    __tablename__ = "refs"

    tract_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ref_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    commit_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        nullable=True,
    )
    symbolic_target: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class AnnotationRow(Base):
    """Lightweight priority annotation (like git tags).

    Append-only: each change creates a new row for provenance.
    The latest row for a given target_hash is the current annotation.
    """

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        nullable=False,
    )
    priority: Mapped[Priority] = mapped_column(nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_annotations_target_time", "target_hash", "created_at"),
    )


class CommitParentRow(Base):
    """Association table for multi-parent commits (merge commits).

    For non-merge commits, only CommitRow.parent_hash is used (single parent).
    For merge commits, this table stores ALL parents (including the first).
    The 'position' column preserves parent ordering (important for merge semantics).
    """

    __tablename__ = "commit_parents"

    commit_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        primary_key=True,
    )
    parent_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    # position 0 = first parent (current branch tip), 1 = merged branch tip

    __table_args__ = (
        Index("ix_commit_parents_commit", "commit_hash"),
    )


class TraceMetaRow(Base):
    """Key-value metadata for the Trace database itself (e.g., schema version)."""

    __tablename__ = "_trace_meta"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
