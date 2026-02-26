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
    edit_target: Mapped[Optional[str]] = mapped_column(
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
        Index("ix_commits_edit_target", "edit_target"),
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
    retention_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
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


class OperationEventRow(Base):
    """A unified operation event record.

    Tracks any operation that transforms commits: compression, reorganization,
    import, etc.
    """

    __tablename__ = "operation_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # "compress", "reorganize", "import"
    branch_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    original_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    compressed_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    params_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    original_instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    effective_instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_operation_events_tract_type", "tract_id", "event_type"),
        Index("ix_operation_events_original_tokens", "original_tokens"),
        Index("ix_operation_events_compressed_tokens", "compressed_tokens"),
    )


class OperationCommitRow(Base):
    """Association between an operation event and its commits.

    Each commit can be a "source" (input to the operation) or a "result"
    (output of the operation). Position preserves ordering within each role.
    """

    __tablename__ = "operation_commits"

    event_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("operation_events.event_id"),
        primary_key=True,
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(
        String(10), primary_key=True
    )  # "source" or "result"
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_operation_commits_event", "event_id"),
        Index("ix_operation_commits_hash_role", "commit_hash", "role"),
    )


class CompileRecordRow(Base):
    """A record of a compile (context materialization) operation.

    Captures the state of a compile: which head it was built from,
    how many tokens/commits were included, and what parameters were used.
    """

    __tablename__ = "compile_records"

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    head_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    commit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_source: Mapped[str] = mapped_column(String(50), nullable=False)
    params_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_compile_records_tract_time", "tract_id", "created_at"),
    )


class CompileEffectiveRow(Base):
    """Association between a compile record and the commits that were effective.

    Position preserves the order of commits as they appeared in the compiled context.
    """

    __tablename__ = "compile_effectives"

    record_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("compile_records.record_id"),
        primary_key=True,
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)


class SpawnPointerRow(Base):
    """Cross-tract linkage for multi-agent spawn relationships.

    Each row represents a parent-child spawn relationship between two tracts.
    A child tract has at most one parent (the tract that spawned it).
    A parent tract can have many children.
    """

    __tablename__ = "spawn_pointers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_tract_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_commit_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash", ondelete="SET NULL"),
        nullable=True,
    )
    child_tract_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    inheritance_mode: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "full_clone", "head_snapshot", "selective"
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_spawn_parent_tract", "parent_tract_id"),
        Index("ix_spawn_child_tract", "child_tract_id"),
    )


class PolicyProposalRow(Base):
    """A policy proposal awaiting approval or rejection.

    Proposals are created when a policy runs in collaborative mode.
    They can be approved (executed), rejected, or expire.
    """

    __tablename__ = "policy_proposals"

    proposal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    policy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    action_params_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # "pending", "approved", "rejected", "expired", "executed"
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_policy_proposals_tract_status", "tract_id", "status"),
    )


class PolicyLogRow(Base):
    """Audit log entry for a policy evaluation.

    Records every policy evaluation: what triggered it, what action
    was proposed or executed, and what the outcome was.
    """

    __tablename__ = "policy_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    policy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    trigger: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "compile" or "commit"
    action_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    outcome: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "executed", "proposed", "skipped", "error"
    commit_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # the commit produced by this action, if any
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_policy_log_tract_time", "tract_id", "created_at"),
    )


class ToolSchemaRow(Base):
    """Content-addressed storage for tool JSON schemas.

    Each unique tool definition (identified by SHA-256 of its canonical JSON)
    is stored once. Multiple commits can reference the same tool schema.
    """

    __tablename__ = "tool_definitions"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class CommitToolRow(Base):
    """Associates a commit with the tool definitions active at that point.

    Each commit can have zero or more tool definitions linked to it.
    The position column preserves the tool ordering (important for API calls).
    """

    __tablename__ = "commit_tools"

    commit_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), primary_key=True,
    )
    tool_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("tool_definitions.content_hash"), primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_commit_tools_commit", "commit_hash"),
    )


class TraceMetaRow(Base):
    """Key-value metadata for the Trace database itself (e.g., schema version)."""

    __tablename__ = "_trace_meta"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
