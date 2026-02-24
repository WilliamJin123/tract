"""Abstract repository interfaces for Trace storage.

Defines ABC interfaces for all database operations. No SQLAlchemy
imports here -- pure abstract contracts.

Concrete implementations are in sqlite.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from datetime import datetime

    from tract.storage.schema import (
        AnnotationRow,
        BlobRow,
        CommitParentRow,
        CommitRow,
        CompileEffectiveRow,
        CompileRecordRow,
        OperationCommitRow,
        OperationEventRow,
        PolicyLogRow,
        PolicyProposalRow,
        SpawnPointerRow,
        ToolSchemaRow,
    )


class CommitRepository(ABC):
    """Abstract interface for commit storage operations."""

    @abstractmethod
    def get(self, commit_hash: str) -> CommitRow | None:
        """Get a commit by its hash. Returns None if not found."""
        ...

    @abstractmethod
    def save(self, commit: CommitRow) -> None:
        """Save a commit to storage."""
        ...

    @abstractmethod
    def get_ancestors(
        self,
        commit_hash: str,
        limit: int | None = None,
        *,
        op_filter: object | None = None,
    ) -> Sequence[CommitRow]:
        """Get ancestor chain from commit to root (inclusive).

        Args:
            commit_hash: Starting commit hash.
            limit: Maximum number of commits to return.
            op_filter: If set, only include commits with this CommitOperation.

        Returns commits in reverse chronological order (newest first).
        """
        ...

    @abstractmethod
    def get_by_type(self, content_type: str, tract_id: str) -> Sequence[CommitRow]:
        """Get all commits of a given content type in a tract."""
        ...

    @abstractmethod
    def get_children(self, commit_hash: str) -> Sequence[CommitRow]:
        """Get all commits whose parent_hash is the given hash."""
        ...

    @abstractmethod
    def get_by_config(
        self, tract_id: str, json_path: str, operator: str, value: object
    ) -> Sequence[CommitRow]:
        """Get commits where generation_config matches the given condition.

        Args:
            tract_id: Tract identifier to scope the query.
            json_path: JSON field name (e.g., "temperature", "model").
            operator: Comparison operator ("=", "!=", ">", "<", ">=", "<=").
            value: Value to compare against.

        Returns:
            Matching commits ordered by created_at ascending.
        """
        ...

    @abstractmethod
    def get_by_config_multi(
        self, tract_id: str, conditions: list[tuple[str, str, object]]
    ) -> Sequence[CommitRow]:
        """Get commits matching multiple generation config conditions (AND semantics).

        Args:
            tract_id: Tract identifier to scope the query.
            conditions: List of (json_path, operator, value) tuples.
                Operators: "=", "!=", ">", "<", ">=", "<=", "in".
                The "in" operator expects value to be a list/tuple.
            All conditions are combined with AND.

        Returns:
            Matching commits ordered by created_at ascending.
        """
        ...

    @abstractmethod
    def get_by_prefix(self, prefix: str, tract_id: str | None = None) -> CommitRow | None:
        """Find commit by hash prefix (min 4 chars).

        Raises AmbiguousPrefixError if multiple matches.
        Returns None if no match.
        """
        ...

    @abstractmethod
    def get_all(self, tract_id: str) -> Sequence[CommitRow]:
        """Get all commits for a tract, ordered by created_at ascending."""
        ...

    @abstractmethod
    def get_edits_for(self, commit_hash: str, tract_id: str) -> Sequence[CommitRow]:
        """Get the original commit and all its edits in chronological order.

        Returns commits where commit_hash matches OR edit_target matches,
        ordered by created_at ascending. The first element is always the
        original commit (operation=APPEND).

        Args:
            commit_hash: The original commit hash (must not be an edit itself).
            tract_id: Tract identifier to scope the query.

        Returns:
            List of CommitRow in chronological order: [original, edit1, edit2, ...].
        """
        ...

    @abstractmethod
    def delete(self, commit_hash: str) -> None:
        """Delete a commit by hash. Also cleans up CommitParentRow entries."""
        ...


class BlobRepository(ABC):
    """Abstract interface for blob storage operations."""

    @abstractmethod
    def get(self, content_hash: str) -> BlobRow | None:
        """Get a blob by its content hash. Returns None if not found."""
        ...

    @abstractmethod
    def save_if_absent(self, blob: BlobRow) -> None:
        """Store a blob only if its content_hash is not already present.

        Content-addressable: same content = same hash = stored once.
        """
        ...

    @abstractmethod
    def delete_if_orphaned(self, content_hash: str) -> bool:
        """Delete a blob if no commit still references it.

        Returns True if the blob was deleted, False if still referenced.
        """
        ...


class RefRepository(ABC):
    """Abstract interface for ref (branch/HEAD pointer) operations."""

    @abstractmethod
    def get_head(self, tract_id: str) -> str | None:
        """Get the HEAD commit hash for a tract. Returns None if no HEAD."""
        ...

    @abstractmethod
    def update_head(self, tract_id: str, commit_hash: str) -> None:
        """Update the HEAD pointer for a tract."""
        ...

    @abstractmethod
    def get_branch(self, tract_id: str, branch_name: str) -> str | None:
        """Get the commit hash for a named branch. Returns None if not found."""
        ...

    @abstractmethod
    def set_branch(self, tract_id: str, branch_name: str, commit_hash: str) -> None:
        """Set or update a named branch to point at a commit."""
        ...

    @abstractmethod
    def list_branches(self, tract_id: str) -> list[str]:
        """List all branch names for a tract."""
        ...

    @abstractmethod
    def is_detached(self, tract_id: str) -> bool:
        """Check if HEAD is in detached state (pointing directly at a commit)."""
        ...

    @abstractmethod
    def attach_head(self, tract_id: str, branch_name: str) -> None:
        """Attach HEAD to a branch (symbolic ref: HEAD -> refs/heads/{branch_name})."""
        ...

    @abstractmethod
    def detach_head(self, tract_id: str, commit_hash: str) -> None:
        """Detach HEAD to point directly at a commit hash."""
        ...

    @abstractmethod
    def get_ref(self, tract_id: str, ref_name: str) -> str | None:
        """Get the commit hash for a named ref. Returns None if not found."""
        ...

    @abstractmethod
    def set_ref(self, tract_id: str, ref_name: str, commit_hash: str) -> None:
        """Set or update a named ref to point at a commit hash."""
        ...

    @abstractmethod
    def delete_ref(self, tract_id: str, ref_name: str) -> None:
        """Delete a named ref. No-op if ref doesn't exist."""
        ...

    @abstractmethod
    def set_symbolic_ref(self, tract_id: str, ref_name: str, symbolic_target: str) -> None:
        """Set a ref to point at another ref symbolically (commit_hash=None)."""
        ...

    @abstractmethod
    def get_symbolic_ref(self, tract_id: str, ref_name: str) -> str | None:
        """Get the symbolic target of a ref. Returns None if not found or not symbolic."""
        ...

    @abstractmethod
    def get_current_branch(self, tract_id: str) -> str | None:
        """Get the current branch name if HEAD is attached. Returns None if detached."""
        ...


class CommitParentRepository(ABC):
    """Abstract interface for multi-parent commit storage (merge commits)."""

    @abstractmethod
    def add_parent(self, commit_hash: str, parent_hash: str, position: int) -> None:
        """Add a single parent entry for a commit."""
        ...

    @abstractmethod
    def get_parents(self, commit_hash: str) -> list[str]:
        """Get parent hashes for a commit, ordered by position.

        Returns empty list for non-merge commits (no entries in table).
        """
        ...

    @abstractmethod
    def add_parents(self, commit_hash: str, parent_hashes: list[str]) -> None:
        """Batch add parents for a commit. Position = list index."""
        ...


class AnnotationRepository(ABC):
    """Abstract interface for priority annotation operations."""

    @abstractmethod
    def get_latest(self, target_hash: str) -> AnnotationRow | None:
        """Get the most recent annotation for a commit. Returns None if none."""
        ...

    @abstractmethod
    def save(self, annotation: AnnotationRow) -> None:
        """Save an annotation (append-only)."""
        ...

    @abstractmethod
    def get_history(self, target_hash: str) -> Sequence[AnnotationRow]:
        """Get all annotations for a commit, ordered by created_at ascending."""
        ...

    @abstractmethod
    def batch_get_latest(self, target_hashes: list[str]) -> dict[str, AnnotationRow]:
        """Get the latest annotation for each of multiple commits.

        Returns a dict mapping target_hash to the latest AnnotationRow.
        Commits with no annotations are omitted from the result.

        This avoids N+1 queries during compilation.
        """
        ...


class OperationEventRepository(ABC):
    """Abstract interface for unified operation event storage.

    Tracks operation events (compress, reorganize, import) and their
    associated commits (sources and results).
    """

    @abstractmethod
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
        """Save an operation event."""
        ...

    @abstractmethod
    def add_commit(
        self, event_id: str, commit_hash: str, role: str, position: int
    ) -> None:
        """Add a commit association to an operation event."""
        ...

    @abstractmethod
    def get_event(self, event_id: str) -> OperationEventRow | None:
        """Get an operation event by ID. Returns None if not found."""
        ...

    @abstractmethod
    def get_commits(
        self, event_id: str, role: str | None = None
    ) -> list[OperationCommitRow]:
        """Get commit associations for an event, optionally filtered by role."""
        ...

    @abstractmethod
    def is_source_of(self, commit_hash: str) -> bool:
        """Check if a commit is a source in any operation event."""
        ...

    @abstractmethod
    def get_all_source_hashes(self, tract_id: str) -> set[str]:
        """Get all source commit hashes for a tract across all events."""
        ...

    @abstractmethod
    def get_all_ids(self, tract_id: str) -> list[str]:
        """Get all event IDs for a tract."""
        ...

    @abstractmethod
    def delete_commit(self, commit_hash: str) -> None:
        """Delete all OperationCommitRow entries for a commit hash."""
        ...

    @abstractmethod
    def delete_event(self, event_id: str) -> None:
        """Delete an event and all its commit associations."""
        ...


class CompileRecordRepository(ABC):
    """Abstract interface for compile record storage.

    Tracks compile operations: which head was compiled, how many
    tokens/commits were included, and which commits were effective.
    """

    @abstractmethod
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
        """Save a compile record."""
        ...

    @abstractmethod
    def add_effective(
        self, record_id: str, commit_hash: str, position: int
    ) -> None:
        """Add an effective commit to a compile record."""
        ...

    @abstractmethod
    def get_record(self, record_id: str) -> CompileRecordRow | None:
        """Get a compile record by ID. Returns None if not found."""
        ...

    @abstractmethod
    def get_all(self, tract_id: str) -> list[CompileRecordRow]:
        """Get all compile records for a tract, ordered by created_at."""
        ...

    @abstractmethod
    def get_effectives(self, record_id: str) -> list[CompileEffectiveRow]:
        """Get effective commits for a compile record, ordered by position."""
        ...


class SpawnPointerRepository(ABC):
    """Abstract interface for spawn pointer storage.

    Tracks parent-child relationships between tracts in a spawn tree.
    A child tract has at most one parent. A parent can have many children.
    """

    @abstractmethod
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
        """Save a new spawn pointer. Returns the created row."""
        ...

    @abstractmethod
    def get(self, id: int) -> SpawnPointerRow | None:
        """Get spawn pointer by ID. Returns None if not found."""
        ...

    @abstractmethod
    def get_by_child(self, child_tract_id: str) -> SpawnPointerRow | None:
        """Get the spawn pointer where this tract is the child.

        A tract has at most one parent. Returns None if this tract
        was not spawned from another.
        """
        ...

    @abstractmethod
    def get_children(self, parent_tract_id: str) -> list[SpawnPointerRow]:
        """Get all spawn pointers where this tract is the parent.

        Returns pointers ordered by created_at ascending.
        """
        ...

    @abstractmethod
    def get_all(self, tract_id: str) -> list[SpawnPointerRow]:
        """Get all spawn pointers involving this tract (as parent OR child)."""
        ...

    @abstractmethod
    def has_ancestor(self, child_tract_id: str, potential_ancestor: str) -> bool:
        """Check if potential_ancestor is an ancestor of child_tract_id in the spawn tree.

        Walks up the spawn tree from child_tract_id. Returns True if
        potential_ancestor is found as a parent at any level.
        Used for cycle detection during spawn operations.
        """
        ...


class PolicyRepository(ABC):
    """Abstract interface for policy proposal and log storage.

    Provides CRUD for policy proposals (collaborative approval workflow)
    and audit log entries (tracking all policy evaluations).
    """

    @abstractmethod
    def save_proposal(self, proposal: PolicyProposalRow) -> None:
        """Save a new policy proposal."""
        ...

    @abstractmethod
    def get_proposal(self, proposal_id: str) -> PolicyProposalRow | None:
        """Get a proposal by its ID. Returns None if not found."""
        ...

    @abstractmethod
    def get_pending_proposals(self, tract_id: str) -> list[PolicyProposalRow]:
        """Get all pending proposals for a tract.

        Returns proposals with status="pending", ordered by created_at ascending.
        """
        ...

    @abstractmethod
    def update_proposal_status(
        self, proposal_id: str, status: str, resolved_at: datetime
    ) -> None:
        """Update a proposal's status and resolution timestamp."""
        ...

    @abstractmethod
    def save_log_entry(self, entry: PolicyLogRow) -> None:
        """Save a policy evaluation log entry."""
        ...

    @abstractmethod
    def get_log(
        self,
        tract_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        policy_name: str | None = None,
        limit: int = 100,
    ) -> list[PolicyLogRow]:
        """Get policy log entries for a tract.

        Returns entries ordered by created_at DESC, with optional filters.
        """
        ...

    @abstractmethod
    def delete_log_entries(self, tract_id: str, before: datetime) -> int:
        """Delete log entries older than the given timestamp.

        Returns the number of entries deleted. Used for audit log GC.
        """
        ...


class ToolSchemaRepository(ABC):
    """Abstract interface for tool schema storage.

    Content-addressed storage for tool JSON schemas (function definitions
    passed to LLM APIs). Each unique schema is stored once; commits
    reference schemas through the CommitToolRow join table.
    """

    @abstractmethod
    def store(
        self, content_hash: str, name: str, schema: dict, created_at: datetime
    ) -> ToolSchemaRow:
        """Store a tool schema (idempotent -- returns existing if hash matches).

        Args:
            content_hash: SHA-256 of the canonical JSON.
            name: Tool function name.
            schema: Full tool definition dict.
            created_at: Creation timestamp.

        Returns:
            The stored or existing ToolSchemaRow.
        """
        ...

    @abstractmethod
    def get(self, content_hash: str) -> ToolSchemaRow | None:
        """Get a tool schema by its content hash. Returns None if not found."""
        ...

    @abstractmethod
    def get_by_name(self, name: str) -> Sequence[ToolSchemaRow]:
        """Get all tool schema versions with the given name."""
        ...

    @abstractmethod
    def get_for_commit(self, commit_hash: str) -> Sequence[ToolSchemaRow]:
        """Get tool schemas linked to a commit, ordered by position."""
        ...

    @abstractmethod
    def link_to_commit(
        self, commit_hash: str, tool_hash: str, position: int
    ) -> None:
        """Link a tool schema to a commit at a given position."""
        ...

    @abstractmethod
    def get_commit_tool_hashes(self, commit_hash: str) -> Sequence[str]:
        """Get content hashes of tools linked to a commit, ordered by position."""
        ...
