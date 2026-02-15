"""Abstract repository interfaces for Trace storage.

Defines ABC interfaces for all database operations. No SQLAlchemy
imports here -- pure abstract contracts.

Concrete implementations are in sqlite.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from tract.storage.schema import AnnotationRow, BlobRow, CommitParentRow, CommitRow


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
    def get_by_prefix(self, prefix: str, tract_id: str | None = None) -> CommitRow | None:
        """Find commit by hash prefix (min 4 chars).

        Raises AmbiguousPrefixError if multiple matches.
        Returns None if no match.
        """
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
