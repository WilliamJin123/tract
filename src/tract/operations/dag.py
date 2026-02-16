"""DAG utilities for Trace -- merge base computation and ancestor queries.

These utilities operate on the commit DAG, following both first-parent
(CommitRow.parent_hash) and extra parents (CommitParentRow) for merge commits.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.storage.repositories import CommitParentRepository, CommitRepository
    from tract.storage.schema import CommitRow


def _bfs_walk(
    start: str,
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository | None,
    *,
    stop_at: set[str] | None = None,
) -> Iterator[str]:
    """BFS walk from a start hash, yielding each visited commit hash once.

    Follows both first-parent (parent_hash) and extra parents from
    the commit_parents table (merge commits).

    Args:
        start: Starting commit hash.
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository for multi-parent traversal.
        stop_at: Optional set of known-visited hashes. When a commit is
            in this set, it is yielded but its parents are not enqueued.
            This avoids redundant walks when building reachability sets
            across multiple branches.
    """
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        yield current
        # Short-circuit: if this commit is already known reachable,
        # skip expanding its parents (they are already in stop_at).
        if stop_at is not None and current in stop_at:
            continue
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue.append(commit.parent_hash)
        if parent_repo is not None:
            for extra in parent_repo.get_parents(current):
                if extra not in visited:
                    queue.append(extra)


def find_merge_base(
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository | None,
    hash_a: str,
    hash_b: str,
) -> str | None:
    """Find the best common ancestor (merge base) of two commits.

    Walks both ancestor chains using BFS and returns the first intersection.
    For linear history, this is the point where branches diverged.
    For merge commits, follows ALL parents.

    Args:
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository for multi-parent traversal.
        hash_a: First commit hash.
        hash_b: Second commit hash.

    Returns:
        The commit hash of the merge base, or None if no common ancestor.
    """
    ancestors_a = set(_bfs_walk(hash_a, commit_repo, parent_repo))

    for h in _bfs_walk(hash_b, commit_repo, parent_repo):
        if h in ancestors_a:
            return h

    return None


def get_all_ancestors(
    commit_hash: str,
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository | None,
    *,
    stop_at: set[str] | None = None,
) -> set[str]:
    """Get all ancestor hashes of a commit (including itself).

    Follows both first-parent (parent_hash) and extra parents (commit_parents table).

    Args:
        commit_hash: Starting commit hash.
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository for multi-parent traversal.
        stop_at: Optional set of known-reachable hashes to short-circuit
            the walk. Parents of commits in this set are not explored.

    Returns:
        Set of all ancestor commit hashes (including commit_hash).
    """
    return set(_bfs_walk(commit_hash, commit_repo, parent_repo, stop_at=stop_at))


def get_branch_commits(
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository | None,
    branch_tip: str,
    merge_base: str,
) -> list[CommitRow]:
    """Get commits between merge_base (exclusive) and branch_tip (inclusive).

    Only follows first-parent chain. Returns commits in chronological
    order (root to tip).

    Args:
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository (unused, reserved for future).
        branch_tip: The tip commit hash of the branch.
        merge_base: The merge base commit hash (excluded from results).

    Returns:
        List of CommitRow in chronological order (oldest first).
    """
    commits: list[CommitRow] = []
    current_hash: str | None = branch_tip

    while current_hash is not None and current_hash != merge_base:
        commit = commit_repo.get(current_hash)
        if commit is None:
            break
        commits.append(commit)
        current_hash = commit.parent_hash

    # Reverse to chronological order (root first)
    commits.reverse()
    return commits


def is_ancestor(
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository | None,
    potential_ancestor: str,
    commit_hash: str,
) -> bool:
    """Check if potential_ancestor is reachable from commit_hash.

    Uses early-termination BFS via _bfs_walk — stops as soon as the
    target is found, avoiding the cost of building the complete ancestor set.

    Args:
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository for multi-parent traversal.
        potential_ancestor: The commit hash to check as ancestor.
        commit_hash: The commit hash to walk backwards from.

    Returns:
        True if potential_ancestor is reachable from commit_hash.
    """
    for h in _bfs_walk(commit_hash, commit_repo, parent_repo):
        if h == potential_ancestor:
            return True
    return False
