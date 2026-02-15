"""DAG utilities for Trace -- merge base computation and ancestor queries.

These utilities operate on the commit DAG, following both first-parent
(CommitRow.parent_hash) and extra parents (CommitParentRow) for merge commits.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.storage.repositories import CommitParentRepository, CommitRepository
    from tract.storage.schema import CommitRow


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
    # Build set of all ancestors of A (including A itself)
    ancestors_a: set[str] = set()
    queue: deque[str] = deque([hash_a])
    while queue:
        current = queue.popleft()
        if current in ancestors_a:
            continue
        ancestors_a.add(current)
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue.append(commit.parent_hash)
        # Follow extra parents (merge commits)
        if parent_repo is not None:
            for extra in parent_repo.get_parents(current):
                if extra not in ancestors_a:
                    queue.append(extra)

    # BFS from B; first hit in ancestors_a is merge base
    visited: set[str] = set()
    queue = deque([hash_b])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current in ancestors_a:
            return current
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue.append(commit.parent_hash)
        if parent_repo is not None:
            for extra in parent_repo.get_parents(current):
                if extra not in visited:
                    queue.append(extra)

    return None


def get_all_ancestors(
    commit_hash: str,
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository | None,
) -> set[str]:
    """Get all ancestor hashes of a commit (including itself).

    Follows both first-parent (parent_hash) and extra parents (commit_parents table).

    Args:
        commit_hash: Starting commit hash.
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository for multi-parent traversal.

    Returns:
        Set of all ancestor commit hashes (including commit_hash).
    """
    ancestors: set[str] = set()
    queue: deque[str] = deque([commit_hash])

    while queue:
        current = queue.popleft()
        if current in ancestors:
            continue
        ancestors.add(current)
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue.append(commit.parent_hash)
        if parent_repo is not None:
            for extra in parent_repo.get_parents(current):
                if extra not in ancestors:
                    queue.append(extra)

    return ancestors


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

    Args:
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository for multi-parent traversal.
        potential_ancestor: The commit hash to check as ancestor.
        commit_hash: The commit hash to walk backwards from.

    Returns:
        True if potential_ancestor is reachable from commit_hash.
    """
    if potential_ancestor == commit_hash:
        return True

    ancestors = get_all_ancestors(commit_hash, commit_repo, parent_repo)
    return potential_ancestor in ancestors
