"""Navigation operations for Trace -- reset and checkout.

These operations manipulate HEAD position without creating new commits.
They compose storage primitives (ref repo, commit repo) into higher-level
user-facing actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tract.exceptions import CommitNotFoundError, TraceError

if TYPE_CHECKING:
    from tract.storage.repositories import CommitRepository, RefRepository


def resolve_commit(
    ref_or_prefix: str,
    tract_id: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
) -> str:
    """Resolve a commit reference to a full commit hash.

    Resolution order:
    1. Full commit hash (exact match)
    2. Branch name (refs/heads/{name})
    3. Hash prefix (min 4 chars, via get_by_prefix)

    Args:
        ref_or_prefix: A commit hash, branch name, or hash prefix.
        tract_id: The tract identifier.
        commit_repo: Commit repository for hash lookups.
        ref_repo: Ref repository for branch lookups.

    Returns:
        The full commit hash.

    Raises:
        CommitNotFoundError: If no commit can be resolved.
        AmbiguousPrefixError: If a prefix matches multiple commits.
    """
    # 1. Exact commit hash (scoped to tract_id)
    row = commit_repo.get(ref_or_prefix)
    if row is not None and row.tract_id == tract_id:
        return row.commit_hash

    # 2. Branch name
    branch_hash = ref_repo.get_branch(tract_id, ref_or_prefix)
    if branch_hash is not None:
        return branch_hash

    # 3. Hash prefix (min 4 chars)
    if len(ref_or_prefix) >= 4:
        row = commit_repo.get_by_prefix(ref_or_prefix, tract_id=tract_id)
        if row is not None:
            return row.commit_hash

    raise CommitNotFoundError(ref_or_prefix)


def reset(
    target_hash: str,
    mode: Literal["soft", "hard"],
    tract_id: str,
    ref_repo: RefRepository,
) -> str:
    """Reset HEAD to a target commit.

    Stores the current HEAD as ORIG_HEAD before moving.

    - soft: moves HEAD only (forward history still accessible via hashes)
    - hard: moves HEAD only (same as soft in Trace -- no working tree)

    Note: In Trace, soft and hard have identical behavior because there
    is no working directory to clean.  The distinction exists for API
    compatibility with git semantics and future extensions.

    Args:
        target_hash: The commit hash to reset to.
        mode: "soft" or "hard".
        tract_id: The tract identifier.
        ref_repo: Ref repository for HEAD manipulation.

    Returns:
        The target commit hash (new HEAD).
    """
    # Store current HEAD as ORIG_HEAD
    current_head = ref_repo.get_head(tract_id)
    if current_head is not None:
        ref_repo.set_ref(tract_id, "ORIG_HEAD", current_head)

    # Move HEAD (update_head handles attached/detached correctly)
    ref_repo.update_head(tract_id, target_hash)

    return target_hash


def checkout(
    target: str,
    tract_id: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
) -> tuple[str, bool]:
    """Checkout a commit or branch.

    - Branch name: attach HEAD to that branch
    - Commit hash/prefix: detach HEAD at that commit
    - "-": return to PREV_HEAD position

    Stores the current HEAD as PREV_HEAD before switching.

    Args:
        target: A branch name, commit hash, hash prefix, or "-".
        tract_id: The tract identifier.
        commit_repo: Commit repository for hash lookups.
        ref_repo: Ref repository for HEAD manipulation.

    Returns:
        Tuple of (resolved_commit_hash, is_detached).
        is_detached is True if HEAD is now detached.

    Raises:
        CommitNotFoundError: If the target cannot be resolved.
        TraceError: If "-" is used but no PREV_HEAD exists.
    """
    current_head = ref_repo.get_head(tract_id)

    # Handle "-" (return to previous position)
    # Read PREV_HEAD *before* overwriting it with current HEAD
    if target == "-":
        prev_head = ref_repo.get_ref(tract_id, "PREV_HEAD")
        if prev_head is None:
            raise TraceError("No previous position to return to (PREV_HEAD not set)")
        prev_branch_ref = ref_repo.get_symbolic_ref(tract_id, "PREV_BRANCH")

        # Store current state as PREV_HEAD/PREV_BRANCH (swap)
        if current_head is not None:
            ref_repo.set_ref(tract_id, "PREV_HEAD", current_head)
            current_branch = ref_repo.get_current_branch(tract_id)
            if current_branch:
                ref_repo.set_symbolic_ref(
                    tract_id, "PREV_BRANCH", f"refs/heads/{current_branch}"
                )
            else:
                ref_repo.delete_ref(tract_id, "PREV_BRANCH")

        # Restore previous position with branch attachment
        if prev_branch_ref:
            branch_name = prev_branch_ref.removeprefix("refs/heads/")
            ref_repo.attach_head(tract_id, branch_name)
            return prev_head, False
        else:
            ref_repo.detach_head(tract_id, prev_head)
            return prev_head, True

    # Store current HEAD as PREV_HEAD before switching
    if current_head is not None:
        ref_repo.set_ref(tract_id, "PREV_HEAD", current_head)
        current_branch = ref_repo.get_current_branch(tract_id)
        if current_branch:
            ref_repo.set_symbolic_ref(
                tract_id, "PREV_BRANCH", f"refs/heads/{current_branch}"
            )
        else:
            ref_repo.delete_ref(tract_id, "PREV_BRANCH")

    # Check if target is a branch name
    branch_hash = ref_repo.get_branch(tract_id, target)
    if branch_hash is not None:
        # Attach HEAD to the branch
        ref_repo.attach_head(tract_id, target)
        return branch_hash, False

    # Resolve as commit hash or prefix
    resolved = resolve_commit(target, tract_id, commit_repo, ref_repo)
    ref_repo.detach_head(tract_id, resolved)
    return resolved, True
