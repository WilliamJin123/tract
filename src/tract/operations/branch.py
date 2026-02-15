"""Branch CRUD operations for Trace.

Create, delete, list, and validate branches.
Composes storage primitives (ref repo, commit repo) into higher-level actions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tract.exceptions import (
    BranchExistsError,
    BranchNotFoundError,
    InvalidBranchNameError,
    TraceError,
    UnmergedBranchError,
)

if TYPE_CHECKING:
    from tract.storage.repositories import (
        CommitParentRepository,
        CommitRepository,
        RefRepository,
    )


# Characters forbidden in branch names (git-style)
_FORBIDDEN_CHARS = re.compile(r"[\s~^:?*\[\\]")


def validate_branch_name(name: str) -> None:
    """Validate a branch name against git-style naming rules.

    Raises InvalidBranchNameError on violation.
    """
    if not name:
        raise InvalidBranchNameError(name, "branch name cannot be empty")

    if ".." in name:
        raise InvalidBranchNameError(name, "branch name cannot contain '..'")

    if name.endswith(".lock"):
        raise InvalidBranchNameError(name, "branch name cannot end with '.lock'")

    if name.startswith("."):
        raise InvalidBranchNameError(name, "branch name cannot start with '.'")

    if name.endswith("."):
        raise InvalidBranchNameError(name, "branch name cannot end with '.'")

    if _FORBIDDEN_CHARS.search(name):
        raise InvalidBranchNameError(
            name, "branch name contains forbidden characters (whitespace, ~, ^, :, ?, *, [, \\)"
        )

    if name.startswith("/") or name.endswith("/") or "//" in name:
        raise InvalidBranchNameError(name, "branch name has invalid slash usage")


def create_branch(
    name: str,
    tract_id: str,
    ref_repo: RefRepository,
    commit_repo: CommitRepository,
    *,
    source: str | None = None,
    switch: bool = True,
) -> str:
    """Create a new branch pointing at source commit.

    Args:
        name: Branch name (validated against naming rules).
        tract_id: The tract identifier.
        ref_repo: Ref repository for branch storage.
        commit_repo: Commit repository for validation.
        source: Commit hash to branch from. Defaults to HEAD.
        switch: If True, switch HEAD to new branch.

    Returns:
        The commit hash the new branch points to.

    Raises:
        BranchExistsError: If branch name already exists.
        InvalidBranchNameError: If branch name is invalid.
        TraceError: If no commits exist and no source specified.
    """
    validate_branch_name(name)

    # Check branch doesn't already exist
    existing = ref_repo.get_branch(tract_id, name)
    if existing is not None:
        raise BranchExistsError(name)

    # Resolve source
    if source is None:
        source = ref_repo.get_head(tract_id)
        if source is None:
            raise TraceError("Cannot create branch: no commits exist")

    # Create branch ref
    ref_repo.set_branch(tract_id, name, source)

    # Switch to new branch if requested
    if switch:
        ref_repo.attach_head(tract_id, name)

    return source


def delete_branch(
    name: str,
    tract_id: str,
    ref_repo: RefRepository,
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository | None = None,
    *,
    force: bool = False,
) -> None:
    """Delete a branch.

    Args:
        name: Branch name to delete.
        tract_id: The tract identifier.
        ref_repo: Ref repository for branch storage.
        commit_repo: Commit repository for reachability checks.
        parent_repo: Parent repository for multi-parent reachability.
        force: If True, delete even if branch has unmerged commits.

    Raises:
        BranchNotFoundError: If branch doesn't exist.
        TraceError: If trying to delete the current branch.
        UnmergedBranchError: If branch has unmerged commits (without force).
    """
    # Check branch exists
    branch_hash = ref_repo.get_branch(tract_id, name)
    if branch_hash is None:
        raise BranchNotFoundError(name)

    # Block deletion of current branch
    current = ref_repo.get_current_branch(tract_id)
    if current == name:
        raise TraceError(f"Cannot delete the current branch '{name}'")

    # Check if branch tip is reachable from current branch (merged check)
    if not force:
        current_head = ref_repo.get_head(tract_id)
        if current_head is not None and branch_hash != current_head:
            from tract.operations.dag import is_ancestor

            if not is_ancestor(commit_repo, parent_repo, branch_hash, current_head):
                raise UnmergedBranchError(name)

    # Delete the branch ref
    ref_name = f"refs/heads/{name}"
    ref_repo.delete_ref(tract_id, ref_name)


def list_branches(
    tract_id: str,
    ref_repo: RefRepository,
) -> list[str]:
    """List all branch names for a tract.

    Args:
        tract_id: The tract identifier.
        ref_repo: Ref repository for branch lookups.

    Returns:
        List of branch names.
    """
    return ref_repo.list_branches(tract_id)
