"""Branch manager for Tract.

Extracted from BranchMixin (_branch.py) into a standalone class with explicit
constructor dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tract.engine.cache import CacheManager
    from tract.models.branch import BranchInfo
    from tract.operations.config_index import ConfigIndex
    from tract.storage.repositories import (
        CommitRepository,
        ParentRepository,
        RefRepository,
    )


class BranchManager:
    """Branch creation, switching, checkout, reset, listing, deletion, and commit resolution."""

    def __init__(
        self,
        tract_id: str,
        ref_repo: RefRepository,
        commit_repo: CommitRepository,
        parent_repo: ParentRepository,
        cache: CacheManager,
        check_open: Callable[[], None],
        commit_session: Callable[[], None],
        get_config_index: Callable[[], ConfigIndex | None],
    ) -> None:
        self._tract_id = tract_id
        self._ref_repo = ref_repo
        self._commit_repo = commit_repo
        self._parent_repo = parent_repo
        self._cache = cache
        self._check_open = check_open
        self._commit_session = commit_session
        self._get_config_index = get_config_index

    def create(
        self,
        name: str,
        *,
        source: str | None = None,
        switch: bool = True,
    ) -> str:
        """Create a new branch.

        Args:
            name: Branch name (git-style naming rules apply).
            source: Commit hash to branch from.  Defaults to HEAD.
            switch: If True (default), switch HEAD to the new branch.

        Returns:
            The commit hash the new branch points to.

        Raises:
            BranchExistsError: If branch name already exists.
            InvalidBranchNameError: If branch name is invalid.
            TraceError: If no commits exist and no source specified.
        """
        self._check_open()
        from tract.operations.branch import create_branch

        result = create_branch(
            name,
            self._tract_id,
            self._ref_repo,
            self._commit_repo,
            source=source,
            switch=switch,
        )
        self._commit_session()
        return result

    def switch(self, target: str) -> str:
        """Switch to a branch (branch-only, unlike checkout).

        Unlike :meth:`checkout`, this method ONLY accepts branch names.
        It will not silently detach HEAD on commit hashes -- use
        :meth:`checkout` for that.

        Args:
            target: A branch name.

        Returns:
            The commit hash at the target branch HEAD.

        Raises:
            BranchNotFoundError: If target is not a valid branch name.
        """
        self._check_open()
        from tract.exceptions import BranchNotFoundError

        # Validate that target is a branch
        branch_hash = self._ref_repo.get_branch(self._tract_id, target)
        if branch_hash is None:
            raise BranchNotFoundError(target)

        from tract.operations.navigation import checkout as _checkout

        commit_hash, _is_detached = _checkout(
            target, self._tract_id, self._commit_repo, self._ref_repo
        )
        self._commit_session()
        config_index = self._get_config_index()
        if config_index is not None:
            config_index.invalidate()
        return commit_hash

    def checkout(self, target: str) -> str:
        """Checkout a commit or branch.

        - Branch name: attach HEAD to that branch (enables commits).
        - Commit hash/prefix: detach HEAD (read-only inspection).
        - ``"-"``: return to previous position via PREV_HEAD.

        Stores the current HEAD as PREV_HEAD before switching.

        Args:
            target: A branch name, commit hash, hash prefix, or ``"-"``.

        Returns:
            The resolved commit hash at the new HEAD position.

        Raises:
            CommitNotFoundError: If the target cannot be resolved.
            TraceError: If ``"-"`` is used but no PREV_HEAD exists.
        """
        self._check_open()
        from tract.operations.navigation import checkout as _checkout

        commit_hash, _is_detached = _checkout(
            target, self._tract_id, self._commit_repo, self._ref_repo
        )
        self._commit_session()
        config_index = self._get_config_index()
        if config_index is not None:
            config_index.invalidate()
        return commit_hash

    def reset(
        self,
        target: str,
        *,
        mode: str = "soft",
    ) -> str:
        """Reset HEAD to a target commit.

        Stores the current HEAD as ORIG_HEAD before moving.

        Args:
            target: A commit hash, branch name, or hash prefix.
            mode: ``"soft"`` (default) or ``"hard"``.  In Trace both behave
                identically (no working directory to clean).

        Returns:
            The resolved target commit hash (new HEAD).

        Raises:
            CommitNotFoundError: If target cannot be resolved.
        """
        self._check_open()
        from tract.operations.navigation import reset as _reset

        resolved = self.resolve(target)
        result = _reset(resolved, mode, self._tract_id, self._ref_repo)  # type: ignore[arg-type]
        self._commit_session()
        return result

    def list(self) -> list[BranchInfo]:
        """List all branches with current branch indicator.

        Returns:
            List of :class:`BranchInfo` with ``is_current=True`` for
            the active branch.
        """
        from tract.models.branch import BranchInfo
        from tract.operations.branch import list_branches

        branch_names = list_branches(self._tract_id, self._ref_repo)
        current = self._ref_repo.get_current_branch(self._tract_id)

        branches: list[BranchInfo] = []
        for name in branch_names:
            commit_hash = self._ref_repo.get_branch(self._tract_id, name)
            if commit_hash is not None:
                branches.append(
                    BranchInfo(
                        name=name,
                        commit_hash=commit_hash,
                        is_current=(name == current),
                    )
                )
        return branches

    def delete(self, name: str, *, force: bool = False) -> None:
        """Delete a branch.

        Args:
            name: Branch name to delete.
            force: If True, delete even if branch has unmerged commits.

        Raises:
            BranchNotFoundError: If branch doesn't exist.
            TraceError: If trying to delete the current branch.
            UnmergedBranchError: If branch has unmerged commits (without force).
        """
        self._check_open()
        from tract.operations.branch import delete_branch

        delete_branch(
            name,
            self._tract_id,
            self._ref_repo,
            self._commit_repo,
            self._parent_repo,
            force=force,
        )
        self._commit_session()

    def resolve(self, ref_or_prefix: str) -> str:
        """Resolve a commit reference to a full commit hash.

        Resolution order:
        1. Full commit hash (exact match)
        2. Branch name
        3. Hash prefix (min 4 chars)

        Args:
            ref_or_prefix: A commit hash, branch name, or hash prefix.

        Returns:
            The full commit hash.

        Raises:
            CommitNotFoundError: If no commit can be resolved.
            AmbiguousPrefixError: If a prefix matches multiple commits.
        """
        from tract.operations.navigation import resolve_commit as _resolve

        return _resolve(
            ref_or_prefix, self._tract_id, self._commit_repo, self._ref_repo
        )
