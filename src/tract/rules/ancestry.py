"""Shared DAG ancestry walk for the rule engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.storage.repositories import (
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
    )
    from tract.storage.schema import CommitRow


def walk_ancestry(
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    head_hash: str,
    *,
    content_type_filter: set[str] | None = None,
    parent_repo: CommitParentRepository | None = None,
) -> list[CommitRow]:
    """Walk DAG ancestry from head, optionally filtering by content type.

    Returns commits in root-to-head order.

    Args:
        commit_repo: Commit storage.
        blob_repo: Blob storage (unused currently, reserved for content inspection).
        head_hash: Starting commit hash.
        content_type_filter: If provided, only include commits whose
            content_type is in this set. None = include all.
        parent_repo: If provided, used for merge-parent walking.
    """
    # get_ancestors returns newest-first; reverse to root-first
    ancestors = list(commit_repo.get_ancestors(head_hash))

    # Handle merge parents if parent_repo is available
    if parent_repo is not None:
        seen = {c.commit_hash for c in ancestors}
        extra: list[CommitRow] = []
        for commit in ancestors:
            merge_parents = parent_repo.get_parents(commit.commit_hash)
            for parent_hash in merge_parents:
                if parent_hash not in seen:
                    # Walk the merge parent's ancestors too
                    merge_ancestors = list(commit_repo.get_ancestors(parent_hash))
                    for ma in merge_ancestors:
                        if ma.commit_hash not in seen:
                            seen.add(ma.commit_hash)
                            extra.append(ma)
        if extra:
            ancestors.extend(extra)

    # Reverse to root-first order
    ancestors.reverse()

    if content_type_filter is not None:
        ancestors = [c for c in ancestors if c.content_type in content_type_filter]

    return ancestors
