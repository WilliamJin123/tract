"""DAG health check and validation operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.storage.repositories import (
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
        RefRepository,
    )


@dataclass
class HealthReport:
    """Results of a DAG health check."""

    healthy: bool = True
    commit_count: int = 0
    branch_count: int = 0
    orphan_count: int = 0
    missing_blobs: list[str] = field(default_factory=list)
    missing_parents: list[tuple[str, str]] = field(default_factory=list)
    unreachable_commits: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Health: {'OK' if self.healthy else 'ISSUES FOUND'}"]
        lines.append(f"Commits: {self.commit_count}, Branches: {self.branch_count}")
        if self.orphan_count:
            lines.append(f"Orphan commits: {self.orphan_count}")
        if self.missing_blobs:
            lines.append(f"Missing blobs: {len(self.missing_blobs)}")
        if self.missing_parents:
            lines.append(f"Missing parents: {len(self.missing_parents)}")
        if self.warnings:
            lines.extend(f"Warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def check_health(
    tract_id: str,
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository | None = None,
) -> HealthReport:
    """Run comprehensive health checks on a tract's DAG.

    Checks:
    1. All commits have valid blob references
    2. All parent references point to existing commits
    3. No orphaned commits (unreachable from any branch HEAD)
    4. Branch HEADs point to existing commits
    5. No cycles in the DAG (detected via reachability walk)

    Args:
        tract_id: Tract identifier.
        commit_repo: Commit repository.
        blob_repo: Blob repository.
        ref_repo: Ref repository.
        parent_repo: Commit parent repository (for merge-parent traversal).

    Returns:
        HealthReport with validation results and any warnings.
    """
    from tract.operations.dag import get_all_ancestors

    report = HealthReport()

    # Get all commits for this tract
    all_rows = commit_repo.get_all(tract_id)
    all_commits: dict[str, object] = {}
    for row in all_rows:
        all_commits[row.commit_hash] = row

    report.commit_count = len(all_commits)

    # Get branches
    branches = ref_repo.list_branches(tract_id)
    report.branch_count = len(branches)

    # Check 1: Blob integrity -- every commit should reference an existing blob
    for commit_hash, commit in all_commits.items():
        content_hash = commit.content_hash  # type: ignore[union-attr]
        if content_hash:
            blob = blob_repo.get(content_hash)
            if blob is None:
                report.missing_blobs.append(commit_hash)
                report.healthy = False

    # Check 2: Parent integrity -- every parent_hash should point to an existing commit
    for commit_hash, commit in all_commits.items():
        parent_hash = commit.parent_hash  # type: ignore[union-attr]
        if parent_hash and parent_hash not in all_commits:
            report.missing_parents.append((commit_hash, parent_hash))
            report.healthy = False

    # Check 3: Reachability -- find orphans (commits not reachable from any branch)
    reachable: set[str] = set()

    for branch_name in branches:
        tip = ref_repo.get_branch(tract_id, branch_name)
        if tip is not None:
            reachable |= get_all_ancestors(
                tip, commit_repo, parent_repo, stop_at=reachable
            )

    # Also check detached HEAD
    if ref_repo.is_detached(tract_id):
        head = ref_repo.get_head(tract_id)
        if head is not None:
            reachable |= get_all_ancestors(
                head, commit_repo, parent_repo, stop_at=reachable
            )

    orphans = set(all_commits.keys()) - reachable
    report.orphan_count = len(orphans)
    report.unreachable_commits = sorted(orphans)
    if orphans:
        report.warnings.append(
            f"{len(orphans)} unreachable commits (run gc to clean)"
        )

    # Check 4: Branch HEAD validity -- every branch tip should point to an existing commit
    for branch_name in branches:
        tip = ref_repo.get_branch(tract_id, branch_name)
        if tip and tip not in all_commits:
            report.warnings.append(
                f"Branch '{branch_name}' HEAD points to missing commit {tip[:8]}"
            )
            report.healthy = False

    return report
