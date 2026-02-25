"""Rebase and cherry-pick operations for Trace.

Implements commit replay with new parentage, EDIT target remapping detection,
and semantic safety checks for reordering.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tract.exceptions import ImportCommitError, RebaseError, SemanticSafetyError
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.merge import (
    ImportIssue,
    ImportResult,
    RebaseResult,
    RebaseWarning,
)
from tract.operations.dag import find_merge_base, get_all_ancestors, get_branch_commits

if TYPE_CHECKING:
    from tract.engine.commit import CommitEngine
    from tract.llm.protocols import ResolverCallable
    from tract.storage.repositories import (
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
        OperationEventRepository,
        RefRepository,
    )
    from tract.storage.schema import CommitRow


from tract.operations import row_to_info as _row_to_info


def _load_content_model(blob_repo: BlobRepository, content_hash: str) -> object | None:
    """Load content from a blob and return a validated Pydantic model.

    Returns None if the blob cannot be found or parsed.
    """
    from tract.models.content import validate_content

    blob = blob_repo.get(content_hash)
    if blob is None:
        return None
    try:
        data = json.loads(blob.payload_json)
        return validate_content(data)
    except (json.JSONDecodeError, TypeError, Exception):
        return None


def replay_commit(
    original_row: CommitRow,
    new_parent_hash: str | None,
    commit_engine: CommitEngine,
    blob_repo: BlobRepository,
    *,
    edit_target_remap: str | None = None,
) -> CommitInfo:
    """Replay a single commit with a new parent, creating a new commit.

    The caller must ensure HEAD is positioned at new_parent_hash before calling.
    The commit engine reads HEAD internally to set the parent.

    Args:
        original_row: The original CommitRow to replay.
        new_parent_hash: The new parent hash (for documentation; HEAD must be here).
        commit_engine: Commit engine for creating the new commit.
        blob_repo: Blob repository for loading original content.
        edit_target_remap: If provided, override the edit_target field.

    Returns:
        CommitInfo for the newly created replayed commit.

    Raises:
        RebaseError: If the original content cannot be loaded.
    """
    # Load original content model from blob
    content = _load_content_model(blob_repo, original_row.content_hash)
    if content is None:
        raise RebaseError(
            f"Cannot replay commit {original_row.commit_hash}: "
            f"blob {original_row.content_hash} not found or invalid"
        )

    # Determine edit_target
    edit_target = edit_target_remap
    if edit_target is None and original_row.edit_target is not None:
        edit_target = original_row.edit_target

    # Create the new commit via the engine (engine reads HEAD for parent)
    return commit_engine.create_commit(
        content=content,  # type: ignore[arg-type]
        operation=original_row.operation,
        message=original_row.message,
        edit_target=edit_target if original_row.operation == CommitOperation.EDIT else None,
        metadata=dict(original_row.metadata_json) if original_row.metadata_json else None,
        generation_config=(
            dict(original_row.generation_config_json)
            if original_row.generation_config_json
            else None
        ),
    )


def import_commit(
    commit_hash: str,
    tract_id: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    blob_repo: BlobRepository,
    commit_engine: CommitEngine,
    parent_repo: CommitParentRepository | None = None,
    *,
    resolver: ResolverCallable | None = None,
    event_repo: OperationEventRepository | None = None,
) -> ImportResult:
    """Import a commit onto the current branch (replaces cherry-pick).

    Creates a new commit with the same content but new parentage (current HEAD).
    Optionally records an "import" event for provenance tracking.

    Args:
        commit_hash: Hash of the commit to import.
        tract_id: The tract identifier.
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        blob_repo: Blob repository.
        commit_engine: Commit engine for creating the new commit.
        parent_repo: Optional parent repository for multi-parent traversal.
        resolver: Optional resolver for handling issues.
        event_repo: Optional operation event repository for provenance.

    Returns:
        ImportResult describing the outcome.

    Raises:
        ImportCommitError: If issues detected and no resolver, or resolver aborts.
    """
    # Get the commit to import
    original_row = commit_repo.get(commit_hash)
    if original_row is None:
        raise ImportCommitError(f"Commit not found: {commit_hash}")

    original_info = _row_to_info(original_row)

    # Get current HEAD
    current_head = ref_repo.get_head(tract_id)

    # Build target branch head info for issue context
    target_head_info = None
    if current_head is not None:
        target_row = commit_repo.get(current_head)
        if target_row is not None:
            target_head_info = _row_to_info(target_row)

    # Check for issues
    issues: list[ImportIssue] = []

    if original_row.operation == CommitOperation.EDIT and original_row.edit_target is not None:
        # Check if the edit_target exists in current branch's history
        if current_head is not None:
            ancestors = get_all_ancestors(current_head, commit_repo, parent_repo)
            if original_row.edit_target not in ancestors:
                issues.append(
                    ImportIssue(
                        issue_type="edit_target_missing",
                        commit=original_info,
                        target_branch_head=target_head_info,
                        missing_target=original_row.edit_target,
                        description=(
                            f"EDIT commit targets {original_row.edit_target[:12]}... "
                            f"which does not exist on the current branch"
                        ),
                    )
                )
        else:
            # No commits on current branch, EDIT target definitely missing
            issues.append(
                ImportIssue(
                    issue_type="edit_target_missing",
                    commit=original_info,
                    target_branch_head=None,
                    missing_target=original_row.edit_target,
                    description=(
                        f"EDIT commit targets {original_row.edit_target[:12]}... "
                        f"but current branch has no commits"
                    ),
                )
            )

    # Handle issues
    resolved_content = None
    if issues:
        if resolver is None:
            raise ImportCommitError(
                f"Import has {len(issues)} issue(s): "
                + "; ".join(i.description for i in issues)
            )

        # Call resolver for each issue
        for issue in issues:
            resolution = resolver(issue)
            if resolution.action == "abort":
                raise ImportCommitError(
                    f"Resolver aborted import: {resolution.reasoning}"
                )
            if resolution.action == "skip":
                return ImportResult(
                    original_commit=original_info,
                    new_commit=None,
                    issues=issues,
                )
            if resolution.action == "resolved" and resolution.content_text is not None:
                resolved_content = resolution.content_text

    # Create the new commit
    if resolved_content is not None:
        # Use resolved content -- create as APPEND since EDIT target is missing.
        # Preserve original operation provenance in metadata for auditing.
        from tract.models.content import FreeformContent

        new_content = FreeformContent(payload={"text": resolved_content})
        meta = dict(original_row.metadata_json) if original_row.metadata_json else {}
        meta["original_operation"] = "EDIT"
        meta["original_edit_target"] = original_row.edit_target
        new_info = commit_engine.create_commit(
            content=new_content,
            operation=CommitOperation.APPEND,
            message=original_row.message,
            metadata=meta,
            generation_config=(
                dict(original_row.generation_config_json)
                if original_row.generation_config_json
                else None
            ),
        )
    else:
        # Normal replay -- HEAD is already at current branch tip
        new_info = replay_commit(
            original_row=original_row,
            new_parent_hash=current_head,
            commit_engine=commit_engine,
            blob_repo=blob_repo,
        )

    result = ImportResult(
        original_commit=original_info,
        new_commit=new_info,
        issues=issues,
    )

    # Record import event for provenance
    if event_repo is not None:
        import uuid as _uuid
        from datetime import datetime, timezone
        event_id = _uuid.uuid4().hex
        event_repo.save_event(
            event_id=event_id,
            tract_id=tract_id,
            event_type="import",
            branch_name=None,
            created_at=datetime.now(timezone.utc),
            original_tokens=0,
            compressed_tokens=0,
            params_json={"original_commit": commit_hash},
        )
        event_repo.add_commit(event_id, commit_hash, "source", 0)
        if result.new_commit is not None:
            event_repo.add_commit(event_id, result.new_commit.commit_hash, "result", 0)

    return result


def plan_rebase(
    tract_id: str,
    target_branch: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository,
    *,
    resolver: ResolverCallable | None = None,
) -> tuple[list[CommitRow], str, list[RebaseWarning], str, str] | None:
    """Plan phase of rebase -- determine what to replay and check safety.

    Returns None if no rebase is needed (already up-to-date).

    Args:
        tract_id: The tract identifier.
        target_branch: Name of the branch to rebase onto.
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        parent_repo: Parent repository for multi-parent traversal.
        resolver: Optional resolver for semantic safety warnings.

    Returns:
        Tuple of (commits_to_replay, target_tip, warnings, current_branch, current_tip)
        or None if no rebase needed.

    Raises:
        RebaseError: On detached HEAD, merge commits in range, resolver abort.
        SemanticSafetyError: If safety warnings and no resolver.
    """
    # Get current branch
    current_branch = ref_repo.get_current_branch(tract_id)
    if current_branch is None:
        raise RebaseError("Cannot rebase in detached HEAD state")

    current_tip = ref_repo.get_head(tract_id)
    if current_tip is None:
        raise RebaseError("Cannot rebase: no commits on current branch")

    # Get target branch tip
    target_tip = ref_repo.get_branch(tract_id, target_branch)
    if target_tip is None:
        from tract.exceptions import BranchNotFoundError

        raise BranchNotFoundError(target_branch)

    # If current tip is already an ancestor of target (or same), nothing to do
    if current_tip == target_tip:
        return None

    # Find merge base
    merge_base = find_merge_base(commit_repo, parent_repo, current_tip, target_tip)

    # If target is already an ancestor of current (current is ahead), nothing to replay
    if merge_base == target_tip:
        return None

    # Collect commits to replay (merge_base..current_tip, chronological order)
    if merge_base is not None:
        commits_to_replay = get_branch_commits(commit_repo, parent_repo, current_tip, merge_base)
    else:
        # No common ancestor -- replay all commits on current branch
        commits_to_replay = list(reversed(list(commit_repo.get_ancestors(current_tip))))

    if not commits_to_replay:
        return None

    # Pre-flight: block if any commit in replay range has merge parents
    for c in commits_to_replay:
        parents = parent_repo.get_parents(c.commit_hash)
        if parents:
            raise RebaseError("Cannot rebase branch containing merge commits")

    # Build info list for original commits
    original_infos = [_row_to_info(c) for c in commits_to_replay]

    # Get target branch ancestors for EDIT target checking
    target_ancestors = get_all_ancestors(target_tip, commit_repo, parent_repo)

    # Semantic safety checks
    warnings: list[RebaseWarning] = []

    # Get target tip info for warning context
    target_tip_row = commit_repo.get(target_tip)
    target_tip_info = _row_to_info(target_tip_row) if target_tip_row else None

    for original_row, original_info in zip(commits_to_replay, original_infos):
        if original_row.operation == CommitOperation.EDIT and original_row.edit_target is not None:
            # Check if EDIT target exists in target branch history
            if original_row.edit_target not in target_ancestors:
                warnings.append(
                    RebaseWarning(
                        warning_type="edit_target_missing",
                        commit=original_info,
                        new_base=target_tip_info,
                        description=(
                            f"EDIT commit targets {original_row.edit_target[:12]}... "
                            f"which does not exist on target branch '{target_branch}'"
                        ),
                    )
                )

    # Handle warnings (only if resolver provided)
    if warnings:
        if resolver is None:
            raise SemanticSafetyError(
                f"Rebase has {len(warnings)} semantic safety warning(s): "
                + "; ".join(w.description for w in warnings)
            )

        for warning in warnings:
            resolution = resolver(warning)
            if resolution.action == "abort":
                raise RebaseError(
                    f"Resolver aborted rebase: {resolution.reasoning}"
                )
            # "resolved" or "skip" -- continue with the rebase

    return (commits_to_replay, target_tip, warnings, current_branch, current_tip)


def execute_rebase(
    tract_id: str,
    commits_to_replay: list[CommitRow],
    target_tip: str,
    current_branch: str,
    current_tip: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository,
    blob_repo: BlobRepository,
    commit_engine: CommitEngine,
    *,
    event_repo: OperationEventRepository | None = None,
    warnings: list[RebaseWarning] | None = None,
) -> RebaseResult:
    """Execute phase of rebase -- replay commits onto target.

    Args:
        tract_id: The tract identifier.
        commits_to_replay: Ordered list of CommitRow to replay.
        target_tip: Hash of the target branch tip.
        current_branch: Name of the current branch.
        current_tip: Original tip of the current branch (for rollback).
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        parent_repo: Parent repository.
        blob_repo: Blob repository.
        commit_engine: Commit engine for creating replayed commits.
        event_repo: Optional operation event repository for provenance.
        warnings: Optional list of warnings from the plan phase.

    Returns:
        RebaseResult describing the outcome.
    """
    original_infos = [_row_to_info(c) for c in commits_to_replay]

    # Replay commits atomically
    # Move HEAD to target branch tip (detach)
    ref_repo.detach_head(tract_id, target_tip)

    try:
        replayed_infos: list[CommitInfo] = []
        current_replay_parent = target_tip

        for original_row in commits_to_replay:
            # Replay the commit -- HEAD is at current_replay_parent
            new_info = replay_commit(
                original_row=original_row,
                new_parent_hash=current_replay_parent,
                commit_engine=commit_engine,
                blob_repo=blob_repo,
            )
            replayed_infos.append(new_info)
            current_replay_parent = new_info.commit_hash

        # Update current branch ref to point at the last replayed commit
        new_head = replayed_infos[-1].commit_hash
        ref_repo.set_branch(tract_id, current_branch, new_head)

        # Re-attach HEAD to the current branch
        ref_repo.attach_head(tract_id, current_branch)

    except Exception:
        # On any failure, re-attach HEAD to original branch position
        # The session will be rolled back by the caller (Tract facade)
        ref_repo.set_branch(tract_id, current_branch, current_tip)
        ref_repo.attach_head(tract_id, current_branch)
        raise

    # Record reorganize event for provenance
    if event_repo is not None:
        import uuid as _uuid
        from datetime import datetime, timezone
        event_id = _uuid.uuid4().hex
        event_repo.save_event(
            event_id=event_id,
            tract_id=tract_id,
            event_type="reorganize",
            branch_name=current_branch,
            created_at=datetime.now(timezone.utc),
            original_tokens=0,
            compressed_tokens=0,
            params_json={"target_branch": "rebase"},
        )
        for pos, orig in enumerate(commits_to_replay):
            event_repo.add_commit(event_id, orig.commit_hash, "source", pos)
        for pos, replayed in enumerate(replayed_infos):
            event_repo.add_commit(event_id, replayed.commit_hash, "result", pos)

    return RebaseResult(
        replayed_commits=replayed_infos,
        original_commits=original_infos,
        warnings=warnings or [],
        new_head=new_head,
    )


def rebase(
    tract_id: str,
    target_branch: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository,
    blob_repo: BlobRepository,
    commit_engine: CommitEngine,
    *,
    resolver: ResolverCallable | None = None,
    event_repo: OperationEventRepository | None = None,
) -> RebaseResult:
    """Rebase the current branch onto a target branch.

    Replays commits from the current branch onto the target branch tip,
    producing new commits with new hashes and parentage.

    Args:
        tract_id: The tract identifier.
        target_branch: Name of the branch to rebase onto.
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        parent_repo: Parent repository for multi-parent traversal.
        blob_repo: Blob repository.
        commit_engine: Commit engine for creating replayed commits.
        resolver: Optional resolver for semantic safety warnings.

    Returns:
        RebaseResult describing the outcome.

    Raises:
        RebaseError: On merge commits in range, resolver abort, or other errors.
        SemanticSafetyError: If safety warnings detected and no resolver.
    """
    plan = plan_rebase(
        tract_id, target_branch, commit_repo, ref_repo, parent_repo,
        resolver=resolver,
    )

    if plan is None:
        current_tip = ref_repo.get_head(tract_id)
        return RebaseResult(new_head=current_tip or "")

    commits_to_replay, target_tip, warnings, current_branch, current_tip = plan

    return execute_rebase(
        tract_id, commits_to_replay, target_tip, current_branch, current_tip,
        commit_repo, ref_repo, parent_repo, blob_repo, commit_engine,
        event_repo=event_repo, warnings=warnings,
    )
