"""Merge operations for Trace.

Implements merge strategies: fast-forward, clean auto-merge (branch-blocks),
structural conflict detection, and LLM-mediated semantic merge.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tract.exceptions import MergeConflictError, MergeError, NothingToMergeError
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.merge import ConflictInfo, MergeResult
from tract.operations.dag import find_merge_base, get_branch_commits, is_ancestor

if TYPE_CHECKING:
    from tract.engine.commit import CommitEngine
    from tract.llm.protocols import ResolverCallable
    from tract.protocols import TokenCounter
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
        RefRepository,
    )
    from tract.storage.schema import CommitRow


def _row_to_info(row: CommitRow) -> CommitInfo:
    """Convert a CommitRow to CommitInfo."""
    return CommitInfo(
        commit_hash=row.commit_hash,
        tract_id=row.tract_id,
        parent_hash=row.parent_hash,
        content_hash=row.content_hash,
        content_type=row.content_type,
        operation=row.operation,
        response_to=row.response_to,
        message=row.message,
        token_count=row.token_count,
        metadata=row.metadata_json,
        generation_config=row.generation_config_json,
        created_at=row.created_at,
    )


def _load_content_text(blob_repo: BlobRepository, content_hash: str) -> str:
    """Load content text from a blob, returning empty string on failure."""
    blob = blob_repo.get(content_hash)
    if blob is None:
        return ""
    try:
        data = json.loads(blob.payload_json)
        # Try common text fields
        if "text" in data:
            return data["text"]
        if "content" in data and isinstance(data["content"], str):
            return data["content"]
        if "payload" in data:
            return json.dumps(data["payload"], sort_keys=True)
        return json.dumps(data, sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return ""


def detect_conflicts(
    branch_a_commits: list[CommitRow],
    branch_b_commits: list[CommitRow],
    annotation_repo: AnnotationRepository,
    blob_repo: BlobRepository,
    tract_id: str,
    merge_base_hash: str | None,
) -> list[ConflictInfo]:
    """Detect structural merge conflicts between two divergent branches.

    Checks for three conflict types:
    1. both_edit: Both branches EDIT the same target commit.
    2. skip_vs_edit: One branch SKIPs a commit the other EDITs.
    3. edit_plus_append: One branch EDITs a pre-merge-base commit while
       the other has APPENDs (EDITs to post-merge-base commits are fine).

    Args:
        branch_a_commits: Commits unique to branch A (current/target).
        branch_b_commits: Commits unique to branch B (source).
        annotation_repo: For checking SKIP annotations.
        blob_repo: For loading content text.
        tract_id: The tract identifier.
        merge_base_hash: The merge base commit hash for pre/post distinction.

    Returns:
        List of ConflictInfo objects describing each conflict.
    """
    conflicts: list[ConflictInfo] = []

    # Convert rows to infos for ConflictInfo models
    a_infos = [_row_to_info(r) for r in branch_a_commits]
    b_infos = [_row_to_info(r) for r in branch_b_commits]

    # Classify commits by operation
    a_edits = {
        r.response_to: (r, info)
        for r, info in zip(branch_a_commits, a_infos)
        if r.operation == CommitOperation.EDIT and r.response_to is not None
    }
    b_edits = {
        r.response_to: (r, info)
        for r, info in zip(branch_b_commits, b_infos)
        if r.operation == CommitOperation.EDIT and r.response_to is not None
    }

    a_commit_hashes = {r.commit_hash for r in branch_a_commits}
    b_commit_hashes = {r.commit_hash for r in branch_b_commits}

    # --- 1. Both EDIT same target ---
    common_edit_targets = set(a_edits.keys()) & set(b_edits.keys())
    for target in common_edit_targets:
        row_a, info_a = a_edits[target]
        row_b, info_b = b_edits[target]
        conflicts.append(
            ConflictInfo(
                conflict_type="both_edit",
                commit_a=info_a,
                commit_b=info_b,
                content_a_text=_load_content_text(blob_repo, row_a.content_hash),
                content_b_text=_load_content_text(blob_repo, row_b.content_hash),
                target_hash=target,
                branch_a_commits=a_infos,
                branch_b_commits=b_infos,
            )
        )

    # --- 2. SKIP vs EDIT ---
    # Check if branch A has SKIP annotations on commits that branch B EDITs
    for target, (row_b, info_b) in b_edits.items():
        if target in common_edit_targets:
            continue  # Already handled as both_edit
        annotation = annotation_repo.get_latest(target)
        if annotation is not None:
            from tract.models.annotations import Priority

            if annotation.priority == Priority.SKIP:
                # Find the corresponding info for display
                conflicts.append(
                    ConflictInfo(
                        conflict_type="skip_vs_edit",
                        commit_a=info_b,  # The edit commit
                        commit_b=info_b,  # Same commit (the EDIT)
                        content_a_text="[SKIPPED]",
                        content_b_text=_load_content_text(blob_repo, row_b.content_hash),
                        target_hash=target,
                        branch_a_commits=a_infos,
                        branch_b_commits=b_infos,
                    )
                )

    # Check if branch B has SKIP annotations on commits that branch A EDITs
    for target, (row_a, info_a) in a_edits.items():
        if target in common_edit_targets:
            continue
        annotation = annotation_repo.get_latest(target)
        if annotation is not None:
            from tract.models.annotations import Priority

            if annotation.priority == Priority.SKIP:
                conflicts.append(
                    ConflictInfo(
                        conflict_type="skip_vs_edit",
                        commit_a=info_a,
                        commit_b=info_a,
                        content_a_text=_load_content_text(blob_repo, row_a.content_hash),
                        content_b_text="[SKIPPED]",
                        target_hash=target,
                        branch_a_commits=a_infos,
                        branch_b_commits=b_infos,
                    )
                )

    # --- 3. EDIT + APPEND (pre-merge-base edit only) ---
    # Collect all post-merge-base commit hashes (the branch's own divergent history)
    post_merge_base_hashes = a_commit_hashes | b_commit_hashes

    a_has_appends = any(
        r.operation == CommitOperation.APPEND for r in branch_a_commits
    )
    b_has_appends = any(
        r.operation == CommitOperation.APPEND for r in branch_b_commits
    )

    # Check branch A edits targeting pre-merge-base commits while branch B has appends
    for target, (row_a, info_a) in a_edits.items():
        if target in common_edit_targets:
            continue
        # Only flag if the EDIT targets a pre-merge-base commit (shared history)
        if target not in post_merge_base_hashes and b_has_appends:
            # Find the first append in branch B for the conflict pair
            first_b_append = next(
                (info for r, info in zip(branch_b_commits, b_infos) if r.operation == CommitOperation.APPEND),
                b_infos[0] if b_infos else info_a,
            )
            conflicts.append(
                ConflictInfo(
                    conflict_type="edit_plus_append",
                    commit_a=info_a,
                    commit_b=first_b_append,
                    content_a_text=_load_content_text(blob_repo, row_a.content_hash),
                    content_b_text=_load_content_text(blob_repo, first_b_append.content_hash),
                    target_hash=target,
                    branch_a_commits=a_infos,
                    branch_b_commits=b_infos,
                )
            )

    # Check branch B edits targeting pre-merge-base commits while branch A has appends
    for target, (row_b, info_b) in b_edits.items():
        if target in common_edit_targets:
            continue
        if target not in post_merge_base_hashes and a_has_appends:
            first_a_append = next(
                (info for r, info in zip(branch_a_commits, a_infos) if r.operation == CommitOperation.APPEND),
                a_infos[0] if a_infos else info_b,
            )
            conflicts.append(
                ConflictInfo(
                    conflict_type="edit_plus_append",
                    commit_a=first_a_append,
                    commit_b=info_b,
                    content_a_text=_load_content_text(blob_repo, first_a_append.content_hash),
                    content_b_text=_load_content_text(blob_repo, row_b.content_hash),
                    target_hash=target,
                    branch_a_commits=a_infos,
                    branch_b_commits=b_infos,
                )
            )

    return conflicts


def merge_branches(
    tract_id: str,
    source_branch: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository,
    blob_repo: BlobRepository,
    annotation_repo: AnnotationRepository,
    commit_engine: CommitEngine,
    token_counter: TokenCounter,
    *,
    resolver: ResolverCallable | None = None,
    strategy: str = "auto",
    no_ff: bool = False,
) -> MergeResult:
    """Execute a merge of source_branch into the current branch.

    Strategies:
    - Fast-forward: moves branch pointer when possible (unless no_ff=True).
    - Clean merge: auto-merges divergent histories with only APPENDs.
    - Conflict merge: detects structural conflicts, calls resolver if available.
    - Semantic: also runs resolver on full merged context (strategy="semantic").

    Args:
        tract_id: The tract identifier.
        source_branch: Name of the branch to merge in.
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        parent_repo: Commit parent repository.
        blob_repo: Blob repository.
        annotation_repo: Annotation repository.
        commit_engine: Commit engine for creating merge commits.
        token_counter: Token counter.
        resolver: Optional conflict resolver callable.
        strategy: Merge strategy ("auto" or "semantic").
        no_ff: If True, always create a merge commit (no fast-forward).

    Returns:
        MergeResult describing the outcome.

    Raises:
        MergeError: If HEAD is detached or other merge errors.
        NothingToMergeError: If source is already merged.
        MergeConflictError: If conflicts detected and no resolver available
            and no way to return for review (internal use).
    """
    # Get current branch
    current_branch = ref_repo.get_current_branch(tract_id)
    if current_branch is None:
        raise MergeError("Cannot merge in detached HEAD state")

    # Resolve source branch to commit hash
    source_hash = ref_repo.get_branch(tract_id, source_branch)
    if source_hash is None:
        from tract.exceptions import BranchNotFoundError

        raise BranchNotFoundError(source_branch)

    # Get current HEAD hash
    current_hash = ref_repo.get_head(tract_id)
    if current_hash is None:
        raise MergeError("Cannot merge: no commits on current branch")

    # Same commit: nothing to merge
    if current_hash == source_hash:
        raise NothingToMergeError(source_branch)

    # --- Fast-forward check ---
    if is_ancestor(commit_repo, parent_repo, current_hash, source_hash):
        if not no_ff:
            # Fast-forward: move branch pointer to source tip
            ref_repo.set_branch(tract_id, current_branch, source_hash)
            return MergeResult(
                merge_type="fast_forward",
                source_branch=source_branch,
                target_branch=current_branch,
                committed=True,
                merge_commit_hash=source_hash,
                _source_tip_hash=source_hash,
                _target_tip_hash=current_hash,
            )

    # --- Find merge base ---
    merge_base = find_merge_base(commit_repo, parent_repo, current_hash, source_hash)

    # Check if source is already merged (source tip is ancestor of current)
    if merge_base == source_hash:
        raise NothingToMergeError(source_branch)

    # --- Collect branch commits ---
    if merge_base is None:
        # No common ancestor; treat everything as divergent
        a_commits = list(reversed(list(commit_repo.get_ancestors(current_hash))))
        b_commits = list(reversed(list(commit_repo.get_ancestors(source_hash))))
    else:
        a_commits = get_branch_commits(commit_repo, parent_repo, current_hash, merge_base)
        b_commits = get_branch_commits(commit_repo, parent_repo, source_hash, merge_base)

    # --- Detect conflicts ---
    conflicts = detect_conflicts(
        a_commits, b_commits, annotation_repo, blob_repo, tract_id, merge_base
    )

    # --- No conflicts: clean merge ---
    if not conflicts:
        from tract.models.content import FreeformContent

        merge_content = FreeformContent(
            payload={"message": f"Merged {source_branch} into {current_branch}"}
        )
        merge_info = create_merge_commit(
            commit_engine=commit_engine,
            parent_repo=parent_repo,
            content=merge_content,
            parent_hashes=[current_hash, source_hash],
            message=f"Merge branch '{source_branch}' into {current_branch}",
        )
        return MergeResult(
            merge_type="clean",
            source_branch=source_branch,
            target_branch=current_branch,
            merge_base_hash=merge_base,
            auto_merged_content=[_row_to_info(c) for c in a_commits + b_commits],
            committed=True,
            merge_commit_hash=merge_info.commit_hash,
            _source_tip_hash=source_hash,
            _target_tip_hash=current_hash,
        )

    # --- Conflicts exist ---
    result = MergeResult(
        merge_type="conflict",
        source_branch=source_branch,
        target_branch=current_branch,
        merge_base_hash=merge_base,
        conflicts=conflicts,
        auto_merged_content=[_row_to_info(c) for c in a_commits + b_commits],
        _source_tip_hash=source_hash,
        _target_tip_hash=current_hash,
    )

    if resolver is not None:
        # Call resolver for each conflict
        all_resolved = True
        for conflict in conflicts:
            resolution = resolver(conflict)
            if resolution.action == "resolved" and resolution.content_text is not None:
                target_key = conflict.target_hash or conflict.commit_b.commit_hash
                result.resolutions[target_key] = resolution.content_text
                if resolution.reasoning:
                    result.resolution_reasoning[target_key] = resolution.reasoning
                if resolution.generation_config:
                    result.generation_configs[target_key] = resolution.generation_config
            elif resolution.action == "abort":
                raise MergeError(f"Resolver aborted merge: {resolution.reasoning}")
            else:
                all_resolved = False

        if all_resolved and len(result.resolutions) == len(conflicts):
            result.merge_type = "semantic" if strategy == "semantic" else "conflict"

    return result


def create_merge_commit(
    commit_engine: CommitEngine,
    parent_repo: CommitParentRepository,
    content: object,
    parent_hashes: list[str],
    *,
    message: str | None = None,
    metadata: dict | None = None,
    generation_config: dict | None = None,
) -> CommitInfo:
    """Create a merge commit with multiple parents.

    Uses the commit engine for the actual commit creation, then records
    all parent hashes in the commit_parents table.

    Args:
        commit_engine: Commit engine for creating the commit.
        parent_repo: Parent repository for recording multiple parents.
        content: Content model for the merge commit.
        parent_hashes: List of parent hashes [current_branch_tip, source_tip].
        message: Optional commit message.
        metadata: Optional metadata dict.
        generation_config: Optional generation config dict.

    Returns:
        CommitInfo for the new merge commit.
    """
    from pydantic import BaseModel

    # Create the commit using the engine
    # The engine will use HEAD as parent_hash and update HEAD
    info = commit_engine.create_commit(
        content=content,  # type: ignore[arg-type]
        operation=CommitOperation.APPEND,
        message=message,
        metadata=metadata,
        generation_config=generation_config,
    )

    # Record all parents in the commit_parents table
    parent_repo.add_parents(info.commit_hash, parent_hashes)

    return info
