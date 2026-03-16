"""Merge operations for Trace.

Implements merge strategies: fast-forward, clean auto-merge (branch-blocks),
structural conflict detection, and LLM-mediated semantic merge.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tract.exceptions import MergeError, NothingToMergeError
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.merge import ConflictInfo, ConflictType, MergeResult, MergeStrategy
from tract.operations import row_to_info as _row_to_info
from tract.operations.dag import find_merge_base, get_branch_commits, is_ancestor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Any

    from pydantic import BaseModel

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


def _load_content_text(blob_repo: BlobRepository, content_hash: str) -> str:
    """Load content text from a blob, returning a sentinel on failure."""
    blob = blob_repo.get(content_hash)
    if blob is None:
        logger.warning("Blob not found for content_hash=%s", content_hash)
        return "[content unavailable]"
    try:
        data = json.loads(blob.payload_json)
        # Try common text fields
        if "text" in data:
            return data["text"]
        if "content" in data and isinstance(data["content"], str):
            return data["content"]
        if "payload" in data:
            return json.dumps(data["payload"])
        return json.dumps(data)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse blob %s: %s", content_hash, exc)
        return "[content unavailable]"


def _detect_skip_vs_edit(
    edits: dict[str, tuple[CommitRow, CommitInfo]],
    common_edit_targets: set[str],
    annotations_map: dict[str, Any],
    blob_repo: BlobRepository,
    commit_repo: CommitRepository,
    a_infos: list[CommitInfo],
    b_infos: list[CommitInfo],
    *,
    editor_is_a: bool,
) -> list[ConflictInfo]:
    """Detect skip-vs-edit conflicts for one direction.

    When ``editor_is_a=True``, *edits* come from branch A and the SKIP
    annotation is on the other side (B).  When ``False``, it's reversed.

    ``annotations_map`` is a pre-fetched dict from
    ``annotation_repo.batch_get_latest()`` — avoids N+1 queries.
    """
    from tract.models.annotations import Priority

    conflicts: list[ConflictInfo] = []
    for target, (row, info) in edits.items():
        if target in common_edit_targets:
            continue
        annotation = annotations_map.get(target)
        if annotation is None or annotation.priority != Priority.SKIP:
            continue
        target_row = commit_repo.get(target)
        target_info = _row_to_info(target_row) if target_row is not None else info
        edit_text = _load_content_text(blob_repo, row.content_hash)
        conflicts.append(
            ConflictInfo(
                conflict_type=ConflictType.SKIP_VS_EDIT,
                commit_a=info if editor_is_a else target_info,
                commit_b=target_info if editor_is_a else info,
                content_a_text=edit_text if editor_is_a else "[SKIPPED]",
                content_b_text="[SKIPPED]" if editor_is_a else edit_text,
                target_hash=target,
                branch_a_commits=a_infos,
                branch_b_commits=b_infos,
            )
        )
    return conflicts


def detect_conflicts(
    branch_a_commits: list[CommitRow],
    branch_b_commits: list[CommitRow],
    annotation_repo: AnnotationRepository,
    blob_repo: BlobRepository,
    commit_repo: CommitRepository,
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
        r.edit_target: (r, info)
        for r, info in zip(branch_a_commits, a_infos)
        if r.operation == CommitOperation.EDIT and r.edit_target is not None
    }
    b_edits = {
        r.edit_target: (r, info)
        for r, info in zip(branch_b_commits, b_infos)
        if r.operation == CommitOperation.EDIT and r.edit_target is not None
    }

    a_commit_hashes = {r.commit_hash for r in branch_a_commits}
    b_commit_hashes = {r.commit_hash for r in branch_b_commits}

    # --- 1. Both EDIT same target ---
    common_edit_targets = a_edits.keys() & b_edits.keys()
    for target in common_edit_targets:
        row_a, info_a = a_edits[target]
        row_b, info_b = b_edits[target]
        conflicts.append(
            ConflictInfo(
                conflict_type=ConflictType.BOTH_EDIT,
                commit_a=info_a,
                commit_b=info_b,
                content_a_text=_load_content_text(blob_repo, row_a.content_hash),
                content_b_text=_load_content_text(blob_repo, row_b.content_hash),
                target_hash=target,
                branch_a_commits=a_infos,
                branch_b_commits=b_infos,
            )
        )

    # --- 2. SKIP vs EDIT (both directions) ---
    # Batch-fetch annotations for all edit targets to avoid N+1 queries
    all_edit_targets = list(a_edits.keys() | b_edits.keys())
    annotations_map = (
        annotation_repo.batch_get_latest(all_edit_targets)
        if all_edit_targets
        else {}
    )

    conflicts.extend(_detect_skip_vs_edit(
        b_edits, common_edit_targets, annotations_map, blob_repo, commit_repo,
        a_infos, b_infos, editor_is_a=False,
    ))
    conflicts.extend(_detect_skip_vs_edit(
        a_edits, common_edit_targets, annotations_map, blob_repo, commit_repo,
        a_infos, b_infos, editor_is_a=True,
    ))

    # --- 3. EDIT + APPEND (pre-merge-base edit only) ---
    # Collect all post-merge-base commit hashes (the branch's own divergent history)
    post_merge_base_hashes = a_commit_hashes | b_commit_hashes

    a_has_appends = any(
        r.operation == CommitOperation.APPEND for r in branch_a_commits
    )
    b_has_appends = any(
        r.operation == CommitOperation.APPEND for r in branch_b_commits
    )

    # Cache first-append CommitInfo for each branch once, outside the loops.
    # Previously these were recomputed via next() on every iteration (O(n^2)).
    first_b_append = next(
        (info for r, info in zip(branch_b_commits, b_infos) if r.operation == CommitOperation.APPEND),
        b_infos[0] if b_infos else None,
    )
    first_a_append = next(
        (info for r, info in zip(branch_a_commits, a_infos) if r.operation == CommitOperation.APPEND),
        a_infos[0] if a_infos else None,
    )

    # Check branch A edits targeting pre-merge-base commits while branch B has appends
    for target, (row_a, info_a) in a_edits.items():
        if target in common_edit_targets:
            continue
        # Only flag if the EDIT targets a pre-merge-base commit (shared history)
        if target not in post_merge_base_hashes and b_has_appends:
            append_info = first_b_append if first_b_append is not None else info_a
            conflicts.append(
                ConflictInfo(
                    conflict_type=ConflictType.EDIT_PLUS_APPEND,
                    commit_a=info_a,
                    commit_b=append_info,
                    content_a_text=_load_content_text(blob_repo, row_a.content_hash),
                    content_b_text=_load_content_text(blob_repo, append_info.content_hash),
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
            append_info = first_a_append if first_a_append is not None else info_b
            conflicts.append(
                ConflictInfo(
                    conflict_type=ConflictType.EDIT_PLUS_APPEND,
                    commit_a=append_info,
                    commit_b=info_b,
                    content_a_text=_load_content_text(blob_repo, append_info.content_hash),
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
    strategy: str | MergeStrategy = MergeStrategy.AUTO,
    no_ff: bool = False,
) -> MergeResult:
    """Execute a merge of source_branch into the current branch.

    Strategies:
    - Fast-forward: moves branch pointer when possible (unless no_ff=True).
    - Clean merge: auto-merges divergent histories with only APPENDs.
    - Conflict merge: detects structural conflicts, calls resolver if available.
    - Semantic: also runs resolver on full merged context (strategy="semantic").

    The ``strategy`` parameter controls conflict resolution:
    - ``"auto"`` (default): detect conflicts, use resolver or return for review.
    - ``"ours"``: on conflict, always take the current branch's version.
    - ``"theirs"``: on conflict, always take the source branch's version.
    - ``"semantic"``: like auto, but marks result as semantic merge.

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
        strategy: Merge strategy (``"auto"``, ``"ours"``, ``"theirs"``,
            or ``"semantic"``).
        no_ff: If True, always create a merge commit (no fast-forward).

    Returns:
        MergeResult describing the outcome.

    Raises:
        MergeError: If HEAD is detached or other merge errors.
        NothingToMergeError: If source is already merged.
        MergeConflictError: If conflicts detected and no resolver available
            and no way to return for review (internal use).
    """
    # Normalize strategy to MergeStrategy enum
    if isinstance(strategy, str):
        try:
            strategy_enum = MergeStrategy(strategy.lower())
        except ValueError:
            # Allow "semantic" as a pass-through (not a MergeStrategy member)
            if strategy.lower() == "semantic":
                strategy_enum = MergeStrategy.AUTO
            else:
                raise MergeError(
                    f"Unknown merge strategy: {strategy!r}. "
                    f"Use 'auto', 'ours', 'theirs', or 'semantic'."
                )
    else:
        strategy_enum = strategy
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
                source_tip_hash=source_hash,
                target_tip_hash=current_hash,
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
        a_commits = get_branch_commits(commit_repo, current_hash, merge_base)
        b_commits = get_branch_commits(commit_repo, source_hash, merge_base)

    # --- Detect conflicts ---
    conflicts = detect_conflicts(
        a_commits, b_commits, annotation_repo, blob_repo, commit_repo, tract_id, merge_base
    )

    # --- No conflicts: clean merge ---
    if not conflicts:
        from tract.models.content import FreeformContent

        merge_content = FreeformContent(
            payload={"message": f"Merged {source_branch} into {current_branch}"}
        )
        merge_info = create_merge_commit(
            commit_engine=commit_engine,
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
            source_tip_hash=source_hash,
            target_tip_hash=current_hash,
        )

    # --- Conflicts exist ---
    result = MergeResult(
        merge_type="conflict",
        source_branch=source_branch,
        target_branch=current_branch,
        merge_base_hash=merge_base,
        conflicts=conflicts,
        auto_merged_content=[_row_to_info(c) for c in a_commits + b_commits],
        source_tip_hash=source_hash,
        target_tip_hash=current_hash,
    )

    # --- Ours/Theirs: auto-resolve conflicts without a resolver ---
    if strategy_enum == MergeStrategy.OURS:
        for conflict in conflicts:
            target_key = conflict.target_hash or conflict.commit_a.commit_hash
            result.resolutions[target_key] = conflict.content_a_text
            result.resolution_reasoning[target_key] = "Auto-resolved: strategy=ours"
        return result

    if strategy_enum == MergeStrategy.THEIRS:
        for conflict in conflicts:
            target_key = conflict.target_hash or conflict.commit_b.commit_hash
            result.resolutions[target_key] = conflict.content_b_text
            result.resolution_reasoning[target_key] = "Auto-resolved: strategy=theirs"
        return result

    # --- Auto/Semantic: use resolver if available ---
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
            is_semantic = isinstance(strategy, str) and strategy.lower() == "semantic"
            result.merge_type = "semantic" if is_semantic else "conflict"

    return result


def create_merge_commit(
    commit_engine: CommitEngine,
    content: BaseModel,
    parent_hashes: list[str],
    *,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
) -> CommitInfo:
    """Create a merge commit with multiple parents.

    Delegates to ``CommitEngine.create_merge_commit()`` which includes all
    parent hashes in the commit hash computation and records them in the
    commit_parents table.

    Args:
        commit_engine: Commit engine for creating the commit.
        content: Content model for the merge commit.
        parent_hashes: List of parent hashes [current_branch_tip, source_tip].
        message: Optional commit message.
        metadata: Optional metadata dict.
        generation_config: Optional generation config dict.

    Returns:
        CommitInfo for the new merge commit.
    """
    return commit_engine.create_merge_commit(
        content=content,
        parent_hashes=parent_hashes,
        message=message,
        metadata=metadata,
        generation_config=generation_config,
    )
