"""Compression operations for Trace.

Implements context compression: summarizing commit chains into shorter
summaries to fit token budgets. Supports three autonomy modes:
- Autonomous (LLM summarization with auto-commit)
- Collaborative (LLM summarization, review before commit)
- Manual (user-provided summary text)

PINNED commits survive compression verbatim. SKIP commits are ignored.
Original commits remain in DB as unreachable (non-destructive).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tract.exceptions import CompressionError
from tract.models.annotations import Priority
from tract.models.commit import CommitOperation
from tract.models.compression import CompressResult, GCResult, PendingCompression
from tract.models.content import DialogueContent
from tract.prompts.summarize import DEFAULT_SUMMARIZE_SYSTEM, build_summarize_prompt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from tract.engine.commit import CommitEngine
    from tract.protocols import TokenCounter
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
        CompressionRepository,
        RefRepository,
    )
    from tract.storage.schema import CommitRow




def _resolve_commit_range(
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    annotation_repo: AnnotationRepository,
    tract_id: str,
    head_hash: str,
    *,
    commits: list[str] | None = None,
    from_commit: str | None = None,
    to_commit: str | None = None,
) -> list[CommitRow]:
    """Resolve commits to compress into an ordered list of CommitRows.

    Uses first-parent chain walking (same as compiler). Returns commits
    in chain order (oldest first).

    Args:
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        annotation_repo: Annotation repository.
        tract_id: Tract identifier.
        head_hash: Current HEAD hash.
        commits: Explicit list of commit hashes to compress.
        from_commit: Start of range (inclusive).
        to_commit: End of range (inclusive).

    Returns:
        List of CommitRow in chain order (oldest first).

    Raises:
        CompressionError: If range is empty or invalid.
    """
    # Walk full first-parent chain from HEAD (newest first)
    all_ancestors = list(commit_repo.get_ancestors(head_hash))

    if not all_ancestors:
        raise CompressionError("No commits to compress")

    # Reverse to oldest-first order for processing
    chain = list(reversed(all_ancestors))

    if commits is not None:
        # Explicit commit list: filter chain to those hashes, preserve chain order
        commit_set = set(commits)
        result = [row for row in chain if row.commit_hash in commit_set]
        missing = commit_set - {r.commit_hash for r in result}
        if missing:
            raise CompressionError(
                f"Commits not found in current chain: {', '.join(sorted(missing))}"
            )
        if not result:
            raise CompressionError("No commits matched the provided list")
        return result

    if from_commit is not None or to_commit is not None:
        # Range: filter chain to from_commit..to_commit (inclusive)
        chain_hashes = [r.commit_hash for r in chain]

        start_idx = 0
        end_idx = len(chain) - 1

        if from_commit is not None:
            try:
                start_idx = chain_hashes.index(from_commit)
            except ValueError:
                raise CompressionError(f"from_commit not found in chain: {from_commit}")

        if to_commit is not None:
            try:
                end_idx = chain_hashes.index(to_commit)
            except ValueError:
                raise CompressionError(f"to_commit not found in chain: {to_commit}")

        if start_idx > end_idx:
            raise CompressionError(
                "Invalid range: from_commit is after to_commit in chain"
            )

        result = chain[start_idx : end_idx + 1]
        if not result:
            raise CompressionError("Empty commit range")
        return result

    # Default: return all commits (full chain)
    return chain


def _classify_by_priority(
    range_commits: list[CommitRow],
    annotation_repo: AnnotationRepository,
    preserve: list[str] | None = None,
) -> tuple[list[CommitRow], list[CommitRow], list[CommitRow]]:
    """Classify commits by their priority annotation.

    Args:
        range_commits: Commits to classify.
        annotation_repo: For looking up priorities.
        preserve: Additional hashes to treat as PINNED for this invocation.

    Returns:
        (pinned_commits, normal_commits, skip_commits)
    """
    preserve_set = set(preserve) if preserve else set()
    target_hashes = [r.commit_hash for r in range_commits]
    latest_annotations = annotation_repo.batch_get_latest(target_hashes)

    pinned: list[CommitRow] = []
    normal: list[CommitRow] = []
    skip: list[CommitRow] = []

    for row in range_commits:
        h = row.commit_hash

        # Check preserve list first (temporary PINNED for this invocation)
        if h in preserve_set:
            pinned.append(row)
            continue

        annotation = latest_annotations.get(h)
        if annotation is not None:
            if annotation.priority == Priority.PINNED:
                pinned.append(row)
            elif annotation.priority == Priority.SKIP:
                skip.append(row)
            else:
                normal.append(row)
        else:
            normal.append(row)

    return pinned, normal, skip


def _partition_around_pinned(
    range_commits: list[CommitRow],
    pinned_hashes: set[str],
    skip_hashes: set[str],
) -> list[list[CommitRow]]:
    """Partition commits into groups of consecutive NORMAL commits.

    PINNED commits act as boundaries. SKIP commits are excluded entirely.

    Args:
        range_commits: All commits in chain order (oldest first).
        pinned_hashes: Set of hashes that are PINNED.
        skip_hashes: Set of hashes that are SKIP.

    Returns:
        List of groups, where each group is consecutive NORMAL commits.
    """
    groups: list[list[CommitRow]] = []
    current_group: list[CommitRow] = []

    for row in range_commits:
        h = row.commit_hash
        if h in pinned_hashes:
            # PINNED: boundary -- save current group if non-empty
            if current_group:
                groups.append(current_group)
                current_group = []
        elif h in skip_hashes:
            # SKIP: just skip entirely
            continue
        else:
            # NORMAL: add to current group
            current_group.append(row)

    # Don't forget the last group
    if current_group:
        groups.append(current_group)

    return groups


def _build_messages_text(
    group: list[CommitRow],
    blob_repo: BlobRepository,
) -> str:
    """Build text representation of a group of commits for LLM summarization.

    Args:
        group: List of CommitRow in chain order.
        blob_repo: For loading blob content.

    Returns:
        Formatted text with role labels.
    """
    parts: list[str] = []

    for row in group:
        blob = blob_repo.get(row.content_hash)
        if blob is None:
            logger.warning("Blob not found for content_hash=%s", row.content_hash)
            parts.append("[content unavailable]")
            continue

        try:
            data = json.loads(blob.payload_json)
        except (json.JSONDecodeError, TypeError):
            parts.append("[content unavailable]")
            continue

        # Format based on content type
        role = data.get("role", row.content_type)
        text = data.get("text", "")
        if not text:
            if "content" in data and isinstance(data["content"], str):
                text = data["content"]
            elif "payload" in data:
                text = json.dumps(data["payload"], sort_keys=True)
            else:
                text = json.dumps(data, sort_keys=True)

        parts.append(f"[{role}]: {text}")

    return "\n\n".join(parts)


def _reconstruct_content(
    commit_row: CommitRow,
    blob_repo: BlobRepository,
    type_registry: dict[str, type] | None = None,
) -> BaseModel:
    """Reconstruct a content model from a commit's blob.

    Args:
        commit_row: The commit whose content to reconstruct.
        blob_repo: For loading the blob.
        type_registry: Optional custom type registry.

    Returns:
        Validated Pydantic content model.

    Raises:
        CompressionError: If blob not found or can't be parsed.
    """
    from tract.models.content import validate_content

    blob = blob_repo.get(commit_row.content_hash)
    if blob is None:
        raise CompressionError(
            f"Blob not found for commit {commit_row.commit_hash}"
        )

    try:
        data = json.loads(blob.payload_json)
        return validate_content(data, custom_registry=type_registry)
    except Exception as exc:
        raise CompressionError(
            f"Failed to reconstruct content for commit {commit_row.commit_hash}: {exc}"
        ) from exc


def _summarize_group(
    messages_text: str,
    llm_client: object,
    token_counter: TokenCounter,
    *,
    target_tokens: int | None = None,
    instructions: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Summarize a group of messages using the LLM client.

    Args:
        messages_text: Text to summarize.
        llm_client: LLM client with a ``chat(messages)`` method that returns
            an OpenAI-style response dict::

                {"choices": [{"message": {"content": "..."}}], ...}

        token_counter: For token counting.
        target_tokens: Optional target token count.
        instructions: Optional additional instructions.
        system_prompt: Optional custom system prompt.

    Returns:
        Summary text string.

    Raises:
        CompressionError: If LLM returns empty or malformed response.
    """
    system = system_prompt if system_prompt is not None else DEFAULT_SUMMARIZE_SYSTEM
    user_prompt = build_summarize_prompt(
        messages_text,
        target_tokens=target_tokens,
        instructions=instructions,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    response = llm_client.chat(messages)  # type: ignore[union-attr]

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CompressionError(f"Invalid LLM response structure: {exc}") from exc

    if not content or not content.strip():
        raise CompressionError("LLM returned empty summary")

    return content


def compress_range(
    tract_id: str,
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    annotation_repo: AnnotationRepository,
    ref_repo: RefRepository,
    commit_engine: CommitEngine,
    token_counter: TokenCounter,
    compression_repo: CompressionRepository,
    parent_repo: CommitParentRepository,
    *,
    commits: list[str] | None = None,
    from_commit: str | None = None,
    to_commit: str | None = None,
    target_tokens: int | None = None,
    preserve: list[str] | None = None,
    auto_commit: bool = True,
    llm_client: object | None = None,
    content: str | None = None,
    instructions: str | None = None,
    system_prompt: str | None = None,
    type_registry: dict[str, type] | None = None,
) -> CompressResult | PendingCompression:
    """Core compression operation.

    Compresses a range of commits into summaries, preserving PINNED commits
    and ignoring SKIP commits. Supports three modes:
    - Manual (content provided): uses user text as summary
    - LLM (llm_client provided): uses LLM for summarization
    - Error: neither provided

    Args:
        tract_id: Tract identifier.
        commit_repo: Commit repository.
        blob_repo: Blob repository.
        annotation_repo: Annotation repository.
        ref_repo: Ref repository.
        commit_engine: Commit engine for creating commits.
        token_counter: Token counter.
        compression_repo: Compression provenance repository.
        parent_repo: Commit parent repository.
        commits: Optional explicit list of commit hashes.
        from_commit: Optional range start (inclusive).
        to_commit: Optional range end (inclusive).
        target_tokens: Optional target token count for summaries.
        preserve: Optional list of hashes to treat as PINNED.
        auto_commit: If True, commit immediately. If False, return PendingCompression.
        llm_client: Optional LLM client for summarization.
        content: Optional manual summary text (bypasses LLM).
        instructions: Optional LLM instructions.
        system_prompt: Optional custom system prompt for LLM.
        type_registry: Optional custom content type registry.

    Returns:
        CompressResult (if auto_commit=True) or PendingCompression (if auto_commit=False).

    Raises:
        CompressionError: On various error conditions.
    """
    # a. Resolve HEAD and current branch
    head_hash = ref_repo.get_head(tract_id)
    if head_hash is None:
        raise CompressionError("No commits to compress")

    branch_name = ref_repo.get_current_branch(tract_id)

    # b. Resolve commit range
    range_commits = _resolve_commit_range(
        commit_repo, ref_repo, annotation_repo, tract_id, head_hash,
        commits=commits, from_commit=from_commit, to_commit=to_commit,
    )

    # c. Classify by priority
    pinned_commits, normal_commits, skip_commits = _classify_by_priority(
        range_commits, annotation_repo, preserve=preserve,
    )

    # d. Nothing to compress?
    if not normal_commits:
        raise CompressionError(
            "Nothing to compress -- all commits are pinned or skipped"
        )

    # e. Partition around pinned
    pinned_hashes = {r.commit_hash for r in pinned_commits}
    skip_hashes = {r.commit_hash for r in skip_commits}
    groups = _partition_around_pinned(range_commits, pinned_hashes, skip_hashes)

    # f. Generate summaries
    if content is not None:
        # Manual mode: single summary for all groups
        if len(groups) > 1:
            raise CompressionError(
                f"Manual mode provides a single summary but PINNED commits "
                f"create {len(groups)} separate groups. Use LLM mode "
                f"(configure_llm()) for multi-group compression, or remove "
                f"PINNED annotations from interleaving commits."
            )
        summaries = [content]
    elif llm_client is not None:
        # LLM mode: one summary per group
        summaries = []
        for group in groups:
            text = _build_messages_text(group, blob_repo)
            summary = _summarize_group(
                text, llm_client, token_counter,
                target_tokens=target_tokens,
                instructions=instructions,
                system_prompt=system_prompt,
            )
            summaries.append(summary)
    else:
        raise CompressionError(
            "No LLM client configured and no manual content provided. "
            "Call configure_llm() first or pass content='...'."
        )

    # g. Calculate token counts
    original_tokens = sum(c.token_count for c in normal_commits)
    estimated_tokens = sum(token_counter.count_text(s) for s in summaries)

    # h. Collaborative mode: return PendingCompression
    if not auto_commit:
        pending = PendingCompression(
            summaries=summaries,
            source_commits=[c.commit_hash for c in normal_commits],
            preserved_commits=[c.commit_hash for c in pinned_commits],
            original_tokens=original_tokens,
            estimated_tokens=estimated_tokens,
        )
        # Store context needed for later commit
        pending._range_commits = range_commits
        pending._pinned_commits = pinned_commits
        pending._normal_commits = normal_commits
        pending._pinned_hashes = pinned_hashes
        pending._skip_hashes = skip_hashes
        pending._groups = groups
        pending._branch_name = branch_name
        pending._target_tokens = target_tokens
        pending._instructions = instructions
        return pending

    # i. Autonomous mode: commit immediately
    return _commit_compression(
        tract_id=tract_id,
        commit_repo=commit_repo,
        blob_repo=blob_repo,
        ref_repo=ref_repo,
        commit_engine=commit_engine,
        token_counter=token_counter,
        compression_repo=compression_repo,
        summaries=summaries,
        range_commits=range_commits,
        pinned_commits=pinned_commits,
        normal_commits=normal_commits,
        pinned_hashes=pinned_hashes,
        skip_hashes=skip_hashes,
        groups=groups,
        original_tokens=original_tokens,
        target_tokens=target_tokens,
        instructions=instructions,
        branch_name=branch_name,
        type_registry=type_registry,
    )


def _commit_compression(
    *,
    tract_id: str,
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    ref_repo: RefRepository,
    commit_engine: CommitEngine,
    token_counter: TokenCounter,
    compression_repo: CompressionRepository,
    summaries: list[str],
    range_commits: list[CommitRow],
    pinned_commits: list[CommitRow],
    normal_commits: list[CommitRow],
    pinned_hashes: set[str],
    skip_hashes: set[str],
    groups: list[list[CommitRow]],
    original_tokens: int,
    target_tokens: int | None,
    instructions: str | None,
    branch_name: str | None,
    type_registry: dict[str, type] | None = None,
) -> CompressResult:
    """Create summary commits, interleave with PINNED, record provenance.

    This is the core commit logic shared between autonomous and collaborative modes.

    Args:
        See compress_range() for parameter descriptions.

    Returns:
        CompressResult with all fields populated.
    """
    # a. Find the parent of the compressed range
    first_in_range = range_commits[0]
    pre_range_parent_hash = first_in_range.parent_hash

    # a2. CRITICAL: Reset branch ref to pre-range parent
    if pre_range_parent_hash is not None:
        ref_repo.update_head(tract_id, pre_range_parent_hash)
    else:
        # Range starts at root: clear HEAD by creating a state where
        # next create_commit will have no parent.
        # We set HEAD to None by updating to a special sentinel...
        # Actually, we need to carefully handle this. The create_commit
        # reads HEAD for parent_hash. If range starts at root,
        # we need HEAD to return None.
        # Let's use the ref_repo to clear HEAD.
        # We'll delete the HEAD ref and the branch ref.
        _clear_head_for_root(ref_repo, tract_id, branch_name)

    # b. Generate compression_id
    compression_id = uuid.uuid4().hex

    # c. Create summary commits in chain order, interleaving with PINNED
    summary_commit_hashes: list[str] = []
    all_new_commit_hashes: list[str] = []
    group_idx = 0

    # Walk the range in order. When we encounter a group boundary or PINNED,
    # handle appropriately.
    # Strategy: iterate through range_commits. Track which group we're in.
    # When we see the first commit of a group, emit the summary.
    # When we see a PINNED commit, re-create it.
    # Skip SKIP commits.

    # Build a map of group-first-commit to group index
    group_first_commits: dict[str, int] = {}
    for gidx, group in enumerate(groups):
        if group:
            group_first_commits[group[0].commit_hash] = gidx

    # Track which groups have been emitted
    emitted_groups: set[int] = set()
    # Track which NORMAL commits have been seen (to know when a group boundary is hit)
    seen_normal: set[str] = set()

    for row in range_commits:
        h = row.commit_hash

        if h in skip_hashes:
            continue

        if h in pinned_hashes:
            # Re-create PINNED commit with correct parent pointer
            content_model = _reconstruct_content(row, blob_repo, type_registry)
            info = commit_engine.create_commit(
                content=content_model,
                operation=row.operation,
                message=row.message or f"Preserved pinned commit",
                metadata=row.metadata_json,
                generation_config=row.generation_config_json,
            )
            all_new_commit_hashes.append(info.commit_hash)
            continue

        # NORMAL commit: check if this is the first commit of a group
        if h in group_first_commits:
            gidx = group_first_commits[h]
            if gidx not in emitted_groups:
                emitted_groups.add(gidx)

                # Determine which summary to use
                summary_text = summaries[gidx]

                # Create summary commit
                n_commits = len(groups[gidx])
                summary_content = DialogueContent(
                    role="assistant",
                    text=summary_text,
                )
                info = commit_engine.create_commit(
                    content=summary_content,
                    message=f"Compressed {n_commits} commits",
                )
                summary_commit_hashes.append(info.commit_hash)
                all_new_commit_hashes.append(info.commit_hash)

        # For non-first NORMAL commits in a group, we skip (already summarized)
        seen_normal.add(h)

    # d. Branch ref now points to the last commit (create_commit updated HEAD)
    new_head = ref_repo.get_head(tract_id)

    # e. Calculate compressed tokens
    compressed_tokens = sum(
        token_counter.count_text(s) for s in summaries
    )
    # Add pinned commit tokens
    for pc in pinned_commits:
        compressed_tokens += pc.token_count

    # f. Save CompressionRecord
    compression_repo.save_record(
        compression_id=compression_id,
        tract_id=tract_id,
        branch_name=branch_name,
        created_at=datetime.now(timezone.utc),
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        target_tokens=target_tokens,
        instructions=instructions,
    )

    # Record sources (normal commits that were compressed)
    for pos, nc in enumerate(normal_commits):
        compression_repo.add_source(compression_id, nc.commit_hash, pos)

    # Record results (summary commits produced)
    for pos, sc_hash in enumerate(summary_commit_hashes):
        compression_repo.add_result(compression_id, sc_hash, pos)

    # g. Return CompressResult
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0.0

    return CompressResult(
        compression_id=compression_id,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        source_commits=tuple(c.commit_hash for c in normal_commits),
        summary_commits=tuple(summary_commit_hashes),
        preserved_commits=tuple(c.commit_hash for c in pinned_commits),
        compression_ratio=ratio,
        new_head=new_head or "",
    )


def check_reorder_safety(
    order: list[str],
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
) -> list[ReorderWarning]:
    """Check for structural safety issues when reordering commits.

    Detects two types of issues:
    - "edit_before_target": An EDIT commit appears before its target in the
      reordered sequence (the target should come first for the edit to make sense).
    - "response_chain_break": A commit references a response_to commit that is
      not present in the reordered set at all.

    Args:
        order: Ordered list of commit hashes (the proposed reorder).
        commit_repo: Commit repository for hash lookups.
        blob_repo: Blob repository (reserved for future use).

    Returns:
        List of ReorderWarning objects (may be empty if no issues found).
    """
    from tract.models.compression import ReorderWarning

    warnings: list[ReorderWarning] = []
    order_set = set(order)
    hash_to_pos = {h: i for i, h in enumerate(order)}

    for h in order:
        row = commit_repo.get(h)
        if row is None:
            continue

        # Check for EDIT before target
        if row.operation == CommitOperation.EDIT and row.response_to:
            if row.response_to in hash_to_pos:
                # Target is in the order list -- check position
                if hash_to_pos[h] < hash_to_pos[row.response_to]:
                    warnings.append(
                        ReorderWarning(
                            warning_type="edit_before_target",
                            commit_hash=row.commit_hash,
                            description=(
                                f"EDIT commit {row.commit_hash[:8]} appears before "
                                f"its target {row.response_to[:8]}"
                            ),
                            severity="structural",
                        )
                    )

        # Check for response_to chain break
        if row.response_to and row.response_to not in order_set:
            warnings.append(
                ReorderWarning(
                    warning_type="response_chain_break",
                    commit_hash=row.commit_hash,
                    description=(
                        f"Commit {row.commit_hash[:8]} references "
                        f"{row.response_to[:8]} which is not in the reordered set"
                    ),
                    severity="semantic",
                )
            )

    return warnings


def _clear_head_for_root(
    ref_repo: RefRepository,
    tract_id: str,
    branch_name: str | None,
) -> None:
    """Clear HEAD so that next create_commit has no parent.

    This handles the edge case where compression starts at the root commit.
    We need to reset the state so the first create_commit in the new chain
    has parent_hash=None.
    """
    # Delete the branch ref so get_head returns None
    if branch_name is not None:
        ref_repo.delete_ref(tract_id, f"refs/heads/{branch_name}")
    # Also need HEAD to resolve to None
    # The simplest approach: set the branch ref to empty via delete
    # But we need to keep the symbolic HEAD -> branch mapping.
    # Actually, we can set the branch to a non-existent hash temporarily,
    # but that's fragile. Better: just delete the branch ref.
    # The get_head implementation follows symbolic refs, so if
    # HEAD -> refs/heads/main and refs/heads/main doesn't exist,
    # get_head returns None. That's what we want.
    pass


# ===========================================================================
# Garbage Collection
# ===========================================================================


def _get_all_reachable(
    tract_id: str,
    ref_repo: RefRepository,
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository,
    *,
    branch: str | None = None,
) -> set[str]:
    """Get all reachable commit hashes from branch tips (and detached HEAD).

    If ``branch`` is specified, only that branch's ancestors are considered.
    Otherwise, all branches AND a potentially detached HEAD are scanned.

    Args:
        tract_id: Tract identifier.
        ref_repo: Ref repository for branch listings.
        commit_repo: Commit repository for hash lookups.
        parent_repo: Parent repository for multi-parent traversal.
        branch: Optional specific branch to scan.

    Returns:
        Set of reachable commit hashes.
    """
    from tract.operations.dag import get_all_ancestors

    reachable: set[str] = set()

    if branch is not None:
        # Scan only the specified branch
        tip = ref_repo.get_branch(tract_id, branch)
        if tip is not None:
            reachable |= get_all_ancestors(tip, commit_repo, parent_repo)
        return reachable

    # Scan ALL branches, passing stop_at to short-circuit on already-known commits
    for branch_name in ref_repo.list_branches(tract_id):
        tip = ref_repo.get_branch(tract_id, branch_name)
        if tip is not None:
            reachable |= get_all_ancestors(
                tip, commit_repo, parent_repo, stop_at=reachable
            )

    # Also check detached HEAD (may point to a commit not on any branch)
    if ref_repo.is_detached(tract_id):
        head = ref_repo.get_head(tract_id)
        if head is not None:
            reachable |= get_all_ancestors(
                head, commit_repo, parent_repo, stop_at=reachable
            )

    return reachable


def _normalize_dt(dt: datetime) -> datetime:
    """Strip timezone info for SQLite naive datetime comparison."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def gc(
    tract_id: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository,
    blob_repo: BlobRepository,
    compression_repo: CompressionRepository,
    *,
    orphan_retention_days: int = 7,
    archive_retention_days: int | None = None,
    branch: str | None = None,
) -> GCResult:
    """Garbage-collect unreachable commits with configurable retention.

    Finds all commits not reachable from any branch tip (or a specific
    branch if ``branch`` is set), then removes eligible ones based on
    age and archive status.

    Retention policies:
    - Orphans (not archived): removed if older than ``orphan_retention_days``
    - Archives (compression sources): preserved by default. Removed only
      if ``archive_retention_days`` is set and the commit is old enough.

    Args:
        tract_id: Tract identifier.
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        parent_repo: Commit parent repository.
        blob_repo: Blob repository.
        compression_repo: Compression provenance repository.
        orphan_retention_days: Days before orphans become eligible for removal.
        archive_retention_days: If set, days before archived commits become
            eligible for removal. None means archives are never removed.
        branch: If set, only this branch's reachability is considered.

    Returns:
        GCResult with removal counts and duration.
    """
    import time

    start = time.monotonic()

    # a. Find reachable commits
    reachable = _get_all_reachable(
        tract_id, ref_repo, commit_repo, parent_repo, branch=branch,
    )

    # b. Find all commits in tract
    all_commits = commit_repo.get_all(tract_id)

    # c. Compute unreachable
    unreachable = [c for c in all_commits if c.commit_hash not in reachable]

    # d. Classify and apply retention
    now = _normalize_dt(datetime.now(timezone.utc))
    commits_to_remove = []
    archives_removed = 0

    for commit in unreachable:
        is_archive = compression_repo.is_source_of(commit.commit_hash)
        created = _normalize_dt(commit.created_at)
        age_days = (now - created).total_seconds() / 86400

        if is_archive:
            # Archive: only remove if archive_retention_days is set and old enough
            if archive_retention_days is not None and age_days >= archive_retention_days:
                commits_to_remove.append(commit)
                archives_removed += 1
            # Otherwise: preserve (skip)
        else:
            # Orphan: remove if old enough
            if age_days >= orphan_retention_days:
                commits_to_remove.append(commit)

    # e. Delete eligible commits
    blobs_removed = 0
    tokens_freed = 0

    for commit in commits_to_remove:
        content_hash = commit.content_hash
        tokens_freed += commit.token_count

        # Clean up compression provenance
        compression_repo.delete_source(commit.commit_hash)
        compression_repo.delete_result(commit.commit_hash)

        # Delete the commit
        commit_repo.delete(commit.commit_hash)

        # Try to delete the blob if no other commit references it
        if blob_repo.delete_if_orphaned(content_hash):
            blobs_removed += 1

    # f. Clean up orphaned CompressionRow records (no sources AND no results left)
    all_compression_ids = compression_repo.get_all_ids(tract_id)
    for cid in all_compression_ids:
        if not compression_repo.get_sources(cid) and not compression_repo.get_results(cid):
            compression_repo.delete_record(cid)

    duration = time.monotonic() - start

    return GCResult(
        commits_removed=len(commits_to_remove),
        blobs_removed=blobs_removed,
        tokens_freed=tokens_freed,
        archives_removed=archives_removed,
        duration_seconds=duration,
    )
