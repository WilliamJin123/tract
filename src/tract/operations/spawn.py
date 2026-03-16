"""Spawn and collapse operations for multi-agent coordination.

Provides:
- spawn_tract(): Create a child tract linked to a parent via spawn pointer
- collapse_tract(): Compress child tract history into a summary commit in parent
- _head_snapshot(): Compile parent context and seed child with it
- _full_clone(): Replay all parent commits into child tract
- _selective_clone(): Replay filtered subset of parent commits into child tract
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tract.exceptions import SpawnError
from tract.models.content import DialogueContent, InstructionContent
from tract.models.session import CollapseResult, SpawnInfo
from tract.prompts.summarize import DEFAULT_COLLAPSE_SYSTEM, build_collapse_prompt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tract.engine.commit import CommitEngine
    from tract.protocols import CompiledContext, TokenCounter
    from tract.storage.schema import CommitRow
    from tract.storage.sqlite import SqliteSpawnPointerRepository
    from tract.tract import Tract


def spawn_tract(
    session_factory,
    engine,
    spawn_repo: SqliteSpawnPointerRepository,
    parent_tract,
    *,
    purpose: str,
    inheritance: str = "head_snapshot",
    display_name: str | None = None,
    max_tokens: int | None = None,
    filter_func: Callable | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    include_types: list[str] | None = None,
    include_instructions: bool = True,
    inherit_tools: bool = False,
    context_budget: int | None = None,
) -> Tract:
    """Create a child tract linked to parent via spawn pointer.

    Args:
        session_factory: SQLAlchemy sessionmaker for creating new sessions.
        engine: SQLAlchemy engine (shared across session).
        spawn_repo: Repository for spawn pointer storage.
        parent_tract: The parent Tract instance.
        purpose: Description of the child's task.
        inheritance: Inheritance mode: "head_snapshot" (default), "full_clone",
            or "selective".
        display_name: Optional human-readable name for the child.
        max_tokens: Max tokens for head_snapshot (truncates from oldest).
        filter_func: For selective mode: callable ``(commit_row) -> bool``
            that receives each parent commit and returns True to include it.
        include_tags: For selective mode: include commits that have at least
            one of these tags (immutable tags from ``tags_json``).
        exclude_tags: For selective mode: exclude commits that have any of
            these tags.
        include_types: For selective mode: include only commits whose
            ``content_type`` is in this list.
        include_instructions: For selective mode: always include instruction
            and config commits even when filtered out. Default True.
        inherit_tools: If True, copy parent's active tool definitions to
            the child tract after creation. Default False.
        context_budget: If set, limits the total tokens inherited by the
            child. For head_snapshot this is wired to ``max_tokens``. For
            selective mode, commits exceeding the budget are dropped
            (oldest non-instruction first).

    Returns:
        The new child Tract instance.

    Raises:
        SpawnError: If inheritance mode is invalid.
        ValueError: If selective mode is used without any filter criteria.
    """
    if inheritance not in ("head_snapshot", "full_clone", "selective"):
        raise SpawnError(f"Unknown inheritance mode: {inheritance}")

    if inheritance == "selective":
        # Build auto-filter from convenience parameters if no filter_func
        if filter_func is None and (include_tags or exclude_tags or include_types):
            def _auto_filter(commit_row: CommitRow) -> bool:
                tags = set(commit_row.tags_json or [])
                if include_tags and not any(t in tags for t in include_tags):
                    return False
                if exclude_tags and any(t in tags for t in exclude_tags):
                    return False
                if include_types and commit_row.content_type not in include_types:
                    return False
                return True
            filter_func = _auto_filter

        if filter_func is None:
            raise ValueError(
                "selective inheritance requires a filter_func, include_tags, "
                "exclude_tags, or include_types parameter"
            )

    # Import here to avoid circular imports
    from tract.engine.commit import CommitEngine
    from tract.engine.compiler import DefaultContextCompiler
    from tract.engine.tokens import TiktokenCounter
    from tract.models.config import TractConfig
    from tract.storage.sqlite import (
        SqliteAnnotationRepository,
        SqliteBlobRepository,
        SqliteCommitParentRepository,
        SqliteCommitRepository,
        SqliteOperationEventRepository,
        SqliteRefRepository,
    )
    from tract.tract import Tract

    # Generate child tract_id
    child_tract_id = uuid.uuid4().hex

    # Capture parent state BEFORE creating spawn commit
    # (so inheritance doesn't include the spawn commit itself)
    parent_head = parent_tract.head
    parent_compiled = parent_tract.compile() if parent_head else None
    parent_commits_snapshot = None
    if inheritance in ("full_clone", "selective"):
        parent_commits_snapshot = list(
            parent_tract._commit_repo.get_all(parent_tract.tract_id)
        )

    # Create spawn commit in parent BEFORE saving the spawn pointer.
    # If the parent commit fails, we avoid leaving an orphaned pointer.
    parent_tract.commit(
        DialogueContent(
            role="system",
            text=f"Spawned subagent for: {purpose}",
        ),
        message=f"spawn: {purpose}",
    )

    # Now persist the spawn pointer (parent commit succeeded)
    now = datetime.now(timezone.utc)
    spawn_repo.save(
        parent_tract_id=parent_tract.tract_id,
        parent_commit_hash=parent_head,
        child_tract_id=child_tract_id,
        purpose=purpose,
        inheritance_mode=inheritance,
        display_name=display_name,
        created_at=now,
    )
    # Commit the spawn pointer session to release the write lock
    spawn_repo._session.commit()

    # Build child Tract with shared engine
    child_session = session_factory()
    child_config = TractConfig()

    child_commit_repo = SqliteCommitRepository(child_session)
    child_blob_repo = SqliteBlobRepository(child_session)
    child_ref_repo = SqliteRefRepository(child_session)
    child_annotation_repo = SqliteAnnotationRepository(child_session)
    child_parent_repo = SqliteCommitParentRepository(child_session)
    child_event_repo = SqliteOperationEventRepository(child_session)

    child_token_counter = TiktokenCounter(
        encoding_name=child_config.tokenizer_encoding,
    )

    child_commit_engine = CommitEngine(
        commit_repo=child_commit_repo,
        blob_repo=child_blob_repo,
        ref_repo=child_ref_repo,
        annotation_repo=child_annotation_repo,
        token_counter=child_token_counter,
        tract_id=child_tract_id,
        token_budget=child_config.token_budget,
        parent_repo=child_parent_repo,
    )

    child_compiler = DefaultContextCompiler(
        commit_repo=child_commit_repo,
        blob_repo=child_blob_repo,
        annotation_repo=child_annotation_repo,
        token_counter=child_token_counter,
        parent_repo=child_parent_repo,
    )

    # Wire context_budget to max_tokens for head_snapshot
    effective_max_tokens = max_tokens
    if context_budget is not None and inheritance == "head_snapshot":
        # context_budget overrides max_tokens if both are set
        effective_max_tokens = context_budget

    # Apply inheritance using pre-captured parent state
    if inheritance == "head_snapshot":
        _head_snapshot(
            parent_compiled,
            child_commit_engine,
            max_tokens=effective_max_tokens,
            token_counter=child_token_counter,
        )
        child_session.commit()
    elif inheritance == "full_clone":
        _full_clone(
            parent_tract,
            child_commit_engine,
            child_commit_repo,
            child_blob_repo,
            child_ref_repo,
            child_annotation_repo,
            commits_snapshot=parent_commits_snapshot,
        )
        child_session.commit()
    elif inheritance == "selective":
        _selective_clone(
            parent_tract,
            child_commit_engine,
            child_commit_repo,
            child_blob_repo,
            child_ref_repo,
            child_annotation_repo,
            commits_snapshot=parent_commits_snapshot,
            filter_func=filter_func,  # type: ignore[arg-type]  # Optional narrowed by caller
            include_instructions=include_instructions,
            context_budget=context_budget,
            token_counter=child_token_counter,
        )
        child_session.commit()

    # Build child Tract instance
    child = Tract(
        engine=None,  # Engine is owned by Session, not individual Tracts
        session=child_session,
        commit_engine=child_commit_engine,
        compiler=child_compiler,
        tract_id=child_tract_id,
        config=child_config,
        commit_repo=child_commit_repo,
        blob_repo=child_blob_repo,
        ref_repo=child_ref_repo,
        annotation_repo=child_annotation_repo,
        token_counter=child_token_counter,
        parent_repo=child_parent_repo,
        event_repo=child_event_repo,
    )
    child._spawn_repo = spawn_repo

    # Inherit tools from parent if requested
    if inherit_tools:
        parent_tools = parent_tract.get_tools()
        if parent_tools is not None:
            child.set_tools(parent_tools)

    return child


def _head_snapshot(
    parent_compiled: CompiledContext | None,
    child_commit_engine: CommitEngine,
    *,
    max_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
) -> str | None:
    """Compile parent context and seed child with it.

    Args:
        parent_compiled: Pre-captured CompiledContext from parent, or None if empty.
        child_commit_engine: CommitEngine for the child tract.
        max_tokens: If set, truncate oldest messages to fit budget.
        token_counter: Token counter for measuring truncation.

    Returns:
        Child's HEAD hash after seeding, or None if parent is empty.
    """
    if parent_compiled is None or not parent_compiled.messages:
        return None

    # Format messages into text blob
    lines = []
    for msg in parent_compiled.messages:
        lines.append(f"[{msg.role}]: {msg.content}")
    text = "\n".join(lines)

    # Truncate if needed
    if max_tokens is not None and token_counter is not None:
        current_tokens = token_counter.count_text(text)
        if current_tokens > max_tokens:
            # Binary search for the minimum number of lines to drop from oldest
            lo, hi = 1, len(lines) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                candidate = "\n".join(lines[mid:])
                if token_counter.count_text(candidate) <= max_tokens:
                    hi = mid
                else:
                    lo = mid + 1
            text = "\n".join(lines[lo:])

    # Commit as InstructionContent in child
    info = child_commit_engine.create_commit(
        content=InstructionContent(text=text),
        message="Inherited context from parent (head_snapshot)",
    )
    return info.commit_hash


def _full_clone(
    parent_tract,
    child_commit_engine: CommitEngine,
    child_commit_repo,
    child_blob_repo,
    child_ref_repo,
    child_annotation_repo,
    *,
    commits_snapshot: list | None = None,
) -> str | None:
    """Replay all parent commits into child tract.

    Args:
        parent_tract: The parent Tract to clone.
        child_commit_engine: CommitEngine for the child tract.
        child_commit_repo: Child's commit repository.
        child_blob_repo: Child's blob repository.
        child_ref_repo: Child's ref repository.
        child_annotation_repo: Child's annotation repository.
        commits_snapshot: Pre-captured list of parent CommitRows to clone.
            If None, reads from parent_tract (may include spawn commit).

    Returns:
        Child's HEAD hash after cloning, or None if parent is empty.
    """
    from tract.models.content import validate_content

    # Use pre-captured snapshot or read from parent (fallback)
    all_commits = commits_snapshot if commits_snapshot is not None else list(
        parent_tract._commit_repo.get_all(parent_tract.tract_id)
    )
    if not all_commits:
        return None

    # Note: edit_target relationships are not preserved during clone.
    # Cloned commits get new hashes, so old edit_target values become invalid.
    # A hash remapping could be added here if edit_target fidelity is needed.
    last_hash = None
    skipped = 0
    for commit_row in all_commits:
        # Read blob content
        blob = parent_tract._blob_repo.get(commit_row.content_hash)
        if blob is None:
            logger.warning(
                "Skipping commit %s: blob %s not found",
                commit_row.commit_hash,
                commit_row.content_hash,
            )
            skipped += 1
            continue

        # Parse content from blob
        try:
            content_dict = json.loads(blob.payload_json)
            content = validate_content(content_dict)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "Skipping commit %s: failed to parse/validate blob %s: %s",
                commit_row.commit_hash,
                commit_row.content_hash,
                exc,
            )
            skipped += 1
            continue

        # Create commit in child (new hashes, new timestamps)
        info = child_commit_engine.create_commit(
            content=content,
            operation=commit_row.operation,
            message=commit_row.message,
            metadata=commit_row.metadata_json,
            generation_config=commit_row.generation_config_json,
        )
        last_hash = info.commit_hash

        # Copy annotations
        annotations = parent_tract._annotation_repo.get_history(
            commit_row.commit_hash
        )
        for ann in annotations:
            child_commit_engine.annotate(
                info.commit_hash, ann.priority, ann.reason
            )

    if skipped:
        logger.warning("full_clone: skipped %d of %d commits", skipped, len(all_commits))

    return last_hash


def _selective_clone(
    parent_tract,
    child_commit_engine: CommitEngine,
    child_commit_repo,
    child_blob_repo,
    child_ref_repo,
    child_annotation_repo,
    *,
    commits_snapshot: list | None = None,
    filter_func: Callable,
    include_instructions: bool = True,
    context_budget: int | None = None,
    token_counter: TokenCounter | None = None,
) -> str | None:
    """Replay a filtered subset of parent commits into child tract.

    Walks the parent commit chain (oldest first) and applies the filter_func
    to each commit. Commits that pass the filter are replayed into the child.

    Special handling:
    - instruction and config commits are always included when
      ``include_instructions=True`` (default), even if the filter rejects them.
    - EDIT commits whose edit_target was filtered out are skipped (the target
      does not exist in the child, so the edit would be dangling).
    - When ``context_budget`` is set and the total tokens of included commits
      exceed the budget, oldest non-instruction commits are dropped first.

    Args:
        parent_tract: The parent Tract to selectively clone.
        child_commit_engine: CommitEngine for the child tract.
        child_commit_repo: Child's commit repository.
        child_blob_repo: Child's blob repository.
        child_ref_repo: Child's ref repository.
        child_annotation_repo: Child's annotation repository.
        commits_snapshot: Pre-captured list of parent CommitRows (oldest first).
        filter_func: Callable ``(commit_row) -> bool`` that returns True to include.
        include_instructions: If True, instruction and config commits bypass
            the filter and are always included.
        context_budget: If set, limits total inherited tokens. Oldest
            non-instruction commits are dropped first to fit.
        token_counter: Token counter for budget enforcement.

    Returns:
        Child's HEAD hash after selective cloning, or None if nothing was included.
    """
    from tract.models.commit import CommitOperation
    from tract.models.content import validate_content

    # Use pre-captured snapshot or read from parent (fallback)
    all_commits = commits_snapshot if commits_snapshot is not None else list(
        parent_tract._commit_repo.get_all(parent_tract.tract_id)
    )
    if not all_commits:
        return None

    # Types that are always included when include_instructions is True
    _always_include_types = {"instruction", "config"}

    # First pass: determine which commits are included
    included_hashes: set[str] = set()
    for commit_row in all_commits:
        is_always = (
            include_instructions
            and commit_row.content_type in _always_include_types
        )
        if is_always or filter_func(commit_row):
            included_hashes.add(commit_row.commit_hash)

    # Second pass: remove EDIT commits whose edit_target is not included
    final_included: set[str] = set()
    for commit_row in all_commits:
        if commit_row.commit_hash not in included_hashes:
            continue
        if (
            commit_row.operation == CommitOperation.EDIT
            and commit_row.edit_target
            and commit_row.edit_target not in included_hashes
        ):
            # Edit target was filtered out — skip this edit
            continue
        final_included.add(commit_row.commit_hash)

    if not final_included:
        return None

    # Budget enforcement: if context_budget is set, drop oldest
    # non-instruction commits until total tokens fit within budget.
    if context_budget is not None:
        # Compute total tokens for included commits
        total_tokens = sum(
            row.token_count
            for row in all_commits
            if row.commit_hash in final_included
        )
        if total_tokens > context_budget:
            # Separate instruction/config commits from droppable ones
            # Walk oldest-first so we drop oldest first
            droppable = [
                row for row in all_commits
                if row.commit_hash in final_included
                and row.content_type not in _always_include_types
            ]
            for row in droppable:
                if total_tokens <= context_budget:
                    break
                final_included.discard(row.commit_hash)
                total_tokens -= row.token_count

    if not final_included:
        return None

    # Third pass: replay included commits in order (oldest first)
    last_hash = None
    skipped = 0
    for commit_row in all_commits:
        if commit_row.commit_hash not in final_included:
            continue

        # Read blob content
        blob = parent_tract._blob_repo.get(commit_row.content_hash)
        if blob is None:
            logger.warning(
                "Skipping commit %s: blob %s not found",
                commit_row.commit_hash,
                commit_row.content_hash,
            )
            skipped += 1
            continue

        # Parse content from blob
        try:
            content_dict = json.loads(blob.payload_json)
            content = validate_content(content_dict)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "Skipping commit %s: failed to parse/validate blob %s: %s",
                commit_row.commit_hash,
                commit_row.content_hash,
                exc,
            )
            skipped += 1
            continue

        # Create commit in child (new hashes, new timestamps)
        info = child_commit_engine.create_commit(
            content=content,
            operation=commit_row.operation,
            message=commit_row.message,
            metadata=commit_row.metadata_json,
            generation_config=commit_row.generation_config_json,
        )
        last_hash = info.commit_hash

        # Copy annotations
        annotations = parent_tract._annotation_repo.get_history(
            commit_row.commit_hash
        )
        for ann in annotations:
            child_commit_engine.annotate(
                info.commit_hash, ann.priority, ann.reason
            )

    if skipped:
        logger.warning("selective_clone: skipped %d commits", skipped)

    return last_hash


def collapse_tract(
    parent_tract,
    child_tract,
    spawn_repo: SqliteSpawnPointerRepository,
    *,
    content: str | None = None,
    instructions: str | None = None,
    auto_commit: bool | None = None,
    target_tokens: int | None = None,
    llm_client=None,
) -> CollapseResult:
    """Compress child tract history into a summary commit in parent.

    Supports three modes:
    - Manual (content provided): Use as-is for summary
    - Collaborative (no content, LLM available, auto_commit=False/None): LLM drafts, caller reviews
    - Autonomous (no content, LLM available, auto_commit=True): LLM drafts, auto-commits

    Args:
        parent_tract: The parent Tract to receive the summary.
        child_tract: The child Tract to summarize.
        spawn_repo: Repository for spawn pointer lookups.
        content: Manual summary text (bypasses LLM).
        instructions: Additional LLM instructions.
        auto_commit: Whether to auto-commit. None means False.
        target_tokens: Target token count for LLM summary.
        llm_client: LLM client for generating summaries.

    Returns:
        CollapseResult with summary details.

    Raises:
        SpawnError: If no spawn pointer found or LLM required but not available.
    """
    # Look up spawn pointer
    pointer = spawn_repo.get_by_child(child_tract.tract_id)
    if pointer is None:
        raise SpawnError(
            f"No spawn pointer found for child tract: {child_tract.tract_id}"
        )

    purpose = pointer.purpose

    # Compile child's full context
    compiled = child_tract.compile()
    source_tokens = compiled.token_count

    # Format child messages for summarization
    lines = []
    for msg in compiled.messages:
        lines.append(f"[{msg.role}]: {msg.content}")
    messages_text = "\n".join(lines)

    # Determine summary text
    if content is not None:
        # Manual mode
        summary_text = content
    else:
        # Collaborative or autonomous: need LLM
        if llm_client is None:
            raise SpawnError(
                "Cannot collapse without content or LLM client. "
                "Provide content= for manual mode, or configure an LLM client."
            )
        # Build prompt
        user_prompt = build_collapse_prompt(
            messages_text,
            purpose,
            target_tokens=target_tokens,
            instructions=instructions,
        )
        # Call LLM
        try:
            response = llm_client.chat(
                messages=[
                    {"role": "system", "content": DEFAULT_COLLAPSE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if hasattr(llm_client, "extract_content"):
                summary_text = llm_client.extract_content(response)
            else:
                summary_text = response["choices"][0]["message"]["content"] or ""
        except Exception as e:
            # Wrap any LLM/network error into SpawnError for uniform handling.
            logger.warning("LLM collapse call failed: %s", e, exc_info=True)
            raise SpawnError(f"LLM call failed during collapse: {e}") from e

    # Count summary tokens
    summary_tokens = parent_tract._token_counter.count_text(summary_text)

    # Commit to parent if auto_commit
    commit_hash = None
    effective_auto_commit = auto_commit if auto_commit is not None else False
    if effective_auto_commit:
        child_head = child_tract.head
        info = parent_tract.commit(
            DialogueContent(role="assistant", text=summary_text),
            message=f"collapse: {purpose}",
            metadata={
                "collapse_source_tract_id": child_tract.tract_id,
                "collapse_source_head": child_head,
            },
        )
        commit_hash = info.commit_hash

    return CollapseResult(
        parent_commit_hash=commit_hash,
        child_tract_id=child_tract.tract_id,
        summary_text=summary_text,
        summary_tokens=summary_tokens,
        source_tokens=source_tokens,
        purpose=purpose,
    )


def _row_to_spawn_info(row) -> SpawnInfo:
    """Convert a SpawnPointerRow to a SpawnInfo dataclass."""
    return SpawnInfo(
        spawn_id=row.id,
        parent_tract_id=row.parent_tract_id,
        parent_commit_hash=row.parent_commit_hash,
        child_tract_id=row.child_tract_id,
        purpose=row.purpose,
        inheritance_mode=row.inheritance_mode,
        display_name=row.display_name,
        created_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


async def acollapse_tract(
    parent_tract,
    child_tract,
    spawn_repo,
    *,
    content: str | None = None,
    instructions: str | None = None,
    auto_commit: bool | None = None,
    target_tokens: int | None = None,
    llm_client=None,
) -> "CollapseResult":
    """Async version of :func:`collapse_tract`.

    The LLM call is awaited; everything else (compile, commit) remains sync.
    """
    from tract.llm.protocols import acall_llm

    # Look up spawn pointer
    pointer = spawn_repo.get_by_child(child_tract.tract_id)
    if pointer is None:
        raise SpawnError(
            f"No spawn pointer found for child tract: {child_tract.tract_id}"
        )

    purpose = pointer.purpose

    # Compile child's full context (sync -- local operation)
    compiled = child_tract.compile()
    source_tokens = compiled.token_count

    # Format child messages for summarization
    lines = []
    for msg in compiled.messages:
        lines.append(f"[{msg.role}]: {msg.content}")
    messages_text = "\n".join(lines)

    # Determine summary text
    if content is not None:
        summary_text = content
    else:
        if llm_client is None:
            raise SpawnError(
                "Cannot collapse without content or LLM client. "
                "Provide content= for manual mode, or configure an LLM client."
            )
        user_prompt = build_collapse_prompt(
            messages_text,
            purpose,
            target_tokens=target_tokens,
            instructions=instructions,
        )
        try:
            response = await acall_llm(
                llm_client,
                messages=[
                    {"role": "system", "content": DEFAULT_COLLAPSE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if hasattr(llm_client, "extract_content"):
                summary_text = llm_client.extract_content(response)
            else:
                summary_text = response["choices"][0]["message"]["content"] or ""
        except Exception as e:
            # Wrap any LLM/network error into SpawnError for uniform handling.
            logger.warning("Async LLM collapse call failed: %s", e, exc_info=True)
            raise SpawnError(f"LLM call failed during collapse: {e}") from e

    # Count summary tokens
    summary_tokens = parent_tract._token_counter.count_text(summary_text)

    # Commit to parent if auto_commit (sync -- local operation)
    commit_hash = None
    effective_auto_commit = auto_commit if auto_commit is not None else False
    if effective_auto_commit:
        child_head = child_tract.head
        info = parent_tract.commit(
            DialogueContent(role="assistant", text=summary_text),
            message=f"collapse: {purpose}",
            metadata={
                "collapse_source_tract_id": child_tract.tract_id,
                "collapse_source_head": child_head,
            },
        )
        commit_hash = info.commit_hash

    return CollapseResult(
        parent_commit_hash=commit_hash,
        child_tract_id=child_tract.tract_id,
        summary_text=summary_text,
        summary_tokens=summary_tokens,
        source_tokens=source_tokens,
        purpose=purpose,
    )
