"""Spawn and collapse operations for multi-agent coordination.

Provides:
- spawn_tract(): Create a child tract linked to a parent via spawn pointer
- collapse_tract(): Compress child tract history into a summary commit in parent
- _head_snapshot(): Compile parent context and seed child with it
- _full_clone(): Replay all parent commits into child tract
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tract.exceptions import SpawnError
from tract.models.content import DialogueContent, InstructionContent
from tract.models.session import CollapseResult, SpawnInfo
from tract.prompts.summarize import DEFAULT_COLLAPSE_SYSTEM, build_collapse_prompt

if TYPE_CHECKING:
    from tract.engine.commit import CommitEngine
    from tract.protocols import CompiledContext, TokenCounter
    from tract.tract import Tract
    from tract.storage.sqlite import (
        SqliteAnnotationRepository,
        SqliteBlobRepository,
        SqliteCommitRepository,
        SqliteRefRepository,
        SqliteSpawnPointerRepository,
    )


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
) -> Tract:
    """Create a child tract linked to parent via spawn pointer.

    Args:
        session_factory: SQLAlchemy sessionmaker for creating new sessions.
        engine: SQLAlchemy engine (shared across session).
        spawn_repo: Repository for spawn pointer storage.
        parent_tract: The parent Tract instance.
        purpose: Description of the child's task.
        inheritance: Inheritance mode: "head_snapshot" (default), "full_clone",
            or "selective" (not implemented).
        display_name: Optional human-readable name for the child.
        max_tokens: Max tokens for head_snapshot (truncates from oldest).

    Returns:
        The new child Tract instance.

    Raises:
        SpawnError: If inheritance mode is invalid.
        NotImplementedError: If inheritance is "selective".
    """
    if inheritance == "selective":
        raise NotImplementedError(
            "Selective inheritance is not yet implemented (deferred to future version)"
        )

    if inheritance not in ("head_snapshot", "full_clone"):
        raise SpawnError(f"Unknown inheritance mode: {inheritance}")

    # Import here to avoid circular imports
    from tract.engine.cache import CacheManager
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
        SqliteSpawnPointerRepository as _SpawnRepo,
    )
    from tract.tract import Tract

    # Generate child tract_id
    child_tract_id = uuid.uuid4().hex

    # Capture parent state BEFORE creating spawn commit
    # (so inheritance doesn't include the spawn commit itself)
    parent_head = parent_tract.head
    parent_compiled = parent_tract.compile() if parent_head else None
    parent_commits_snapshot = None
    if inheritance == "full_clone":
        parent_commits_snapshot = list(
            parent_tract._commit_repo.get_all(parent_tract.tract_id)
        )

    # Create spawn pointer
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
    # before the parent tract's session tries to write
    spawn_repo._session.commit()

    # Create spawn commit in parent
    parent_tract.commit(
        DialogueContent(
            role="system",
            text=f"Spawned subagent for: {purpose}",
        ),
        message=f"spawn: {purpose}",
    )

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

    # Apply inheritance using pre-captured parent state
    if inheritance == "head_snapshot":
        _head_snapshot(
            parent_compiled,
            child_commit_engine,
            max_tokens=max_tokens,
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

    # Note: response_to relationships are not preserved during clone.
    # Cloned commits get new hashes, so old response_to values become invalid.
    # A hash remapping could be added here if response_to fidelity is needed.
    last_hash = None
    for commit_row in all_commits:
        # Read blob content
        blob = parent_tract._blob_repo.get(commit_row.content_hash)
        if blob is None:
            continue

        # Parse content from blob
        content_dict = json.loads(blob.payload_json)
        content = validate_content(content_dict)

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
            summary_text = response.choices[0].message.content
        except Exception as e:
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
