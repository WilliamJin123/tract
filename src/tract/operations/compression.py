"""Compression operations for Trace.

Implements context compression: summarizing commit chains into shorter
summaries to fit token budgets. compress_range() always returns a
PendingCompress -- the Tract layer handles routing (auto-approve,
hook, or return to caller).

Content modes:
- LLM (llm_client provided): uses LLM for summarization
- Manual (content provided): uses user text as summary

PINNED commits survive compression verbatim. SKIP commits are ignored.
Original commits remain in DB as unreachable (non-destructive).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tract.exceptions import CompressionError
from tract.models.annotations import Priority, RetentionCriteria
from tract.models.commit import CommitOperation
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult, GCResult, ReorderWarning
from tract.models.content import DialogueContent
from tract.prompts.summarize import DEFAULT_SUMMARIZE_SYSTEM, build_summarize_prompt

logger = logging.getLogger(__name__)


def _validate_retention(
    summary: str,
    criteria: list[RetentionCriteria],
) -> tuple[bool, str | None]:
    """Check summary against deterministic retention criteria.

    Iterates over all ``RetentionCriteria`` objects and checks that every
    ``match_patterns`` entry is present in the summary (as substring or
    regex depending on ``match_mode``).

    Args:
        summary: The LLM-generated summary text.
        criteria: List of RetentionCriteria from IMPORTANT commits.

    Returns:
        ``(True, None)`` if all patterns are satisfied, otherwise
        ``(False, "Summary missing: ...")`` with details.
    """
    failures: list[str] = []
    for c in criteria:
        if c.match_patterns:
            for pattern in c.match_patterns:
                if c.match_mode == "regex":
                    if not re.search(pattern, summary):
                        failures.append(f"regex not found: {pattern}")
                else:
                    if pattern not in summary:
                        failures.append(f"substring not found: {pattern}")
    if failures:
        return (False, "Summary missing: " + "; ".join(failures))
    return (True, None)


if TYPE_CHECKING:
    from pydantic import BaseModel

    from tract.engine.commit import CommitEngine
    from tract.llm.protocols import LLMClient
    from tract.protocols import TokenCounter
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
        OperationEventRepository,
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
) -> tuple[list[CommitRow], list[CommitRow], list[CommitRow], list[CommitRow]]:
    """Classify commits by their priority annotation.

    Args:
        range_commits: Commits to classify.
        annotation_repo: For looking up priorities.
        preserve: Additional hashes to treat as PINNED for this invocation.

    Returns:
        (pinned_commits, important_commits, normal_commits, skip_commits)
    """
    preserve_set = set(preserve) if preserve else set()
    target_hashes = [r.commit_hash for r in range_commits]
    latest_annotations = annotation_repo.batch_get_latest(target_hashes)

    pinned: list[CommitRow] = []
    important: list[CommitRow] = []
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
            elif annotation.priority == Priority.IMPORTANT:
                important.append(row)
            elif annotation.priority == Priority.SKIP:
                skip.append(row)
            else:
                normal.append(row)
        else:
            normal.append(row)

    return pinned, important, normal, skip


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


def build_role_label(role: str, meta: dict[str, object]) -> str:
    """Build a role label enriched with tool metadata for summarization.

    Used by :func:`_build_messages_text` and by
    :meth:`Tract.compress_tool_calls` to give the LLM summarizer
    structured context about tool interactions.

    Examples::

        >>> build_role_label("assistant", {})
        '[assistant]'
        >>> build_role_label("assistant", {"tool_calls": [{"name": "grep", "arguments": {"pattern": "error"}}]})
        '[assistant (calls: grep(pattern=error))]'
        >>> build_role_label("tool", {"tool_call_id": "call_abc", "name": "grep"})
        '[tool:grep (call_id=call_abc)]'
    """
    # Assistant message that requested tool calls
    tool_calls = meta.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        args_parts = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_name = tc.get("name", "?")
            tc_args = tc.get("arguments", {})
            if isinstance(tc_args, dict) and tc_args:
                brief = ", ".join(
                    f"{k}={_truncate_value(v)}" for k, v in tc_args.items()
                )
                args_parts.append(f"{tc_name}({brief})")
            else:
                args_parts.append(tc_name)
        if args_parts:
            return f"[{role} (calls: {', '.join(args_parts)})]"

    # Tool result message
    tool_call_id = meta.get("tool_call_id")
    tool_name = meta.get("name")
    if tool_call_id or tool_name:
        tag = f"tool:{tool_name}" if tool_name else "tool"
        if tool_call_id:
            return f"[{tag} (call_id={tool_call_id})]"
        return f"[{tag}]"

    return f"[{role}]"


def _truncate_value(value: object, max_len: int = 60) -> str:
    """Truncate a value repr for display in role labels."""
    s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _build_messages_text(
    group: list[CommitRow],
    blob_repo: BlobRepository,
) -> str:
    """Build text representation of a group of commits for LLM summarization.

    Includes tool metadata (tool names, arguments, call IDs) when available
    so the summarizer can produce structured, tool-aware summaries.

    Args:
        group: List of CommitRow in chain order.
        blob_repo: For loading blob content.

    Returns:
        Formatted text with role labels and tool metadata annotations.

    Raises:
        CompressionError: If all blobs in the group are unavailable.
    """
    parts: list[str] = []
    unavailable_count = 0

    for row in group:
        blob = blob_repo.get(row.content_hash)
        if blob is None:
            logger.warning("Blob not found for content_hash=%s", row.content_hash)
            parts.append("[content unavailable]")
            unavailable_count += 1
            continue

        try:
            data = json.loads(blob.payload_json)
        except (json.JSONDecodeError, TypeError):
            parts.append("[content unavailable]")
            unavailable_count += 1
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

        # Enrich label with tool metadata when available
        meta = row.metadata_json or {}
        label = build_role_label(role, meta)

        parts.append(f"{label}: {text}")

    if unavailable_count > 0:
        logger.warning(
            "%d of %d commits had unavailable content", unavailable_count, len(group)
        )
    if unavailable_count == len(group):
        raise CompressionError(
            f"All {len(group)} commits in group have unavailable content"
        )

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
    llm_client: LLMClient,
    token_counter: TokenCounter,
    *,
    target_tokens: int | None = None,
    instructions: str | None = None,
    system_prompt: str | None = None,
    llm_kwargs: dict | None = None,
    retention_instructions: list[str] | None = None,
) -> str:
    """Summarize a group of messages using the LLM client.

    Args:
        messages_text: Text to summarize.
        llm_client: LLM client implementing the LLMClient protocol.
        token_counter: For token counting.
        target_tokens: Optional target token count.
        instructions: Extra guidance appended to the **user message** of the
            summarization LLM call. The default prompt is preserved; this
            text is added as "Additional instructions: ..." at the end.
        system_prompt: Completely replaces the **system message** of the
            summarization LLM call (DEFAULT_SUMMARIZE_SYSTEM). Controls the
            LLM's persona and behavioral guidelines. When None, the built-in
            system prompt is used.
        retention_instructions: Optional list of retention instructions from
            IMPORTANT commits. Injected into the prompt so the LLM preserves
            the specified details.

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
        retention_instructions=retention_instructions,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    response = llm_client.chat(messages, **(llm_kwargs or {}))

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
    event_repo: OperationEventRepository,
    parent_repo: CommitParentRepository,
    *,
    commits: list[str] | None = None,
    from_commit: str | None = None,
    to_commit: str | None = None,
    target_tokens: int | None = None,
    preserve: list[str] | None = None,
    llm_client: LLMClient | None = None,
    content: str | None = None,
    instructions: str | None = None,
    system_prompt: str | None = None,
    llm_kwargs: dict | None = None,
    generation_config: dict | None = None,
    type_registry: dict[str, type] | None = None,
    validator: object | None = None,
    max_retries: int = 3,
    token_tolerance: int | None = None,
    triggered_by: str | None = None,
    two_stage: bool = False,
) -> PendingCompress:
    """Core compression operation.

    Compresses a range of commits into summaries, preserving PINNED commits
    and ignoring SKIP commits. Always returns a PendingCompress -- the caller
    (Tract.compress) decides whether to auto-execute, fire a hook, or return
    it to the user for review.

    Supports two content modes:
    - Manual (content provided): uses user text as summary
    - LLM (llm_client provided): uses LLM for summarization

    Args:
        tract_id: Tract identifier.
        commit_repo: Commit repository.
        blob_repo: Blob repository.
        annotation_repo: Annotation repository.
        ref_repo: Ref repository.
        commit_engine: Commit engine for creating commits.
        token_counter: Token counter.
        event_repo: Operation event repository for provenance tracking.
        parent_repo: Commit parent repository.
        commits: Optional explicit list of commit hashes.
        from_commit: Optional range start (inclusive).
        to_commit: Optional range end (inclusive).
        target_tokens: Optional target token count for summaries.
        preserve: Optional list of hashes to treat as PINNED.
        llm_client: Optional LLM client for summarization.
        content: Optional manual summary text (bypasses LLM).
        instructions: Optional LLM instructions.
        system_prompt: Optional custom system prompt for LLM.
        llm_kwargs: Optional per-operation LLM config (model, temperature, etc.).
        generation_config: Optional generation config to record on summary commits.
        type_registry: Optional custom content type registry.
        token_tolerance: Additive token tolerance for summary validation.
            When set, summaries up to target_tokens + token_tolerance are
            accepted. Defaults to 500 when None. Use 0 for strict mode.
        triggered_by: Optional provenance string (e.g. "trigger:auto_compress").

    Returns:
        PendingCompress with all state needed for finalization.

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

    # c. Classify by priority (4 groups: pinned, important, normal, skip)
    pinned_commits, important_commits, normal_commits, skip_commits = (
        _classify_by_priority(range_commits, annotation_repo, preserve=preserve)
    )

    # IMPORTANT commits are compressed alongside NORMAL commits --
    # merge them into a single "compressible" list for token counting
    # while tracking which ones are IMPORTANT for retention enrichment.
    compressible_commits = normal_commits + important_commits

    # d. Nothing to compress?
    if not compressible_commits:
        raise CompressionError(
            "Nothing to compress -- all commits are pinned or skipped"
        )

    # e. Partition around pinned (IMPORTANT treated as compressible, not boundary)
    pinned_hashes = {r.commit_hash for r in pinned_commits}
    skip_hashes = {r.commit_hash for r in skip_commits}
    groups = _partition_around_pinned(range_commits, pinned_hashes, skip_hashes)

    # e2. Gather retention criteria from IMPORTANT commits
    important_hashes = {r.commit_hash for r in important_commits}
    important_annotations = annotation_repo.batch_get_latest(
        list(important_hashes)
    ) if important_hashes else {}
    # Build per-group retention criteria and instructions
    group_retention: list[list[RetentionCriteria]] = []
    group_retention_instructions: list[list[str]] = []
    for group in groups:
        criteria: list[RetentionCriteria] = []
        ret_instructions: list[str] = []
        for row in group:
            if row.commit_hash in important_hashes:
                ann = important_annotations.get(row.commit_hash)
                if ann is not None and ann.retention_json:
                    rc = RetentionCriteria(**ann.retention_json)
                    criteria.append(rc)
                    if rc.instructions:
                        ret_instructions.append(rc.instructions)
        group_retention.append(criteria)
        group_retention_instructions.append(ret_instructions)

    # f0. Two-stage guidance generation (if enabled)
    guidance_text: str | None = None
    if two_stage and llm_client is not None:
        from tract.prompts.guidance import (
            COMPRESS_GUIDANCE_SYSTEM,
            build_compress_guidance_prompt,
        )

        # Build combined messages text for all groups
        all_text = "\n\n".join(
            _build_messages_text(group, blob_repo) for group in groups
        )
        guidance_response = llm_client.chat(
            [
                {"role": "system", "content": COMPRESS_GUIDANCE_SYSTEM},
                {"role": "user", "content": build_compress_guidance_prompt(
                    all_text, instructions=instructions
                )},
            ],
            **(llm_kwargs or {}),
        )
        guidance_text = guidance_response["choices"][0]["message"]["content"]
        # Prepend guidance to instructions for all subsequent summarize calls
        instructions = f"Guidance:\n{guidance_text}\n\n{instructions or ''}"
    elif two_stage and llm_client is None and content is None:
        raise CompressionError(
            "two_stage=True requires an LLM client. "
            "Call configure_llm() or pass api_key to Tract.open()."
        )

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
        for gidx, group in enumerate(groups):
            text = _build_messages_text(group, blob_repo)
            g_retention = group_retention[gidx]
            g_ret_instructions = group_retention_instructions[gidx]

            # Check if this group has deterministic retention criteria
            has_deterministic = any(
                rc.match_patterns for rc in g_retention
            )

            # Build a combined validator: user-supplied + retention + token count
            def _make_combined_validator(
                user_validator, retention_criteria,
                target_tok, tok_counter, tok_tolerance,
            ):
                """Build a validator that checks user + retention + token count."""
                def _combined(result: str) -> tuple[bool, str | None]:
                    # Check retention criteria first
                    ok, diag = _validate_retention(result, retention_criteria)
                    if not ok:
                        return (False, diag)
                    # Then check user-supplied validator
                    if user_validator is not None:
                        uok, udiag = user_validator(result)
                        if not uok:
                            return (False, udiag)
                    # Then check token count if target_tokens is set
                    if target_tok is not None:
                        actual = tok_counter.count_text(result)
                        tol = tok_tolerance if tok_tolerance is not None else 500
                        limit = target_tok + tol
                        if actual > limit:
                            return (
                                False,
                                f"Summary is ~{actual} tokens "
                                f"(target: {target_tok}). "
                                f"Condense to ~{target_tok} tokens.",
                            )
                    return (True, None)
                return _combined

            needs_retry = (
                validator is not None
                or has_deterministic
                or target_tokens is not None
            )

            if needs_retry:
                from tract.retry import retry_with_steering

                combined_validator = _make_combined_validator(
                    validator, g_retention,
                    target_tokens, token_counter, token_tolerance,
                )

                # Mutable instructions for steering
                current_instructions = instructions

                def _attempt_summarize(
                    _text=text,
                    _ret_inst=g_ret_instructions,
                ) -> str:
                    return _summarize_group(
                        _text, llm_client, token_counter,
                        target_tokens=target_tokens,
                        instructions=current_instructions,
                        system_prompt=system_prompt,
                        llm_kwargs=llm_kwargs,
                        retention_instructions=_ret_inst or None,
                    )

                def _validate_summary(result: str) -> tuple[bool, str | None]:
                    return combined_validator(result)

                def _steer_summary(diagnosis: str) -> None:
                    nonlocal current_instructions
                    base = current_instructions or ""
                    current_instructions = (
                        f"{base}\n\nPrevious summary was rejected: {diagnosis}"
                    ).strip()

                retry_result = retry_with_steering(
                    attempt=_attempt_summarize,
                    validate=_validate_summary,
                    steer=_steer_summary,
                    head_fn=lambda: "n/a",
                    reset_fn=lambda _h: None,
                    max_retries=max_retries,
                )
                summaries.append(retry_result.value)
            else:
                summary = _summarize_group(
                    text, llm_client, token_counter,
                    target_tokens=target_tokens,
                    instructions=instructions,
                    system_prompt=system_prompt,
                    llm_kwargs=llm_kwargs,
                    retention_instructions=g_ret_instructions or None,
                )
                summaries.append(summary)
    else:
        raise CompressionError(
            "No LLM client configured and no manual content provided. "
            "Call configure_llm() first or pass content='...'."
        )

    # g. Calculate token counts (both normal + important are "source" commits)
    original_tokens = sum(c.token_count for c in compressible_commits)
    estimated_tokens = sum(token_counter.count_text(s) for s in summaries)

    # h. Always build and return PendingCompress -- the caller (Tract.compress)
    #    handles three-tier routing: review=True returns to caller, hook fires
    #    handler, no hook auto-approves.
    pending = PendingCompress(
        operation="compress",
        tract=None,  # type: ignore[arg-type]  # Set by Tract.compress() before routing
        summaries=summaries,
        source_commits=[c.commit_hash for c in compressible_commits],
        preserved_commits=[c.commit_hash for c in pinned_commits],
        original_tokens=original_tokens,
        estimated_tokens=estimated_tokens,
        triggered_by=triggered_by,
    )
    # Store context needed for later commit by _finalize_compression
    pending._range_commits = range_commits
    pending._pinned_commits = pinned_commits
    pending._normal_commits = compressible_commits
    pending._pinned_hashes = pinned_hashes
    pending._skip_hashes = skip_hashes
    pending._groups = groups
    pending._branch_name = branch_name
    pending._target_tokens = target_tokens
    pending._instructions = instructions
    pending._system_prompt = system_prompt
    pending._head_hash = head_hash
    pending._generation_config = generation_config
    pending._two_stage = two_stage
    if two_stage and guidance_text is not None:
        pending.guidance = guidance_text
        pending.guidance_source = "llm"
    return pending


def _commit_compression(
    *,
    tract_id: str,
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    ref_repo: RefRepository,
    commit_engine: CommitEngine,
    token_counter: TokenCounter,
    event_repo: OperationEventRepository,
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
    system_prompt: str | None,
    branch_name: str | None,
    type_registry: dict[str, type] | None = None,
    expected_head: str | None = None,
    generation_config: dict | None = None,
) -> CompressResult:
    """Create summary commits, interleave with PINNED, record provenance.

    This is the core commit logic shared between autonomous and collaborative modes.

    Args:
        See compress_range() for parameter descriptions.
        expected_head: If set, verify HEAD matches before proceeding (TOCTOU guard).

    Returns:
        CompressResult with all fields populated.

    Raises:
        CompressionError: If HEAD changed since compression was planned.
    """
    # TOCTOU guard: verify HEAD hasn't changed since compression was planned
    if expected_head is not None:
        current_head = ref_repo.get_head(tract_id)
        if current_head != expected_head:
            raise CompressionError(
                f"HEAD changed since compression was planned "
                f"(expected {expected_head[:8]}, got {(current_head or 'None')[:8]}). "
                f"Re-run compress() to plan against the current state."
            )

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

                # Create summary commit.
                # Summaries use role="assistant" by design: the LLM generates
                # the summary text, so it's semantically assistant-authored
                # regardless of the roles in the compressed commits.
                n_commits = len(groups[gidx])
                summary_content = DialogueContent(
                    role="assistant",
                    text=summary_text,
                )
                info = commit_engine.create_commit(
                    content=summary_content,
                    message=f"Compressed {n_commits} commits",
                    generation_config=generation_config,
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

    # f. Save OperationEvent
    event_repo.save_event(
        event_id=compression_id,
        tract_id=tract_id,
        event_type="compress",
        branch_name=branch_name,
        created_at=datetime.now(timezone.utc),
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        params_json={
            "target_tokens": target_tokens,
            "instructions": instructions,
            "system_prompt": system_prompt,
        },
    )

    # Record sources (normal commits that were compressed)
    for pos, nc in enumerate(normal_commits):
        event_repo.add_commit(compression_id, nc.commit_hash, "source", pos)

    # Record results (summary commits produced)
    for pos, sc_hash in enumerate(summary_commit_hashes):
        event_repo.add_commit(compression_id, sc_hash, "result", pos)

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
    - "response_chain_break": A commit references an edit_target commit that is
      not present in the reordered set at all.

    Args:
        order: Ordered list of commit hashes (the proposed reorder).
        commit_repo: Commit repository for hash lookups.
        blob_repo: Blob repository (reserved for future use).

    Returns:
        List of ReorderWarning objects (may be empty if no issues found).
    """
    warnings: list[ReorderWarning] = []
    order_set = set(order)
    hash_to_pos = {h: i for i, h in enumerate(order)}

    for h in order:
        row = commit_repo.get(h)
        if row is None:
            continue

        # Check for EDIT before target
        if row.operation == CommitOperation.EDIT and row.edit_target:
            if row.edit_target in hash_to_pos:
                # Target is in the order list -- check position
                if hash_to_pos[h] < hash_to_pos[row.edit_target]:
                    warnings.append(
                        ReorderWarning(
                            warning_type="edit_before_target",
                            commit_hash=row.commit_hash,
                            description=(
                                f"EDIT commit {row.commit_hash[:8]} appears before "
                                f"its target {row.edit_target[:8]}"
                            ),
                            severity="structural",
                        )
                    )

        # Check for edit_target chain break
        if row.edit_target and row.edit_target not in order_set:
            warnings.append(
                ReorderWarning(
                    warning_type="response_chain_break",
                    commit_hash=row.commit_hash,
                    description=(
                        f"Commit {row.commit_hash[:8]} references "
                        f"{row.edit_target[:8]} which is not in the reordered set"
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

    Handles the edge case where compression starts at the root commit.
    Deletes the branch ref so get_head() returns None (symbolic HEAD
    follows the branch ref, so a missing branch ref means no HEAD).
    """
    if branch_name is not None:
        ref_repo.delete_ref(tract_id, f"refs/heads/{branch_name}")


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


def plan_gc(
    tract_id: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository,
    event_repo: OperationEventRepository,
    *,
    orphan_retention_days: int = 7,
    archive_retention_days: int | None = None,
    branch: str | None = None,
) -> tuple[list[CommitRow], int]:
    """Plan phase of garbage collection -- determine what to remove.

    Finds all commits not reachable from any branch tip (or a specific
    branch), then identifies eligible ones based on age and archive status.

    Args:
        tract_id: Tract identifier.
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        parent_repo: Commit parent repository.
        event_repo: Operation event repository for archive status checks.
        orphan_retention_days: Days before orphans become eligible for removal.
        archive_retention_days: If set, days before archived commits become
            eligible for removal. None means archives are never removed.
        branch: If set, only this branch's reachability is considered.

    Returns:
        Tuple of (commits_to_remove, tokens_to_free).
    """
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
    commits_to_remove: list[CommitRow] = []
    tokens_to_free = 0

    for commit in unreachable:
        is_archive = event_repo.is_source_of(commit.commit_hash)
        created = _normalize_dt(commit.created_at)
        age_days = (now - created).total_seconds() / 86400

        if is_archive:
            if archive_retention_days is not None and age_days >= archive_retention_days:
                commits_to_remove.append(commit)
                tokens_to_free += commit.token_count
        else:
            if age_days >= orphan_retention_days:
                commits_to_remove.append(commit)
                tokens_to_free += commit.token_count

    return commits_to_remove, tokens_to_free


def execute_gc(
    tract_id: str,
    commits_to_remove: list[CommitRow],
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    event_repo: OperationEventRepository,
) -> GCResult:
    """Execute phase of garbage collection -- actually delete commits.

    Args:
        tract_id: Tract identifier.
        commits_to_remove: List of CommitRow objects to delete.
        commit_repo: Commit repository.
        blob_repo: Blob repository.
        event_repo: Operation event repository.

    Returns:
        GCResult with removal counts and duration.
    """
    import time

    start = time.monotonic()

    blobs_removed = 0
    tokens_freed = 0
    source_commits_removed = 0

    for commit in commits_to_remove:
        content_hash = commit.content_hash
        tokens_freed += commit.token_count

        # Track archive status before deleting provenance
        if event_repo.is_source_of(commit.commit_hash):
            source_commits_removed += 1

        # Clean up operation event provenance
        event_repo.delete_commit(commit.commit_hash)

        # Delete the commit
        commit_repo.delete(commit.commit_hash)

        # Try to delete the blob if no other commit references it
        if blob_repo.delete_if_orphaned(content_hash):
            blobs_removed += 1

    # Clean up orphaned OperationEvent records (no sources AND no results left)
    all_event_ids = event_repo.get_all_ids(tract_id)
    for eid in all_event_ids:
        if not event_repo.get_commits(eid, "source") and not event_repo.get_commits(eid, "result"):
            event_repo.delete_event(eid)

    duration = time.monotonic() - start

    return GCResult(
        commits_removed=len(commits_to_remove),
        blobs_removed=blobs_removed,
        tokens_freed=tokens_freed,
        source_commits_removed=source_commits_removed,
        duration_seconds=duration,
    )


def gc(
    tract_id: str,
    commit_repo: CommitRepository,
    ref_repo: RefRepository,
    parent_repo: CommitParentRepository,
    blob_repo: BlobRepository,
    event_repo: OperationEventRepository,
    *,
    orphan_retention_days: int = 7,
    archive_retention_days: int | None = None,
    branch: str | None = None,
) -> GCResult:
    """Garbage-collect unreachable commits with configurable retention.

    Finds all commits not reachable from any branch tip (or a specific
    branch if ``branch`` is set), then removes eligible ones based on
    age and archive status.

    Retention triggers:
    - Orphans (not archived): removed if older than ``orphan_retention_days``
    - Archives (compression sources): preserved by default. Removed only
      if ``archive_retention_days`` is set and the commit is old enough.

    Args:
        tract_id: Tract identifier.
        commit_repo: Commit repository.
        ref_repo: Ref repository.
        parent_repo: Commit parent repository.
        blob_repo: Blob repository.
        event_repo: Operation event repository for provenance tracking.
        orphan_retention_days: Days before orphans become eligible for removal.
        archive_retention_days: If set, days before archived commits become
            eligible for removal. None means archives are never removed.
        branch: If set, only this branch's reachability is considered.

    Returns:
        GCResult with removal counts and duration.

    Note:
        Loads all commits for the tract into memory. This is acceptable for
        conversation-scale data (hundreds to low thousands of commits) but
        may need optimization for very large tracts.
    """
    commits_to_remove, _ = plan_gc(
        tract_id, commit_repo, ref_repo, parent_repo, event_repo,
        orphan_retention_days=orphan_retention_days,
        archive_retention_days=archive_retention_days,
        branch=branch,
    )
    return execute_gc(
        tract_id, commits_to_remove, commit_repo, blob_repo, event_repo,
    )
