"""LLM-driven context intelligence: cherry-picking and deduplication.

Provides pure functions (not classes) for intelligent commit selection
and duplicate detection. Both follow the same patterns as SemanticGate
and SemanticMaintainer: manifest-based, fail-open on LLM errors.

Internally delegates to :class:`~tract.judgment.Judgment` for the LLM
call, JSON parsing, and fail-open handling.

* **cherry_pick** -- LLM selects the most relevant commits for a task/query.
* **deduplicate** -- LLM identifies groups of duplicate/overlapping commits.

Example::

    from tract.intelligence import cherry_pick, deduplicate

    result = cherry_pick(t, "Implement the auth module", limit=5)
    for h in result.selected_hashes:
        print(t.search.get_content(h))

    dedup = deduplicate(t, auto_skip=True)
    print(f"Found {len(dedup.duplicate_groups)} duplicate groups")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tract.context_view import ContextView
from tract.judgment import (
    DedupGroups,
    Judgment,
    SelectionResult,
)

if TYPE_CHECKING:
    from tract.tract import Tract

__all__: list[str] = [
    "CherryPickResult",
    "DedupResult",
    "cherry_pick",
    "acherry_pick",
    "deduplicate",
    "adeduplicate",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CherryPickResult:
    """Result of an LLM-driven cherry-pick selection."""

    selected_hashes: tuple[str, ...]
    total_candidates: int
    tokens_used: int
    reasoning: str
    consulted_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DedupResult:
    """Result of an LLM-driven deduplication analysis."""

    duplicate_groups: tuple[tuple[str, ...], ...]  # groups of duplicate commit hashes
    actions_taken: int  # number of SKIP annotations applied
    tokens_used: int
    reasoning: str
    consulted_hashes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CHERRY_PICK_SYSTEM_PROMPT = """\
You are a context relevance evaluator. Your job is to select the most relevant \
commits for a given task or query.

You will receive a manifest of commits (with content previews) and a task/query \
description.

Respond with JSON:
{
  "reasoning": "Brief explanation of why you selected these commits",
  "selected": ["<hash1>", "<hash2>", ...]
}

Select up to the specified limit. Only include commits that are clearly relevant \
to the task.
If no commits are relevant, return an empty selected list with an explanation.
Use the short hash prefixes shown in the manifest."""

_DEDUP_SYSTEM_PROMPT = """\
You are a content deduplication analyzer. Your job is to identify groups of \
commits that contain duplicate or highly overlapping content.

You will receive a manifest of commits with content previews.

Respond with JSON:
{
  "reasoning": "Brief explanation of what duplicates you found",
  "groups": [
    ["<hash1>", "<hash2>"],
    ["<hash3>", "<hash4>", "<hash5>"]
  ]
}

Each group should contain commits whose content is substantially the same or \
heavily overlapping.
A commit should appear in at most one group.
If no duplicates are found, return an empty groups list.
Use the short hash prefixes shown in the manifest."""


# ---------------------------------------------------------------------------
# Manifest builder (with content previews)
# ---------------------------------------------------------------------------

def _build_intelligence_manifest(
    tract: Tract,
    *,
    include_content_preview: bool = True,
    preview_length: int = 300,
    max_log_entries: int = 50,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a text manifest with optional content previews for intelligence tasks.

    Unlike the gate/maintain manifest, this includes content previews
    to give the LLM enough signal for relevance/dedup decisions.

    Returns:
        Tuple of (manifest_text, commit_entries) where commit_entries is
        a list of dicts with commit metadata for result resolution.
    """
    entries = tract.search.log(limit=max_log_entries)
    if not entries:
        return "=== CONTEXT MANIFEST ===\n(no commits)", []

    branch = tract.current_branch or "(detached)"
    head = tract.head
    head_short = head[:8] if head else "(empty)"

    lines: list[str] = [
        "=== CONTEXT MANIFEST ===",
        f"Branch: {branch} | HEAD: {head_short} | Commits shown: {len(entries)}",
        "",
        "COMMITS (newest first):",
    ]

    commit_entries: list[dict[str, Any]] = []

    for entry in entries:
        h = entry.commit_hash[:8]
        ctype = entry.content_type
        tokens = entry.token_count
        tags_str = ",".join(entry.tags) if entry.tags else ""
        priority = entry.effective_priority or "normal"
        msg = entry.message if entry.message else "(no message)"
        if len(msg) > 60:
            msg = msg[:57] + "..."

        lines.append(
            f"  [{h}] {ctype:<12} | {tokens:>5} tok | "
            f"tags:[{tags_str}] | {priority:<9} | \"{msg}\""
        )

        if include_content_preview:
            try:
                content = tract.search.get_content(entry.commit_hash)
                if content is not None:
                    preview = str(content)[:preview_length]
                    if len(str(content)) > preview_length:
                        preview += "..."
                    lines.append(f"    Preview: {preview}")
            except Exception:
                lines.append("    Preview: (could not retrieve)")

        commit_entries.append({
            "hash": entry.commit_hash,
            "short_hash": h,
            "content_type": ctype,
            "token_count": tokens,
            "tags": list(entry.tags) if entry.tags else [],
            "priority": priority,
        })

    return "\n".join(lines), commit_entries


# ---------------------------------------------------------------------------
# Hash resolution helpers (shared by cherry_pick and deduplicate)
# ---------------------------------------------------------------------------

def _resolve_hashes(
    raw_hashes: list[str],
    commit_entries: list[dict[str, Any]],
) -> list[str]:
    """Resolve short hash prefixes to full hashes from the commit entries."""
    prefix_to_full: dict[str, str] = {}
    for entry in commit_entries:
        prefix_to_full[entry["short_hash"]] = entry["hash"]
        prefix_to_full[entry["hash"]] = entry["hash"]

    resolved: list[str] = []
    for h in raw_hashes:
        h_str = str(h).strip()
        if h_str in prefix_to_full:
            resolved.append(prefix_to_full[h_str])
        else:
            for entry in commit_entries:
                if (entry["hash"].startswith(h_str)
                        or entry["short_hash"].startswith(h_str)):
                    resolved.append(entry["hash"])
                    break
    return resolved


# ---------------------------------------------------------------------------
# Response parsing helpers (kept for backward compatibility of test imports)
# ---------------------------------------------------------------------------

def _parse_cherry_pick_response(
    text: str, commit_entries: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    """Parse an LLM cherry-pick response into (reasoning, selected_full_hashes).

    Resolves short hash prefixes to full hashes from the commit entries.
    """
    from tract._helpers import strip_fences as _strip_fences

    cleaned = _strip_fences(text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            reasoning = (
                str(data.get("reasoning") or "").strip()
                or "(no reasoning given)"
            )
            selected = data.get("selected", [])
            if not isinstance(selected, list):
                selected = []
            resolved = _resolve_hashes(selected, commit_entries)
            return reasoning, resolved
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return f"Could not parse cherry-pick response. Raw: {text[:200]}", []


def _parse_dedup_response(
    text: str, commit_entries: list[dict[str, Any]]
) -> tuple[str, list[list[str]]]:
    """Parse an LLM dedup response into (reasoning, groups_of_full_hashes)."""
    from tract._helpers import strip_fences as _strip_fences

    cleaned = _strip_fences(text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            reasoning = (
                str(data.get("reasoning") or "").strip()
                or "(no reasoning given)"
            )
            groups = data.get("groups", [])
            if not isinstance(groups, list):
                groups = []
            resolved_groups: list[list[str]] = []
            for group in groups:
                if not isinstance(group, list):
                    continue
                resolved_group = _resolve_hashes(group, commit_entries)
                if len(resolved_group) >= 2:
                    resolved_groups.append(resolved_group)
            return reasoning, resolved_groups
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return f"Could not parse dedup response. Raw: {text[:200]}", []


# ---------------------------------------------------------------------------
# cherry_pick: sync / async
# ---------------------------------------------------------------------------

def cherry_pick(
    tract: Tract,
    query: str,
    *,
    limit: int = 10,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> CherryPickResult:
    """Select the most relevant commits for a task/query using LLM judgment.

    Builds a manifest of recent commits (with content previews) and asks
    the LLM to select the ``limit`` most relevant ones for the given query.

    Fail-open: on LLM error, returns all candidate commits (no filtering).

    Args:
        tract: The Tract instance to analyze.
        query: Natural-language task or query description.
        limit: Maximum number of commits to select.
        model: Model override for the LLM call.
        temperature: Temperature override.
        max_tokens: Max tokens override.

    Returns:
        :class:`CherryPickResult` with selected commit hashes.
    """
    return _cherry_pick_core(
        tract, query, limit=limit, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


async def acherry_pick(
    tract: Tract,
    query: str,
    *,
    limit: int = 10,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> CherryPickResult:
    """Async version of :func:`cherry_pick`."""
    return await _acherry_pick_core(
        tract, query, limit=limit, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


def _cherry_pick_core(
    tract: Tract,
    query: str,
    *,
    limit: int = 10,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> CherryPickResult:
    """Core cherry_pick implementation using Judgment."""
    # Build custom manifest (includes content previews)
    manifest, commit_entries = _build_intelligence_manifest(
        tract, include_content_preview=True, preview_length=200,
    )

    if not commit_entries:
        return CherryPickResult(
            selected_hashes=(),
            total_candidates=0,
            tokens_used=0,
            reasoning="No commits to evaluate.",
            consulted_hashes=(),
        )

    consulted_hashes = tuple(e["hash"] for e in commit_entries)

    instructions = (
        f"=== TASK/QUERY ===\n"
        f"{query}\n"
        f"\n"
        f"=== SELECTION LIMIT ===\n"
        f"Select up to {limit} most relevant commits.\n"
        f"\n"
        f"{manifest}"
    )

    prompt_override = tract.config.get_prompt("cherry_pick")

    judgment = Judgment(
        instructions=instructions,
        response_model=SelectionResult,
        system_prompt=prompt_override or _CHERRY_PICK_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.1,
        max_tokens=max_tokens,
        operation_name="intelligence",
    )

    result = judgment.evaluate(tract)

    if not result.succeeded or result.output is None:
        # Fail-open: return all commits
        all_hashes = tuple(e["hash"] for e in commit_entries)
        base = result.reasoning or "LLM call failed"
        reasoning = (
            base if "fail-open" in base.lower()
            else f"{base}; returning all commits (fail-open)."
        )
        return CherryPickResult(
            selected_hashes=all_hashes,
            total_candidates=len(commit_entries),
            tokens_used=result.tokens_used,
            reasoning=reasoning,
            consulted_hashes=consulted_hashes,
        )

    # Resolve short hashes from LLM output to full hashes
    selected = _resolve_hashes(result.output.selected, commit_entries)
    selected = selected[:limit]

    return CherryPickResult(
        selected_hashes=tuple(selected),
        total_candidates=len(commit_entries),
        tokens_used=result.tokens_used,
        reasoning=result.output.reasoning or result.reasoning,
        consulted_hashes=consulted_hashes,
    )


async def _acherry_pick_core(
    tract: Tract,
    query: str,
    *,
    limit: int = 10,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> CherryPickResult:
    """Async core cherry_pick implementation using Judgment."""
    manifest, commit_entries = _build_intelligence_manifest(
        tract, include_content_preview=True, preview_length=200,
    )

    if not commit_entries:
        return CherryPickResult(
            selected_hashes=(),
            total_candidates=0,
            tokens_used=0,
            reasoning="No commits to evaluate.",
            consulted_hashes=(),
        )

    consulted_hashes = tuple(e["hash"] for e in commit_entries)

    instructions = (
        f"=== TASK/QUERY ===\n"
        f"{query}\n"
        f"\n"
        f"=== SELECTION LIMIT ===\n"
        f"Select up to {limit} most relevant commits.\n"
        f"\n"
        f"{manifest}"
    )

    prompt_override = tract.config.get_prompt("cherry_pick")

    judgment = Judgment(
        instructions=instructions,
        response_model=SelectionResult,
        system_prompt=prompt_override or _CHERRY_PICK_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.1,
        max_tokens=max_tokens,
        operation_name="intelligence",
    )

    result = await judgment.aevaluate(tract)

    if not result.succeeded or result.output is None:
        all_hashes = tuple(e["hash"] for e in commit_entries)
        base = result.reasoning or "LLM call failed"
        reasoning = (
            base if "fail-open" in base.lower()
            else f"{base}; returning all commits (fail-open)."
        )
        return CherryPickResult(
            selected_hashes=all_hashes,
            total_candidates=len(commit_entries),
            tokens_used=result.tokens_used,
            reasoning=reasoning,
            consulted_hashes=consulted_hashes,
        )

    selected = _resolve_hashes(result.output.selected, commit_entries)
    selected = selected[:limit]

    return CherryPickResult(
        selected_hashes=tuple(selected),
        total_candidates=len(commit_entries),
        tokens_used=result.tokens_used,
        reasoning=result.output.reasoning or result.reasoning,
        consulted_hashes=consulted_hashes,
    )


# ---------------------------------------------------------------------------
# deduplicate: sync / async
# ---------------------------------------------------------------------------

def deduplicate(
    tract: Tract,
    *,
    threshold: float = 0.8,
    auto_skip: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> DedupResult:
    """Detect and optionally handle duplicate/overlapping commits using LLM \
judgment.

    Two-pass approach:
    1. Build manifest with content previews (300 chars).
    2. Ask LLM to identify groups of duplicate/overlapping commits.

    If ``auto_skip=True``, annotates all but the newest commit in each
    duplicate group as SKIP.

    Fail-open: on LLM error, returns empty groups (no action taken).

    Args:
        tract: The Tract instance to analyze.
        threshold: Similarity threshold hint for the LLM (0.0 to 1.0).
            Higher values mean stricter duplicate detection.
        auto_skip: If True, automatically mark older duplicates as SKIP.
        model: Model override for the LLM call.
        temperature: Temperature override.
        max_tokens: Max tokens override.

    Returns:
        :class:`DedupResult` with duplicate groups and actions taken.
    """
    return _deduplicate_core(
        tract, threshold=threshold, auto_skip=auto_skip,
        model=model, temperature=temperature, max_tokens=max_tokens,
    )


async def adeduplicate(
    tract: Tract,
    *,
    threshold: float = 0.8,
    auto_skip: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> DedupResult:
    """Async version of :func:`deduplicate`."""
    return await _adeduplicate_core(
        tract, threshold=threshold, auto_skip=auto_skip,
        model=model, temperature=temperature, max_tokens=max_tokens,
    )


def _build_dedup_instructions(
    manifest: str,
    threshold: float,
) -> str:
    """Build the instructions string for deduplication Judgment."""
    threshold_desc = (
        "very strict (only near-identical content)" if threshold >= 0.9
        else "strict (highly similar content)" if threshold >= 0.7
        else "moderate (substantially overlapping content)" if threshold >= 0.5
        else "loose (any meaningful overlap)"
    )
    return (
        f"=== DEDUPLICATION PARAMETERS ===\n"
        f"Similarity threshold: {threshold} ({threshold_desc})\n"
        f"\n"
        f"{manifest}"
    )


def _deduplicate_core(
    tract: Tract,
    *,
    threshold: float = 0.8,
    auto_skip: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> DedupResult:
    """Core deduplicate implementation using Judgment."""
    manifest, commit_entries = _build_intelligence_manifest(
        tract, include_content_preview=True, preview_length=300,
    )

    if not commit_entries:
        return DedupResult(
            duplicate_groups=(),
            actions_taken=0,
            tokens_used=0,
            reasoning="No commits to analyze.",
            consulted_hashes=(),
        )

    consulted_hashes = tuple(e["hash"] for e in commit_entries)
    instructions = _build_dedup_instructions(manifest, threshold)
    prompt_override = tract.config.get_prompt("dedup")

    judgment = Judgment(
        instructions=instructions,
        response_model=DedupGroups,
        system_prompt=prompt_override or _DEDUP_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.1,
        max_tokens=max_tokens,
        operation_name="intelligence",
    )

    result = judgment.evaluate(tract)
    return _finalize_dedup(
        result, commit_entries, consulted_hashes, auto_skip, tract,
    )


async def _adeduplicate_core(
    tract: Tract,
    *,
    threshold: float = 0.8,
    auto_skip: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> DedupResult:
    """Async core deduplicate implementation using Judgment."""
    manifest, commit_entries = _build_intelligence_manifest(
        tract, include_content_preview=True, preview_length=300,
    )

    if not commit_entries:
        return DedupResult(
            duplicate_groups=(),
            actions_taken=0,
            tokens_used=0,
            reasoning="No commits to analyze.",
            consulted_hashes=(),
        )

    consulted_hashes = tuple(e["hash"] for e in commit_entries)
    instructions = _build_dedup_instructions(manifest, threshold)
    prompt_override = tract.config.get_prompt("dedup")

    judgment = Judgment(
        instructions=instructions,
        response_model=DedupGroups,
        system_prompt=prompt_override or _DEDUP_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.1,
        max_tokens=max_tokens,
        operation_name="intelligence",
    )

    result = await judgment.aevaluate(tract)
    return _finalize_dedup(
        result, commit_entries, consulted_hashes, auto_skip, tract,
    )


def _finalize_dedup(
    result: Any,
    commit_entries: list[dict[str, Any]],
    consulted_hashes: tuple[str, ...],
    auto_skip: bool,
    tract: Tract,
) -> DedupResult:
    """Shared post-Judgment logic for deduplicate."""
    if not result.succeeded or result.output is None:
        base = result.reasoning or "LLM call failed"
        reasoning = (
            base if "fail-open" in base.lower()
            else f"{base}; no deduplication performed (fail-open)."
        )
        return DedupResult(
            duplicate_groups=(),
            actions_taken=0,
            tokens_used=result.tokens_used,
            reasoning=reasoning,
            consulted_hashes=consulted_hashes,
        )

    resolved_groups: list[list[str]] = []
    for group in result.output.groups:
        resolved = _resolve_hashes(group, commit_entries)
        if len(resolved) >= 2:
            resolved_groups.append(resolved)

    actions_taken = 0
    if auto_skip and resolved_groups:
        actions_taken = _apply_skip_annotations(
            tract, resolved_groups, commit_entries,
        )

    return DedupResult(
        duplicate_groups=tuple(tuple(g) for g in resolved_groups),
        actions_taken=actions_taken,
        tokens_used=result.tokens_used,
        reasoning=result.output.reasoning or result.reasoning,
        consulted_hashes=consulted_hashes,
    )


# ---------------------------------------------------------------------------
# Auto-skip helper
# ---------------------------------------------------------------------------

def _apply_skip_annotations(
    tract: Tract,
    groups: list[list[str]],
    commit_entries: list[dict[str, Any]],
) -> int:
    """Annotate older duplicates in each group as SKIP.

    Within each group, the newest commit (first in commit_entries order,
    which is newest-first) is kept; all others get SKIP.

    Returns the number of SKIP annotations applied.
    """
    from tract.models.annotations import Priority

    # Build ordering: commit_entries is newest-first, so lower index = newer
    hash_order: dict[str, int] = {}
    for idx, entry in enumerate(commit_entries):
        hash_order[entry["hash"]] = idx

    actions = 0
    for group in groups:
        if len(group) < 2:
            continue
        # Sort by order: lower index = newer
        sorted_group = sorted(
            group,
            key=lambda h: hash_order.get(h, 999999),
        )
        # Keep the newest (first), skip the rest
        for h in sorted_group[1:]:
            try:
                tract.annotations.set(
                    h,
                    Priority.SKIP,
                    reason="Duplicate content detected by deduplicate()",
                )
                actions += 1
            except Exception:
                logger.warning(
                    "Failed to annotate commit %s as SKIP during dedup.",
                    h[:8],
                    exc_info=True,
                )
    return actions
