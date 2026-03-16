"""Autonomous tract operations: auto-split, auto-rebase, auto-branch.

Provides LLM-driven functions for autonomous context management:

* **auto_split** -- LLM splits a large commit into granular pieces.
* **auto_rebase** -- LLM decides whether to rebase and onto which branch.
* **auto_branch** -- LLM decides whether to create a new branch and names it.

All follow the fail-open pattern from gate.py/maintain.py: on LLM errors,
operations return safe defaults (no action taken).

Example::

    from tract.autonomous import auto_split, auto_rebase, auto_branch

    result = auto_split(t, commit_hash)
    print(f"Split into {result.split_count} commits")

    rebase_result = auto_rebase(t)
    print(f"Rebased: {rebase_result.rebased}")

    branch_result = auto_branch(t, context="Starting auth implementation")
    print(f"Branched: {branch_result.branched}, name: {branch_result.branch_name}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tract._helpers import async_safe_llm_call as _async_safe_llm_call
from tract._helpers import safe_llm_call as _safe_llm_call
from tract._helpers import strip_fences as _strip_fences

if TYPE_CHECKING:
    from tract.tract import Tract

__all__: list[str] = [
    "AutoSplitResult",
    "AutoRebaseResult",
    "AutoBranchResult",
    "auto_split",
    "aauto_split",
    "auto_rebase",
    "aauto_rebase",
    "auto_branch",
    "aauto_branch",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoSplitResult:
    """Result of an LLM-driven commit split."""

    original_hash: str
    new_hashes: tuple[str, ...]
    split_count: int
    tokens_used: int
    reasoning: str


@dataclass(frozen=True)
class AutoRebaseResult:
    """Result of an LLM-driven rebase decision."""

    rebased: bool
    reason: str
    target_branch: str | None
    tokens_used: int


@dataclass(frozen=True)
class AutoBranchResult:
    """Result of an LLM-driven branch decision."""

    branched: bool
    branch_name: str | None
    reason: str
    tokens_used: int


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SPLIT_SYSTEM_PROMPT = """\
You are a context management agent. Your job is to split a single large commit \
into multiple smaller, logically coherent pieces.

You will receive the content of a commit. Split it into granular, self-contained pieces.

Respond with JSON:
{
  "reasoning": "Brief explanation of how you split the content",
  "pieces": [
    {"content": "First piece of content", "message": "Description of first piece"},
    {"content": "Second piece of content", "message": "Description of second piece"}
  ]
}

If the content is already small and coherent (cannot be meaningfully split), return:
{
  "reasoning": "Content is already atomic",
  "pieces": []
}
"""

_REBASE_SYSTEM_PROMPT = """\
You are a context management agent. Your job is to decide whether the current \
branch should be rebased onto another branch.

You will receive information about the current branch and available branches.

Respond with JSON:
{
  "reasoning": "Brief explanation of your decision",
  "should_rebase": true,
  "target_branch": "branch-name"
}

Or if no rebase is needed:
{
  "reasoning": "Brief explanation of why no rebase is needed",
  "should_rebase": false,
  "target_branch": null
}

Consider: divergence from the main branch, whether the current branch would \
benefit from upstream changes, and branch relationships.
"""

_BRANCH_SYSTEM_PROMPT = """\
You are a context management agent. Your job is to decide whether a new branch \
should be created for the current task.

You will receive the current branch state, existing branches, recent commits, \
and a task/context description.

Respond with JSON:
{
  "reasoning": "Brief explanation of your decision",
  "should_branch": true,
  "branch_name": "feature/descriptive-name"
}

Or if no new branch is needed:
{
  "reasoning": "Brief explanation of why no branch is needed",
  "should_branch": false,
  "branch_name": null
}

Branch names must follow git naming rules (no spaces, no special characters). \
Use descriptive, kebab-case names.
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_client(tract: Tract, operation: str = "autonomous") -> Any | None:
    """Resolve the LLM client, trying autonomous > intelligence > chat.

    Returns None if no client available (fail-open).
    """
    for op in (operation, "intelligence", "chat"):
        try:
            return tract._resolve_llm_client(op)
        except RuntimeError:
            continue
    return None


def _build_llm_kwargs(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build LLM kwargs dict from optional parameters."""
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    else:
        kwargs["temperature"] = 0.2
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


# ---------------------------------------------------------------------------
# auto_split
# ---------------------------------------------------------------------------

def auto_split(
    tract: Tract,
    commit_hash: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoSplitResult:
    """Split a commit into smaller, logically coherent pieces using LLM judgment.

    Gets the commit content, asks an LLM to split it into pieces, then
    creates new APPEND commits for each piece and EDITs the original to SKIP it.

    Fail-open: on LLM error, returns original hash unchanged.

    Args:
        tract: The Tract instance.
        commit_hash: Hash of the commit to split.
        model: Model override for the LLM call.
        temperature: Temperature override.
        max_tokens: Max tokens override.

    Returns:
        :class:`AutoSplitResult` with the new commit hashes.
    """
    # Fail-open default
    fail_open = AutoSplitResult(
        original_hash=commit_hash,
        new_hashes=(commit_hash,),
        split_count=1,
        tokens_used=0,
        reasoning="No split performed (fail-open).",
    )

    # Resolve client
    client = _resolve_client(tract)
    if client is None:
        return AutoSplitResult(
            original_hash=commit_hash,
            new_hashes=(commit_hash,),
            split_count=1,
            tokens_used=0,
            reasoning="No LLM client configured; no split performed (fail-open).",
        )

    # Get the commit content
    try:
        content = tract.get_content(commit_hash)
        if content is None:
            return fail_open
        content_str = json.dumps(content, default=str) if isinstance(content, dict) else str(content)
    except Exception:
        logger.warning("Failed to get content for commit %s; no split.", commit_hash[:12], exc_info=True)
        return fail_open

    # Build LLM messages
    messages = [
        {"role": "system", "content": _SPLIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"=== COMMIT CONTENT ===\n{content_str}"},
    ]

    llm_kwargs = _build_llm_kwargs(model, temperature, max_tokens)

    # LLM call
    result = _safe_llm_call(client, messages, llm_kwargs)
    if result is None:
        return fail_open

    raw_text, tokens_used = result

    # Parse response
    pieces = _parse_split_response(raw_text)
    if not pieces:
        return AutoSplitResult(
            original_hash=commit_hash,
            new_hashes=(commit_hash,),
            split_count=1,
            tokens_used=tokens_used,
            reasoning="LLM returned no split pieces; keeping original.",
        )

    # Create new commits and skip original
    return _execute_split(tract, commit_hash, pieces, tokens_used)


async def aauto_split(
    tract: Tract,
    commit_hash: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoSplitResult:
    """Async version of :func:`auto_split`."""
    fail_open = AutoSplitResult(
        original_hash=commit_hash,
        new_hashes=(commit_hash,),
        split_count=1,
        tokens_used=0,
        reasoning="No split performed (fail-open).",
    )

    client = _resolve_client(tract)
    if client is None:
        return AutoSplitResult(
            original_hash=commit_hash,
            new_hashes=(commit_hash,),
            split_count=1,
            tokens_used=0,
            reasoning="No LLM client configured; no split performed (fail-open).",
        )

    try:
        content = tract.get_content(commit_hash)
        if content is None:
            return fail_open
        content_str = json.dumps(content, default=str) if isinstance(content, dict) else str(content)
    except Exception:
        return fail_open

    messages = [
        {"role": "system", "content": _SPLIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"=== COMMIT CONTENT ===\n{content_str}"},
    ]

    llm_kwargs = _build_llm_kwargs(model, temperature, max_tokens)

    result = await _async_safe_llm_call(client, messages, llm_kwargs)
    if result is None:
        return fail_open

    raw_text, tokens_used = result
    pieces = _parse_split_response(raw_text)
    if not pieces:
        return AutoSplitResult(
            original_hash=commit_hash,
            new_hashes=(commit_hash,),
            split_count=1,
            tokens_used=tokens_used,
            reasoning="LLM returned no split pieces; keeping original.",
        )

    return _execute_split(tract, commit_hash, pieces, tokens_used)


def _parse_split_response(text: str) -> list[dict[str, str]]:
    """Parse an LLM split response into a list of {content, message} dicts."""
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            pieces = data.get("pieces", [])
            if not isinstance(pieces, list):
                return []
            valid = []
            for piece in pieces:
                if isinstance(piece, dict) and "content" in piece:
                    valid.append({
                        "content": str(piece["content"]),
                        "message": str(piece.get("message", "Split piece")),
                    })
            return valid
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def _execute_split(
    tract: Tract,
    original_hash: str,
    pieces: list[dict[str, str]],
    tokens_used: int,
) -> AutoSplitResult:
    """Create new commits for each split piece and SKIP the original."""
    from tract.models.annotations import Priority
    from tract.models.commit import CommitOperation

    new_hashes: list[str] = []
    reasoning_parts: list[str] = []

    for piece in pieces:
        try:
            info = tract.commit(
                {"content_type": "freeform", "payload": {"text": piece["content"]}},
                operation=CommitOperation.APPEND,
                message=piece["message"],
            )
            new_hashes.append(info.commit_hash)
            reasoning_parts.append(f"Created: {info.commit_hash[:8]} - {piece['message']}")
        except Exception as exc:
            logger.warning(
                "Failed to create split commit: %s", exc, exc_info=True,
            )

    if not new_hashes:
        # All pieces failed -- fail-open, keep original
        return AutoSplitResult(
            original_hash=original_hash,
            new_hashes=(original_hash,),
            split_count=1,
            tokens_used=tokens_used,
            reasoning="All split commit creations failed; keeping original.",
        )

    # SKIP the original commit
    try:
        tract.annotate(original_hash, Priority.SKIP, reason="Split into smaller commits")
    except Exception:
        logger.warning(
            "Failed to SKIP original commit %s after split.", original_hash[:12], exc_info=True,
        )

    reasoning = f"Split into {len(new_hashes)} pieces. " + "; ".join(reasoning_parts)

    return AutoSplitResult(
        original_hash=original_hash,
        new_hashes=tuple(new_hashes),
        split_count=len(new_hashes),
        tokens_used=tokens_used,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# auto_rebase
# ---------------------------------------------------------------------------

def _build_rebase_manifest(tract: Tract) -> str:
    """Build a manifest for rebase decisions."""
    current = tract.current_branch or "(detached)"
    head = tract.head
    head_short = head[:8] if head else "(empty)"
    branches = tract.list_branches()

    lines = [
        "=== BRANCH STATE ===",
        f"Current branch: {current} | HEAD: {head_short}",
        "",
        "BRANCHES:",
    ]
    for b in branches:
        marker = " *" if b.is_current else ""
        lines.append(f"  {b.name}{marker} -> {b.commit_hash[:8] if b.commit_hash else '(empty)'}")

    # Recent commits on current branch
    entries = tract.log(limit=10)
    if entries:
        lines.append("")
        lines.append("RECENT COMMITS (current branch):")
        for e in entries:
            lines.append(f"  [{e.commit_hash[:8]}] {e.content_type} | \"{e.message or '(no msg)'}\"")

    return "\n".join(lines)


def auto_rebase(
    tract: Tract,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoRebaseResult:
    """Decide whether to rebase the current branch using LLM judgment.

    Builds a manifest of branch state and asks the LLM whether a rebase
    would be beneficial. If yes, executes the rebase.

    Fail-open: on error, returns rebased=False.

    Args:
        tract: The Tract instance.
        model: Model override for the LLM call.
        temperature: Temperature override.
        max_tokens: Max tokens override.

    Returns:
        :class:`AutoRebaseResult`.
    """
    fail_open = AutoRebaseResult(
        rebased=False,
        reason="No rebase performed (fail-open).",
        target_branch=None,
        tokens_used=0,
    )

    client = _resolve_client(tract)
    if client is None:
        return AutoRebaseResult(
            rebased=False,
            reason="No LLM client configured; no rebase performed (fail-open).",
            target_branch=None,
            tokens_used=0,
        )

    # Build manifest
    manifest = _build_rebase_manifest(tract)

    messages = [
        {"role": "system", "content": _REBASE_SYSTEM_PROMPT},
        {"role": "user", "content": manifest},
    ]

    llm_kwargs = _build_llm_kwargs(model, temperature, max_tokens)

    result = _safe_llm_call(client, messages, llm_kwargs)
    if result is None:
        return fail_open

    raw_text, tokens_used = result

    # Parse response
    decision = _parse_rebase_response(raw_text)
    if decision is None:
        return AutoRebaseResult(
            rebased=False,
            reason="Could not parse LLM response; no rebase performed.",
            target_branch=None,
            tokens_used=tokens_used,
        )

    should_rebase, target_branch, reasoning = decision

    if not should_rebase or not target_branch:
        return AutoRebaseResult(
            rebased=False,
            reason=reasoning,
            target_branch=None,
            tokens_used=tokens_used,
        )

    # Execute rebase
    try:
        tract.rebase(target_branch)
        return AutoRebaseResult(
            rebased=True,
            reason=reasoning,
            target_branch=target_branch,
            tokens_used=tokens_used,
        )
    except Exception as exc:
        logger.warning(
            "Auto-rebase onto '%s' failed: %s", target_branch, exc, exc_info=True,
        )
        return AutoRebaseResult(
            rebased=False,
            reason=f"Rebase failed: {exc}",
            target_branch=target_branch,
            tokens_used=tokens_used,
        )


async def aauto_rebase(
    tract: Tract,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoRebaseResult:
    """Async version of :func:`auto_rebase`."""
    fail_open = AutoRebaseResult(
        rebased=False,
        reason="No rebase performed (fail-open).",
        target_branch=None,
        tokens_used=0,
    )

    client = _resolve_client(tract)
    if client is None:
        return AutoRebaseResult(
            rebased=False,
            reason="No LLM client configured; no rebase performed (fail-open).",
            target_branch=None,
            tokens_used=0,
        )

    manifest = _build_rebase_manifest(tract)

    messages = [
        {"role": "system", "content": _REBASE_SYSTEM_PROMPT},
        {"role": "user", "content": manifest},
    ]

    llm_kwargs = _build_llm_kwargs(model, temperature, max_tokens)

    result = await _async_safe_llm_call(client, messages, llm_kwargs)
    if result is None:
        return fail_open

    raw_text, tokens_used = result
    decision = _parse_rebase_response(raw_text)
    if decision is None:
        return AutoRebaseResult(
            rebased=False,
            reason="Could not parse LLM response; no rebase performed.",
            target_branch=None,
            tokens_used=tokens_used,
        )

    should_rebase, target_branch, reasoning = decision
    if not should_rebase or not target_branch:
        return AutoRebaseResult(
            rebased=False,
            reason=reasoning,
            target_branch=None,
            tokens_used=tokens_used,
        )

    try:
        tract.rebase(target_branch)
        return AutoRebaseResult(
            rebased=True,
            reason=reasoning,
            target_branch=target_branch,
            tokens_used=tokens_used,
        )
    except Exception as exc:
        return AutoRebaseResult(
            rebased=False,
            reason=f"Rebase failed: {exc}",
            target_branch=target_branch,
            tokens_used=tokens_used,
        )


def _parse_rebase_response(text: str) -> tuple[bool, str | None, str] | None:
    """Parse an LLM rebase response. Returns (should_rebase, target, reasoning) or None."""
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            reasoning = str(data.get("reasoning") or "").strip() or "(no reasoning given)"
            should_rebase = bool(data.get("should_rebase", False))
            target = data.get("target_branch")
            if target is not None:
                target = str(target).strip() or None
            return should_rebase, target, reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# auto_branch
# ---------------------------------------------------------------------------

def _build_branch_manifest(tract: Tract, context: str = "") -> str:
    """Build a manifest for branch decisions."""
    current = tract.current_branch or "(detached)"
    head = tract.head
    head_short = head[:8] if head else "(empty)"
    branches = tract.list_branches()

    lines = [
        "=== BRANCH STATE ===",
        f"Current branch: {current} | HEAD: {head_short}",
        "",
        "EXISTING BRANCHES:",
    ]
    for b in branches:
        marker = " *" if b.is_current else ""
        lines.append(f"  {b.name}{marker}")

    # Recent commits
    entries = tract.log(limit=10)
    if entries:
        lines.append("")
        lines.append("RECENT COMMITS:")
        for e in entries:
            lines.append(f"  [{e.commit_hash[:8]}] {e.content_type} | \"{e.message or '(no msg)'}\"")

    # Active directives
    try:
        ci = tract.config_index
        if ci.directives:
            lines.append("")
            lines.append("ACTIVE DIRECTIVES:")
            for name, text in ci.directives.items():
                lines.append(f"  {name}: {text[:80]}...")
    except Exception:
        pass

    if context:
        lines.append("")
        lines.append("=== TASK CONTEXT ===")
        lines.append(context)

    return "\n".join(lines)


def auto_branch(
    tract: Tract,
    *,
    context: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoBranchResult:
    """Decide whether to create a new branch using LLM judgment.

    Builds a manifest of current state and asks the LLM whether a new
    branch should be created. If yes, creates and switches to it.

    Fail-open: on error, returns branched=False.

    Args:
        tract: The Tract instance.
        context: Optional task/context description to inform the decision.
        model: Model override for the LLM call.
        temperature: Temperature override.
        max_tokens: Max tokens override.

    Returns:
        :class:`AutoBranchResult`.
    """
    fail_open = AutoBranchResult(
        branched=False,
        branch_name=None,
        reason="No branch created (fail-open).",
        tokens_used=0,
    )

    client = _resolve_client(tract)
    if client is None:
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason="No LLM client configured; no branch created (fail-open).",
            tokens_used=0,
        )

    manifest = _build_branch_manifest(tract, context)

    messages = [
        {"role": "system", "content": _BRANCH_SYSTEM_PROMPT},
        {"role": "user", "content": manifest},
    ]

    llm_kwargs = _build_llm_kwargs(model, temperature, max_tokens)

    result = _safe_llm_call(client, messages, llm_kwargs)
    if result is None:
        return fail_open

    raw_text, tokens_used = result
    decision = _parse_branch_response(raw_text)
    if decision is None:
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason="Could not parse LLM response; no branch created.",
            tokens_used=tokens_used,
        )

    should_branch, branch_name, reasoning = decision

    if not should_branch or not branch_name:
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason=reasoning,
            tokens_used=tokens_used,
        )

    # Create and switch to branch
    try:
        tract.branch(branch_name)
        return AutoBranchResult(
            branched=True,
            branch_name=branch_name,
            reason=reasoning,
            tokens_used=tokens_used,
        )
    except Exception as exc:
        logger.warning(
            "Auto-branch '%s' failed: %s", branch_name, exc, exc_info=True,
        )
        return AutoBranchResult(
            branched=False,
            branch_name=branch_name,
            reason=f"Branch creation failed: {exc}",
            tokens_used=tokens_used,
        )


async def aauto_branch(
    tract: Tract,
    *,
    context: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoBranchResult:
    """Async version of :func:`auto_branch`."""
    fail_open = AutoBranchResult(
        branched=False,
        branch_name=None,
        reason="No branch created (fail-open).",
        tokens_used=0,
    )

    client = _resolve_client(tract)
    if client is None:
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason="No LLM client configured; no branch created (fail-open).",
            tokens_used=0,
        )

    manifest = _build_branch_manifest(tract, context)

    messages = [
        {"role": "system", "content": _BRANCH_SYSTEM_PROMPT},
        {"role": "user", "content": manifest},
    ]

    llm_kwargs = _build_llm_kwargs(model, temperature, max_tokens)

    result = await _async_safe_llm_call(client, messages, llm_kwargs)
    if result is None:
        return fail_open

    raw_text, tokens_used = result
    decision = _parse_branch_response(raw_text)
    if decision is None:
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason="Could not parse LLM response; no branch created.",
            tokens_used=tokens_used,
        )

    should_branch, branch_name, reasoning = decision
    if not should_branch or not branch_name:
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason=reasoning,
            tokens_used=tokens_used,
        )

    try:
        tract.branch(branch_name)
        return AutoBranchResult(
            branched=True,
            branch_name=branch_name,
            reason=reasoning,
            tokens_used=tokens_used,
        )
    except Exception as exc:
        return AutoBranchResult(
            branched=False,
            branch_name=branch_name,
            reason=f"Branch creation failed: {exc}",
            tokens_used=tokens_used,
        )


def _parse_branch_response(text: str) -> tuple[bool, str | None, str] | None:
    """Parse an LLM branch response. Returns (should_branch, name, reasoning) or None."""
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            reasoning = str(data.get("reasoning") or "").strip() or "(no reasoning given)"
            should_branch = bool(data.get("should_branch", False))
            name = data.get("branch_name")
            if name is not None:
                name = str(name).strip() or None
            return should_branch, name, reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None
