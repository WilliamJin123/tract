"""Autonomous tract operations: auto-split, auto-rebase, auto-branch.

Provides LLM-driven functions for autonomous context management:

* **auto_split** -- LLM splits a large commit into granular pieces.
* **auto_rebase** -- LLM decides whether to rebase and onto which branch.
* **auto_branch** -- LLM decides whether to create a new branch and names it.

All follow the fail-open pattern from gate.py/maintain.py: on LLM errors,
operations return safe defaults (no action taken).

Internally delegates to :class:`~tract.judgment.Judgment` for the LLM
call, JSON parsing, and fail-open handling.

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

from tract.context_view import ContextView
from tract.judgment import (
    BooleanDecision,
    Judgment,
    SplitPlan,
)

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
    consulted_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutoRebaseResult:
    """Result of an LLM-driven rebase decision."""

    rebased: bool
    reason: str
    target_branch: str | None
    tokens_used: int
    consulted_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutoBranchResult:
    """Result of an LLM-driven branch decision."""

    branched: bool
    branch_name: str | None
    reason: str
    tokens_used: int
    consulted_hashes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SPLIT_SYSTEM_PROMPT = """\
You are a context management agent. Your job is to split a single large commit \
into multiple smaller, logically coherent pieces.

You will receive the content of a commit. Split it into granular, self-contained \
pieces.

Respond with JSON:
{
  "reasoning": "Brief explanation of how you split the content",
  "pieces": [
    {"content": "First piece of content", "message": "Description of first piece"},
    {"content": "Second piece of content", "message": "Description of second piece"}
  ]
}

If the content is already small and coherent (cannot be meaningfully split), \
return:
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
  "decision": true,
  "params": {"target_branch": "branch-name"}
}

Or if no rebase is needed:
{
  "reasoning": "Brief explanation of why no rebase is needed",
  "decision": false,
  "params": {}
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
  "decision": true,
  "params": {"branch_name": "feature/descriptive-name"}
}

Or if no new branch is needed:
{
  "reasoning": "Brief explanation of why no branch is needed",
  "decision": false,
  "params": {}
}

Branch names must follow git naming rules (no spaces, no special characters). \
Use descriptive, kebab-case names.
"""

# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

def _build_rebase_manifest(tract: Tract) -> str:
    """Build a manifest for rebase decisions."""
    current = tract.current_branch or "(detached)"
    head = tract.head
    head_short = head[:8] if head else "(empty)"
    branches = tract.branches.list()

    lines = [
        "=== BRANCH STATE ===",
        f"Current branch: {current} | HEAD: {head_short}",
        "",
        "BRANCHES:",
    ]
    for b in branches:
        marker = " *" if b.is_current else ""
        bh = b.commit_hash[:8] if b.commit_hash else "(empty)"
        lines.append(f"  {b.name}{marker} -> {bh}")

    entries = tract.search.log(limit=10)
    if entries:
        lines.append("")
        lines.append("RECENT COMMITS (current branch):")
        for e in entries:
            msg = e.message or "(no msg)"
            lines.append(
                f"  [{e.commit_hash[:8]}] {e.content_type} | \"{msg}\""
            )

    return "\n".join(lines)


def _build_branch_manifest(tract: Tract, context: str = "") -> str:
    """Build a manifest for branch decisions."""
    current = tract.current_branch or "(detached)"
    head = tract.head
    head_short = head[:8] if head else "(empty)"
    branches = tract.branches.list()

    lines = [
        "=== BRANCH STATE ===",
        f"Current branch: {current} | HEAD: {head_short}",
        "",
        "EXISTING BRANCHES:",
    ]
    for b in branches:
        marker = " *" if b.is_current else ""
        lines.append(f"  {b.name}{marker}")

    entries = tract.search.log(limit=10)
    if entries:
        lines.append("")
        lines.append("RECENT COMMITS:")
        for e in entries:
            msg = e.message or "(no msg)"
            lines.append(
                f"  [{e.commit_hash[:8]}] {e.content_type} | \"{msg}\""
            )

    try:
        directive_commits = tract.search.find(
            content_type="instruction", limit=20,
        )
        if directive_commits:
            lines.append("")
            lines.append("ACTIVE DIRECTIVES:")
            for dc in directive_commits:
                text = tract.search.get_content(dc) or ""
                label = dc.message or dc.commit_hash[:8]
                lines.append(f"  {label}: {str(text)[:80]}...")
    except Exception:
        pass

    if context:
        lines.append("")
        lines.append("=== TASK CONTEXT ===")
        lines.append(context)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing helpers (kept for backward compatibility of test imports)
# ---------------------------------------------------------------------------

def _parse_split_response(text: str) -> list[dict[str, str]]:
    """Parse an LLM split response into a list of {content, message} dicts."""
    from tract._helpers import strip_fences as _strip_fences

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


def _parse_rebase_response(
    text: str,
) -> tuple[bool, str | None, str] | None:
    """Parse an LLM rebase response.

    Returns (should_rebase, target, reasoning) or None.
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
            # Support both old (should_rebase) and new (decision+params)
            should_rebase = bool(
                data.get("should_rebase", data.get("decision", False))
            )
            target = data.get("target_branch")
            if target is None and isinstance(data.get("params"), dict):
                target = data["params"].get("target_branch")
            if target is not None:
                target = str(target).strip() or None
            return should_rebase, target, reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _parse_branch_response(
    text: str,
) -> tuple[bool, str | None, str] | None:
    """Parse an LLM branch response.

    Returns (should_branch, name, reasoning) or None.
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
            # Support both old (should_branch) and new (decision+params)
            should_branch = bool(
                data.get("should_branch", data.get("decision", False))
            )
            name = data.get("branch_name")
            if name is None and isinstance(data.get("params"), dict):
                name = data["params"].get("branch_name")
            if name is not None:
                name = str(name).strip() or None
            return should_branch, name, reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# auto_split: sync / async
# ---------------------------------------------------------------------------

def auto_split(
    tract: Tract,
    commit_hash: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoSplitResult:
    """Split a commit into smaller, logically coherent pieces using LLM \
judgment.

    Gets the commit content, asks an LLM to split it into pieces, then
    creates new APPEND commits for each piece and EDITs the original to \
SKIP it.

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
    return _auto_split_core(
        tract, commit_hash, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


async def aauto_split(
    tract: Tract,
    commit_hash: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoSplitResult:
    """Async version of :func:`auto_split`."""
    return await _aauto_split_core(
        tract, commit_hash, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


def _auto_split_core(
    tract: Tract,
    commit_hash: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoSplitResult:
    """Core auto_split implementation using Judgment."""
    fail_result = AutoSplitResult(
        original_hash=commit_hash,
        new_hashes=(commit_hash,),
        split_count=1,
        tokens_used=0,
        reasoning="No split performed (fail-open).",
        consulted_hashes=(),
    )

    try:
        content = tract.search.get_content(commit_hash)
        if content is None:
            return fail_result
        content_str = (
            json.dumps(content, default=str)
            if isinstance(content, dict)
            else str(content)
        )
    except Exception:
        logger.warning(
            "Failed to get content for commit %s; no split.",
            commit_hash[:12], exc_info=True,
        )
        return fail_result

    consulted_hashes = (commit_hash,)
    prompt_override = tract.config.get_prompt("split")

    judgment = Judgment(
        instructions=f"=== COMMIT CONTENT ===\n{content_str}",
        response_model=SplitPlan,
        system_prompt=prompt_override or _SPLIT_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.2,
        max_tokens=max_tokens,
        operation_name="autonomous",
    )

    result = judgment.evaluate(tract)
    return _finalize_split(
        result, tract, commit_hash, consulted_hashes,
    )


async def _aauto_split_core(
    tract: Tract,
    commit_hash: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoSplitResult:
    """Async core auto_split implementation using Judgment."""
    fail_result = AutoSplitResult(
        original_hash=commit_hash,
        new_hashes=(commit_hash,),
        split_count=1,
        tokens_used=0,
        reasoning="No split performed (fail-open).",
        consulted_hashes=(),
    )

    try:
        content = tract.search.get_content(commit_hash)
        if content is None:
            return fail_result
        content_str = (
            json.dumps(content, default=str)
            if isinstance(content, dict)
            else str(content)
        )
    except Exception:
        logger.warning(
            "Failed to get content for commit %s; no split.",
            commit_hash[:12], exc_info=True,
        )
        return fail_result

    consulted_hashes = (commit_hash,)
    prompt_override = tract.config.get_prompt("split")

    judgment = Judgment(
        instructions=f"=== COMMIT CONTENT ===\n{content_str}",
        response_model=SplitPlan,
        system_prompt=prompt_override or _SPLIT_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.2,
        max_tokens=max_tokens,
        operation_name="autonomous",
    )

    result = await judgment.aevaluate(tract)
    return _finalize_split(
        result, tract, commit_hash, consulted_hashes,
    )


def _finalize_split(
    result: Any,
    tract: Tract,
    commit_hash: str,
    consulted_hashes: tuple[str, ...],
) -> AutoSplitResult:
    """Shared post-Judgment logic for auto_split."""
    if not result.succeeded or result.output is None:
        base = result.reasoning or "No split performed"
        reasoning = (
            base if "fail-open" in base.lower()
            else f"{base}; no split performed (fail-open)."
        )
        return AutoSplitResult(
            original_hash=commit_hash,
            new_hashes=(commit_hash,),
            split_count=1,
            tokens_used=result.tokens_used,
            reasoning=reasoning,
            consulted_hashes=consulted_hashes,
        )

    pieces = result.output.pieces
    if not pieces:
        return AutoSplitResult(
            original_hash=commit_hash,
            new_hashes=(commit_hash,),
            split_count=1,
            tokens_used=result.tokens_used,
            reasoning="LLM returned no split pieces; keeping original.",
            consulted_hashes=consulted_hashes,
        )

    valid_pieces: list[dict[str, str]] = []
    for piece in pieces:
        if isinstance(piece, dict) and "content" in piece:
            valid_pieces.append({
                "content": str(piece["content"]),
                "message": str(piece.get("message", "Split piece")),
            })

    if not valid_pieces:
        return AutoSplitResult(
            original_hash=commit_hash,
            new_hashes=(commit_hash,),
            split_count=1,
            tokens_used=result.tokens_used,
            reasoning="LLM returned no valid split pieces; keeping original.",
            consulted_hashes=consulted_hashes,
        )

    return _execute_split(
        tract, commit_hash, valid_pieces,
        result.tokens_used, consulted_hashes,
    )


def _execute_split(
    tract: Tract,
    original_hash: str,
    pieces: list[dict[str, str]],
    tokens_used: int,
    consulted_hashes: tuple[str, ...] = (),
) -> AutoSplitResult:
    """Create new commits for each split piece and SKIP the original."""
    from tract.models.annotations import Priority
    from tract.models.commit import CommitOperation

    new_hashes: list[str] = []
    reasoning_parts: list[str] = []

    for piece in pieces:
        try:
            info = tract.commit(
                {
                    "content_type": "freeform",
                    "payload": {"text": piece["content"]},
                },
                operation=CommitOperation.APPEND,
                message=piece["message"],
            )
            new_hashes.append(info.commit_hash)
            reasoning_parts.append(
                f"Created: {info.commit_hash[:8]} - {piece['message']}"
            )
        except Exception as exc:
            logger.warning(
                "Failed to create split commit: %s", exc, exc_info=True,
            )

    if not new_hashes:
        return AutoSplitResult(
            original_hash=original_hash,
            new_hashes=(original_hash,),
            split_count=1,
            tokens_used=tokens_used,
            reasoning="All split commit creations failed; keeping original.",
            consulted_hashes=consulted_hashes,
        )

    try:
        tract.annotations.set(
            original_hash, Priority.SKIP,
            reason="Split into smaller commits",
        )
    except Exception:
        logger.warning(
            "Failed to SKIP original commit %s after split.",
            original_hash[:12], exc_info=True,
        )

    reasoning = (
        f"Split into {len(new_hashes)} pieces. "
        + "; ".join(reasoning_parts)
    )

    return AutoSplitResult(
        original_hash=original_hash,
        new_hashes=tuple(new_hashes),
        split_count=len(new_hashes),
        tokens_used=tokens_used,
        reasoning=reasoning,
        consulted_hashes=consulted_hashes,
    )


# ---------------------------------------------------------------------------
# auto_rebase: sync / async
# ---------------------------------------------------------------------------

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
    return _auto_rebase_core(
        tract, model=model, temperature=temperature,
        max_tokens=max_tokens,
    )


async def aauto_rebase(
    tract: Tract,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoRebaseResult:
    """Async version of :func:`auto_rebase`."""
    return await _aauto_rebase_core(
        tract, model=model, temperature=temperature,
        max_tokens=max_tokens,
    )


def _auto_rebase_core(
    tract: Tract,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoRebaseResult:
    """Core auto_rebase implementation using Judgment."""
    manifest = _build_rebase_manifest(tract)
    entries = tract.search.log(limit=10)
    consulted = tuple(e.commit_hash for e in entries) if entries else ()
    prompt_override = tract.config.get_prompt("rebase")

    judgment = Judgment(
        instructions=manifest,
        response_model=BooleanDecision,
        system_prompt=prompt_override or _REBASE_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.2,
        max_tokens=max_tokens,
        operation_name="autonomous",
    )

    result = judgment.evaluate(tract)
    return _finalize_rebase(result, tract, consulted)


async def _aauto_rebase_core(
    tract: Tract,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoRebaseResult:
    """Async core auto_rebase implementation using Judgment."""
    manifest = _build_rebase_manifest(tract)
    entries = tract.search.log(limit=10)
    consulted = tuple(e.commit_hash for e in entries) if entries else ()
    prompt_override = tract.config.get_prompt("rebase")

    judgment = Judgment(
        instructions=manifest,
        response_model=BooleanDecision,
        system_prompt=prompt_override or _REBASE_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.2,
        max_tokens=max_tokens,
        operation_name="autonomous",
    )

    result = await judgment.aevaluate(tract)
    return _finalize_rebase(result, tract, consulted)


def _finalize_rebase(
    result: Any,
    tract: Tract,
    consulted_hashes: tuple[str, ...],
) -> AutoRebaseResult:
    """Shared post-Judgment logic for auto_rebase."""
    if not result.succeeded or result.output is None:
        base = result.reasoning or "No rebase performed"
        reason = (
            base if "fail-open" in base.lower()
            else f"{base}; no rebase performed (fail-open)."
        )
        return AutoRebaseResult(
            rebased=False,
            reason=reason,
            target_branch=None,
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )

    out = result.output
    # Support both new (decision+params) and old (should_rebase+target_branch)
    should_rebase = out.decision or bool(getattr(out, "should_rebase", False))
    target_branch = (
        out.params.get("target_branch")
        if out.params
        else None
    )
    # Fallback: old format puts target_branch as a top-level field
    if target_branch is None:
        target_branch = getattr(out, "target_branch", None)
    reasoning = out.reasoning or result.reasoning

    if target_branch is not None:
        target_branch = str(target_branch).strip() or None

    if not should_rebase or not target_branch:
        return AutoRebaseResult(
            rebased=False,
            reason=reasoning,
            target_branch=None,
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )

    try:
        tract.rebase(target_branch)
        return AutoRebaseResult(
            rebased=True,
            reason=reasoning,
            target_branch=target_branch,
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )
    except Exception as exc:
        logger.warning(
            "Auto-rebase onto '%s' failed: %s",
            target_branch, exc, exc_info=True,
        )
        return AutoRebaseResult(
            rebased=False,
            reason=f"Rebase failed: {exc}",
            target_branch=target_branch,
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )


# ---------------------------------------------------------------------------
# auto_branch: sync / async
# ---------------------------------------------------------------------------

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
    return _auto_branch_core(
        tract, context=context, model=model,
        temperature=temperature, max_tokens=max_tokens,
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
    return await _aauto_branch_core(
        tract, context=context, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


def _auto_branch_core(
    tract: Tract,
    *,
    context: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoBranchResult:
    """Core auto_branch implementation using Judgment."""
    manifest = _build_branch_manifest(tract, context)
    entries = tract.search.log(limit=10)
    consulted = tuple(e.commit_hash for e in entries) if entries else ()
    prompt_override = tract.config.get_prompt("branch")

    judgment = Judgment(
        instructions=manifest,
        response_model=BooleanDecision,
        system_prompt=prompt_override or _BRANCH_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.2,
        max_tokens=max_tokens,
        operation_name="autonomous",
    )

    result = judgment.evaluate(tract)
    return _finalize_branch(result, tract, consulted)


async def _aauto_branch_core(
    tract: Tract,
    *,
    context: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AutoBranchResult:
    """Async core auto_branch implementation using Judgment."""
    manifest = _build_branch_manifest(tract, context)
    entries = tract.search.log(limit=10)
    consulted = tuple(e.commit_hash for e in entries) if entries else ()
    prompt_override = tract.config.get_prompt("branch")

    judgment = Judgment(
        instructions=manifest,
        response_model=BooleanDecision,
        system_prompt=prompt_override or _BRANCH_SYSTEM_PROMPT,
        context=ContextView(scope=0),
        model=model,
        temperature=temperature if temperature is not None else 0.2,
        max_tokens=max_tokens,
        operation_name="autonomous",
    )

    result = await judgment.aevaluate(tract)
    return _finalize_branch(result, tract, consulted)


def _finalize_branch(
    result: Any,
    tract: Tract,
    consulted_hashes: tuple[str, ...],
) -> AutoBranchResult:
    """Shared post-Judgment logic for auto_branch."""
    if not result.succeeded or result.output is None:
        base = result.reasoning or "No branch created"
        reason = (
            base if "fail-open" in base.lower()
            else f"{base}; no branch created (fail-open)."
        )
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason=reason,
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )

    out = result.output
    # Support both new (decision+params) and old (should_branch+branch_name)
    should_branch = out.decision or bool(getattr(out, "should_branch", False))
    branch_name = (
        out.params.get("branch_name")
        if out.params
        else None
    )
    # Fallback: old format puts branch_name as a top-level field
    if branch_name is None:
        branch_name = getattr(out, "branch_name", None)
    reasoning = out.reasoning or result.reasoning

    if branch_name is not None:
        branch_name = str(branch_name).strip() or None

    if not should_branch or not branch_name:
        return AutoBranchResult(
            branched=False,
            branch_name=None,
            reason=reasoning,
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )

    try:
        tract.branches.create(branch_name)
        return AutoBranchResult(
            branched=True,
            branch_name=branch_name,
            reason=reasoning,
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )
    except Exception as exc:
        logger.warning(
            "Auto-branch '%s' failed: %s",
            branch_name, exc, exc_info=True,
        )
        return AutoBranchResult(
            branched=False,
            branch_name=branch_name,
            reason=f"Branch creation failed: {exc}",
            tokens_used=result.tokens_used,
            consulted_hashes=consulted_hashes,
        )
