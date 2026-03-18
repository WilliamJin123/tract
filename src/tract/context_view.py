"""ContextView: unified context specification for LLM-powered operations.

Every LLM-powered operation in tract reads some slice of the commit DAG
at some level of detail.  ContextView is the uniform specification.

Three detail levels:

- **manifest**: Commit metadata (hash, type, tokens, tags, priority, message).
  Cheapest.  Good for decisions about what exists.
- **full**: Complete content per commit with boundaries preserved.
  Good when the LLM needs to read actual content.
- **compiled**: Assembled into conversation via ContextCompiler.
  Good for "what's the conversation so far?" context.

Example::

    from tract.context_view import ContextView, build_context

    # Manifest of last 20 commits + config state
    view = ContextView(scope=20)
    built = build_context(view, tract)

    # Full content of specific commits
    view = ContextView(scope=["abc123", "def456"], detail="full")
    built = build_context(view, tract)

    # Compiled conversation, no state metadata
    view = ContextView(detail="compiled", include_config=False)
    built = build_context(view, tract)

    # Manifest with selective peek at important commits
    view = ContextView(scope=30, peek=["abc123"])
    built = build_context(view, tract)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.tract import Tract

__all__: list[str] = [
    "ContextView",
    "BuiltContext",
    "build_context",
    "resolve_auto_peek",
    "estimate_tokens",
]

logger = logging.getLogger(__name__)

_PRIORITY_ORDER = {"skip": 0, "normal": 1, "important": 2, "pinned": 3}


# ---------------------------------------------------------------------------
# ContextView — the specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextView:
    """Specifies what context to extract from the commit DAG.

    Every LLM-powered operation (gates, maintainers, compaction, cherry-pick,
    dedup, collapse, autonomous ops) needs "some slice of the DAG at some
    resolution."  ContextView is the uniform specification.

    Scope/filter/budget control *which* commits.
    Detail controls *how much* of each commit.
    State flags control *what metadata* accompanies the commits.
    """

    # ── Scope: which commits ───────────────────────────────
    scope: int | list[str] | None = None
    """``int`` → last N commits from HEAD.
    ``list[str]`` → exactly these commit hashes (or prefixes).
    ``None`` → operation picks its own default via ``default_scope``."""

    since: str | None = None
    """Commit hash cutoff: include this commit and everything newer."""

    branch: str | None = None
    """Read from this branch instead of current.  ``None`` → current HEAD."""

    # ── Filter: narrow within scope ────────────────────────
    include_types: list[str] | None = None
    exclude_types: list[str] | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    min_priority: str | None = None
    """Priority floor: ``"normal"`` | ``"important"`` | ``"pinned"``.
    ``"important"`` → includes IMPORTANT + PINNED, excludes SKIP + NORMAL."""

    # ── Forced inclusion (additive, outside scope window) ──
    always_include_tags: list[str] | None = None
    always_include_hashes: list[str] | None = None
    """Commits matching these criteria are included regardless of scope/filter.
    Forced commits are NOT subject to type/tag/priority filters."""

    # ── Budget ─────────────────────────────────────────────
    max_tokens: int | None = None
    """Hard cap on estimated output tokens.  ``build_context`` fills greedily
    from forced includes first, then newest commits, stopping when exhausted."""

    # ── Detail: how much per commit ────────────────────────
    detail: Literal["manifest", "full", "compiled"] = "manifest"
    """``"manifest"`` → hash, type, tokens, tags, priority, message.
    ``"full"`` → complete content per commit with boundaries.
    ``"compiled"`` → assembled into conversation via ContextCompiler."""

    message_chars: int = 80
    """Commit message truncation length in manifest/full modes."""

    # ── Peek: selective full-content overlay ────────────────
    peek: list[str] | None = None
    """Specific commit hashes shown at full content; rest stay at ``detail`` level."""

    auto_peek: int | None = None
    """LLM auto-selects up to N commits for full content (two-pass).
    Call :func:`resolve_auto_peek` before :func:`build_context` to resolve."""

    peek_chars: int = 0
    """Max chars per peeked commit content.  ``0`` → no truncation."""

    # ── State: tract metadata sections ─────────────────────
    include_config: bool = True
    include_branches: bool = True
    include_tags_summary: bool = True
    include_directives: bool = False
    include_token_status: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting fields at their default values."""
        import dataclasses
        defaults = ContextView()
        result: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            default_val = getattr(defaults, f.name)
            if val != default_val:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextView:
        """Reconstruct a ContextView from a dict (inverse of to_dict)."""
        return cls(**data)


# ---------------------------------------------------------------------------
# BuiltContext — the materialized result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuiltContext:
    """Materialized context from a ContextView."""

    text: str
    """Formatted text for prompt injection."""

    commit_count: int
    """Number of commits included."""

    token_estimate: int
    """Approximate token count of ``text`` (len/4 heuristic)."""

    peeked_hashes: tuple[str, ...] = ()
    """Which commits received full-content treatment."""

    commit_entries: tuple[dict[str, Any], ...] = ()
    """Per-commit metadata dicts for response hash resolution.
    Each entry: ``{"hash", "short_hash", "content_type", "token_count",
    "tags", "priority"}``."""


# ---------------------------------------------------------------------------
# build_context — the materializer
# ---------------------------------------------------------------------------

def build_context(
    view: ContextView,
    tract: Tract,
    *,
    default_scope: int | None = None,
) -> BuiltContext:
    """Materialize a ContextView into formatted text for prompt injection.

    Pure function — no LLM calls.  If ``view.auto_peek`` is set, call
    :func:`resolve_auto_peek` first to resolve it to explicit hashes.

    Args:
        view: The context specification.
        tract: The Tract instance to read from.
        default_scope: Fallback scope when ``view.scope is None``.
            Each operation passes its own sensible default.

    Note:
        ``detail="compiled"`` calls ``tract.compile()`` which fires
        ``pre_compile`` middleware.  Avoid using ``compiled`` detail
        from within middleware handlers to prevent recursion.
    """
    # 1. Resolve scope → list of CommitInfo
    entries = _resolve_entries(view, tract, default_scope)

    # 2. Apply filters
    entries = _apply_filters(view, entries)

    # 3. Add forced includes
    forced_hashes: set[str] = set()
    entries, forced_hashes = _add_forced_includes(view, tract, entries)

    # 4. Apply token budget
    if view.max_tokens is not None:
        entries = _apply_budget(view, entries, forced_hashes)

    # 5. Build commit_entries metadata (for hash resolution in responses)
    commit_entries = _build_commit_entries(entries)

    # 6. Format commits based on detail level
    peek_set = _resolve_peek_set(view, entries)
    if view.detail == "compiled":
        commit_text = _format_compiled(tract)
    else:
        commit_text = _format_commits(entries, tract, view, peek_set)

    # 7. Build state sections
    state_text = _format_state(view, tract, entries)

    # 8. Combine
    parts = [p for p in (commit_text, state_text) if p]
    text = "\n\n".join(parts)

    # 9. Token estimate
    token_est = max(1, len(text) // 4)

    return BuiltContext(
        text=text,
        commit_count=len(entries),
        token_estimate=token_est,
        peeked_hashes=tuple(sorted(peek_set)),
        commit_entries=tuple(commit_entries),
    )


# ---------------------------------------------------------------------------
# Internal: commit resolution
# ---------------------------------------------------------------------------

def _resolve_entries(
    view: ContextView,
    tract: Tract,
    default_scope: int | None,
) -> list[CommitInfo]:
    """Resolve scope to a list of CommitInfo entries (newest first)."""
    scope = view.scope if view.scope is not None else default_scope

    if isinstance(scope, list):
        # Explicit hashes — look up from full log
        log = list(tract.search.log(limit=500))
        log_map = {e.commit_hash: e for e in log}
        result: list[CommitInfo] = []
        seen: set[str] = set()
        for h in scope:
            if h in log_map and h not in seen:
                result.append(log_map[h])
                seen.add(h)
            else:
                # Try prefix match
                for full_h, entry in log_map.items():
                    if full_h.startswith(h) and full_h not in seen:
                        result.append(entry)
                        seen.add(full_h)
                        break
        return result

    # int or None scope — last N commits
    limit = scope if scope is not None else 500
    entries = list(tract.search.log(limit=limit))

    # Apply 'since' cutoff if set
    if view.since:
        cutoff_idx = None
        for i, e in enumerate(entries):
            if e.commit_hash == view.since or e.commit_hash.startswith(view.since):
                cutoff_idx = i
                break
        if cutoff_idx is not None:
            entries = entries[: cutoff_idx + 1]

    return entries


def _apply_filters(
    view: ContextView,
    entries: list[CommitInfo],
) -> list[CommitInfo]:
    """Apply type, tag, and priority filters.  Scope first, then narrow."""
    result = entries

    if view.include_types:
        types = set(view.include_types)
        result = [e for e in result if e.content_type in types]

    if view.exclude_types:
        types = set(view.exclude_types)
        result = [e for e in result if e.content_type not in types]

    if view.include_tags:
        tags = set(view.include_tags)
        result = [e for e in result if tags.intersection(e.tags or [])]

    if view.exclude_tags:
        tags = set(view.exclude_tags)
        result = [e for e in result if not tags.intersection(e.tags or [])]

    if view.min_priority:
        floor = _PRIORITY_ORDER.get(view.min_priority.lower(), 0)
        result = [
            e for e in result
            if _PRIORITY_ORDER.get(
                (e.effective_priority or "normal").lower(), 1,
            ) >= floor
        ]

    return result


def _add_forced_includes(
    view: ContextView,
    tract: Tract,
    entries: list[CommitInfo],
) -> tuple[list[CommitInfo], set[str]]:
    """Add forced-include commits outside the scope window.

    Returns (combined entries, set of forced hashes).
    """
    existing = {e.commit_hash for e in entries}
    extras: list[CommitInfo] = []
    forced: set[str] = set()

    need_full_log = bool(view.always_include_hashes or view.always_include_tags)
    full_log: dict[str, CommitInfo] = {}
    if need_full_log:
        full_log = {e.commit_hash: e for e in tract.search.log(limit=500)}

    if view.always_include_hashes:
        for h in view.always_include_hashes:
            if h in full_log and h not in existing:
                extras.append(full_log[h])
                existing.add(h)
                forced.add(h)
            else:
                # Prefix match
                for fh, entry in full_log.items():
                    if fh.startswith(h) and fh not in existing:
                        extras.append(entry)
                        existing.add(fh)
                        forced.add(fh)
                        break

    if view.always_include_tags:
        tags = set(view.always_include_tags)
        for fh, entry in full_log.items():
            if fh not in existing and tags.intersection(entry.tags or []):
                extras.append(entry)
                existing.add(fh)
                forced.add(fh)

    # Forced includes prepended (they're high-priority)
    return extras + entries, forced


def _apply_budget(
    view: ContextView,
    entries: list[CommitInfo],
    forced_hashes: set[str],
) -> list[CommitInfo]:
    """Trim entries to fit within max_tokens budget.

    Forced includes are guaranteed.  Remaining budget filled newest-first.
    """
    assert view.max_tokens is not None
    budget = view.max_tokens

    forced = [e for e in entries if e.commit_hash in forced_hashes]
    regular = [e for e in entries if e.commit_hash not in forced_hashes]

    result = list(forced)
    for e in forced:
        budget -= _entry_token_cost(e, view.detail)

    for e in regular:
        cost = _entry_token_cost(e, view.detail)
        if budget - cost < 0:
            break
        result.append(e)
        budget -= cost

    return result


def _entry_token_cost(entry: CommitInfo, detail: str) -> int:
    """Estimate token cost for a single entry at the given detail level."""
    if detail == "manifest":
        return 25  # metadata line ≈ 25 tokens
    else:
        return max(entry.token_count, 25)


# ---------------------------------------------------------------------------
# Internal: commit metadata for response resolution
# ---------------------------------------------------------------------------

def _build_commit_entries(entries: list[CommitInfo]) -> list[dict[str, Any]]:
    """Build metadata dicts for hash resolution in LLM responses."""
    return [
        {
            "hash": e.commit_hash,
            "short_hash": e.commit_hash[:8],
            "content_type": e.content_type,
            "token_count": e.token_count,
            "tags": list(e.tags) if e.tags else [],
            "priority": e.effective_priority or "normal",
        }
        for e in entries
    ]


def _resolve_peek_set(view: ContextView, entries: list[CommitInfo]) -> set[str]:
    """Build the set of commit hashes that should get full content."""
    if not view.peek:
        return set()
    peek_resolved: set[str] = set()
    entry_hashes = {e.commit_hash for e in entries}
    for ph in view.peek:
        if ph in entry_hashes:
            peek_resolved.add(ph)
        else:
            # Prefix match
            for eh in entry_hashes:
                if eh.startswith(ph):
                    peek_resolved.add(eh)
                    break
    return peek_resolved


# ---------------------------------------------------------------------------
# Internal: formatting
# ---------------------------------------------------------------------------

def _format_commits(
    entries: list[CommitInfo],
    tract: Tract,
    view: ContextView,
    peek_set: set[str],
) -> str:
    """Format commits as manifest or full, with optional peek overlay."""
    if not entries:
        return "=== CONTEXT ===\n(no commits)"

    is_full = view.detail == "full"
    lines: list[str] = [
        "=== CONTEXT MANIFEST ===" if not is_full else "=== CONTEXT ===",
        f"Commits shown: {len(entries)}",
        "",
    ]
    if not is_full:
        lines.append("COMMIT LOG (newest first):")

    for entry in entries:
        h = entry.commit_hash[:8]
        ctype = entry.content_type
        tokens = entry.token_count
        tags_str = ",".join(entry.tags) if entry.tags else ""
        priority = entry.effective_priority or "normal"
        msg = entry.message or "(no message)"
        if len(msg) > view.message_chars:
            msg = msg[: view.message_chars - 3] + "..."

        meta_line = (
            f"  [{h}] {ctype:<12} | {tokens:>5} tok | "
            f"tags:[{tags_str}] | {priority:<9} | \"{msg}\""
        )

        # Determine if this commit gets full content
        show_content = is_full or entry.commit_hash in peek_set

        if show_content:
            lines.append(f"--- [{h}] {ctype} | {tokens} tok | "
                         f"tags:[{tags_str}] | {priority} | \"{msg}\" ---")
            content_str = _get_content_str(tract, entry.commit_hash, view.peek_chars)
            lines.append(content_str)
            lines.append("")
        else:
            lines.append(meta_line)

    return "\n".join(lines)


def _format_compiled(tract: Tract) -> str:
    """Format as compiled conversation context."""
    try:
        compiled = tract.compile()
        messages = compiled.messages if hasattr(compiled, "messages") else []
        lines: list[str] = ["=== COMPILED CONTEXT ===", ""]
        for msg in messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            lines.append(f"[{role}]: {content}")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Failed to compile context: %s", exc, exc_info=True)
        return "=== COMPILED CONTEXT ===\n(compilation failed)"


def _get_content_str(tract: Tract, commit_hash: str, max_chars: int = 0) -> str:
    """Fetch commit content as a string."""
    try:
        content = tract.search.get_content(commit_hash)
        if content is None:
            return "(no content)"
        text = (
            json.dumps(content, default=str, indent=2)
            if isinstance(content, dict)
            else str(content)
        )
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception:
        return "(content unavailable)"


# ---------------------------------------------------------------------------
# Internal: state formatting
# ---------------------------------------------------------------------------

def _format_state(
    view: ContextView,
    tract: Tract,
    entries: list[CommitInfo],
) -> str:
    """Format tract state metadata sections."""
    sections: list[str] = []

    # Branch + HEAD header (always included as orientation)
    branch = tract.current_branch or "(detached)"
    head = tract.head
    head_short = head[:8] if head else "(empty)"
    sections.append(f"Branch: {branch} | HEAD: {head_short}")

    if view.include_config:
        try:
            config = tract.config.get_all()
            if config:
                sections.append(
                    f"ACTIVE CONFIG: {json.dumps(config, default=str)}"
                )
        except Exception:
            pass

    if view.include_branches:
        try:
            branches = tract.branches.list()
            if branches:
                branch_lines = ["BRANCHES:"]
                for b in branches:
                    marker = " *" if b.is_current else ""
                    bh = b.commit_hash[:8] if b.commit_hash else "(empty)"
                    branch_lines.append(f"  {b.name}{marker} -> {bh}")
                sections.append("\n".join(branch_lines))
        except Exception:
            pass

    if view.include_tags_summary and entries:
        tag_counter: Counter[str] = Counter()
        for entry in entries:
            for tag in entry.tags or []:
                tag_counter[tag] += 1
        if tag_counter:
            tag_summary = ", ".join(
                f"{tag}({count})" for tag, count in tag_counter.most_common()
            )
            sections.append(f"TAGS: {tag_summary}")

    if view.include_directives:
        try:
            # Directives are instruction commits; deduplicate by name
            # (closest to HEAD wins = first in log since newest-first)
            all_entries = tract.search.log(limit=200)
            seen_names: set[str] = set()
            dir_lines: list[str] = ["ACTIVE DIRECTIVES:"]
            found = False
            for entry in all_entries:
                if entry.content_type == "instruction":
                    # Extract name from message "directive: <name>"
                    msg = entry.message or ""
                    name = msg.replace("directive: ", "", 1) if msg.startswith("directive: ") else msg
                    if not name:
                        name = entry.commit_hash[:8]
                    if name not in seen_names:
                        seen_names.add(name)
                        content = tract.search.get_content(entry.commit_hash)
                        text = str(content) if content is not None else "(empty)"
                        dir_lines.append(f"  {name}: {text}")
                        found = True
            if found:
                sections.append("\n".join(dir_lines))
        except Exception:
            pass

    if view.include_token_status:
        try:
            status = tract.search.status()
            sections.append(f"TOKEN STATUS: {status.token_count} tokens")
        except Exception:
            pass

    return "\n".join(sections) if sections else ""


# ---------------------------------------------------------------------------
# resolve_auto_peek — two-pass LLM peek resolution
# ---------------------------------------------------------------------------

_PEEK_SYSTEM_PROMPT = """\
You are a context inspector. You will receive a manifest of commits (metadata only). \
Select the commits whose FULL CONTENT would be most useful to inspect for the given task.

Respond with JSON:
{"selected": ["<hash1>", "<hash2>", ...]}

Select at most the specified limit. Only select commits where reading the full content \
would materially improve decision quality. Use the short hash prefixes shown."""


async def resolve_auto_peek(
    view: ContextView,
    tract: Tract,
    llm_client: Any,
    *,
    task_hint: str = "",
    default_scope: int | None = None,
    **llm_kwargs: Any,
) -> ContextView:
    """If ``auto_peek`` is set, ask an LLM which commits to peek at.

    Returns a new ContextView with ``auto_peek`` cleared and ``peek``
    set to the LLM-selected hashes.  If ``auto_peek`` is ``None`` or 0,
    returns the view unchanged.
    """
    if not view.auto_peek:
        return view

    import dataclasses

    from tract._helpers import async_safe_llm_call, strip_fences

    # Build manifest-only context for the LLM
    manifest_view = dataclasses.replace(
        view,
        detail="manifest",
        peek=None,
        auto_peek=None,
        include_config=False,
        include_branches=False,
        include_tags_summary=False,
        include_directives=False,
        include_token_status=False,
    )
    built = build_context(manifest_view, tract, default_scope=default_scope)

    user_content = (
        f"=== PEEK SELECTION ===\n"
        f"Select up to {view.auto_peek} commits to inspect in full.\n"
    )
    if task_hint:
        user_content += f"\nTask context: {task_hint}\n"
    user_content += f"\n{built.text}"

    messages = [
        {"role": "system", "content": _PEEK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    kw: dict[str, Any] = {"temperature": 0.1}
    kw.update(llm_kwargs)

    result = await async_safe_llm_call(llm_client, messages, kw)
    if result is None:
        return dataclasses.replace(view, auto_peek=None, peek=None)

    raw_text, _ = result
    cleaned = strip_fences(raw_text)

    selected: list[str] = []
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            raw_sel = data.get("selected", [])
            if isinstance(raw_sel, list):
                for h in raw_sel[: view.auto_peek]:
                    h_str = str(h).strip()
                    for entry in built.commit_entries:
                        if (
                            entry["hash"].startswith(h_str)
                            or entry["short_hash"] == h_str
                        ):
                            selected.append(entry["hash"])
                            break
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return dataclasses.replace(
        view,
        auto_peek=None,
        peek=selected or None,
    )


# ---------------------------------------------------------------------------
# estimate_tokens — cheap pre-materialization estimate
# ---------------------------------------------------------------------------

def estimate_tokens(
    view: ContextView,
    tract: Tract,
    *,
    default_scope: int | None = None,
) -> int:
    """Cheap token estimate without full materialization (~80% accurate).

    Walks matching commits, sums raw token counts, applies a detail-level
    multiplier.  Use for budgeting decisions, not billing.
    """
    entries = _resolve_entries(view, tract, default_scope)
    entries = _apply_filters(view, entries)
    entries, _ = _add_forced_includes(view, tract, entries)

    raw_tokens = sum(e.token_count for e in entries)
    metadata_overhead = len(entries) * 25  # ~25 tokens per metadata line

    if view.detail == "manifest":
        return metadata_overhead
    elif view.detail == "compiled":
        return raw_tokens
    else:  # full
        return metadata_overhead + raw_tokens
