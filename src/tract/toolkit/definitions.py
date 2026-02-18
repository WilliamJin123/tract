"""Hand-crafted tool definitions for all Tract operations.

Each tool definition includes a clear, action-oriented description,
proper JSON Schema parameters, and a handler lambda bound to a specific
Tract instance.  Handler lambdas use explicit parameter whitelisting
(no ``**kwargs`` passthrough) to prevent hallucinated arguments.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tract.toolkit.models import ToolDefinition

if TYPE_CHECKING:
    from tract.tract import Tract

logger = logging.getLogger(__name__)


def get_all_tools(tract: Tract) -> list[ToolDefinition]:
    """Build tool definitions for all Tract operations.

    Each call returns fresh lambdas bound to the passed ``tract`` instance.
    No module-level references to any tract are stored.

    Args:
        tract: The Tract instance to bind tool handlers to.

    Returns:
        List of 15 ToolDefinition objects.
    """
    return [
        # 1. commit
        ToolDefinition(
            name="commit",
            description=(
                "Record new context into the tract. Use this to add messages, "
                "instructions, tool results, or any content to the conversation "
                "history. The content dict must include a 'content_type' field "
                "(e.g. 'dialogue', 'instruction', 'tool_io', 'reasoning', "
                "'artifact', 'output', 'freeform')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "description": (
                            "Content dict with 'content_type' field. "
                            "For dialogue: {content_type: 'dialogue', role: 'user'|'assistant'|'system', text: '...'}. "
                            "For instruction: {content_type: 'instruction', text: '...'}."
                        ),
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["append", "edit"],
                        "description": "Operation type. 'append' adds new content (default), 'edit' replaces an existing commit.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional human-readable commit message.",
                    },
                    "response_to": {
                        "type": "string",
                        "description": "For edit operations, the hash of the commit being replaced.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional arbitrary metadata dict.",
                    },
                    "generation_config": {
                        "type": "object",
                        "description": "Optional LLM generation config (temperature, model, etc.).",
                    },
                },
                "required": ["content"],
            },
            handler=lambda content, operation="append", message=None, response_to=None, metadata=None, generation_config=None: _handle_commit(
                tract, content, operation, message, response_to, metadata, generation_config
            ),
        ),
        # 2. compile
        ToolDefinition(
            name="compile",
            description=(
                "Compile the current context into LLM-ready messages. Returns a "
                "summary of the compiled context including token count and message "
                "count. Use this to check what the current context looks like "
                "before making an API call."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=lambda: _handle_compile(tract),
        ),
        # 3. annotate
        ToolDefinition(
            name="annotate",
            description=(
                "Set a priority annotation on a commit. Use 'pinned' to protect "
                "important context from compression, 'skip' to exclude irrelevant "
                "content from compilation, or 'normal' to reset."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_hash": {
                        "type": "string",
                        "description": "Hash of the commit to annotate.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["pinned", "normal", "skip"],
                        "description": "Priority level: 'pinned' (protected), 'normal' (default), 'skip' (excluded).",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for the annotation.",
                    },
                },
                "required": ["target_hash", "priority"],
            },
            handler=lambda target_hash, priority, reason=None: _handle_annotate(
                tract, target_hash, priority, reason
            ),
        ),
        # 4. status
        ToolDefinition(
            name="status",
            description=(
                "Get the current tract status including branch name, HEAD commit, "
                "token count, and budget usage percentage. Use this to understand "
                "the current state before deciding on next actions."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=lambda: _handle_status(tract),
        ),
        # 5. log
        ToolDefinition(
            name="log",
            description=(
                "View recent commit history from HEAD backward. Returns commit "
                "hashes, messages, content types, and token counts. Use this to "
                "understand what content has been recorded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of commits to return (default 20).",
                    },
                    "op_filter": {
                        "type": "string",
                        "enum": ["append", "edit"],
                        "description": "Filter by operation type.",
                    },
                },
            },
            handler=lambda limit=20, op_filter=None: _handle_log(
                tract, limit, op_filter
            ),
        ),
        # 6. diff
        ToolDefinition(
            name="diff",
            description=(
                "Compare two commits and show what changed between them. "
                "Useful for reviewing edits or understanding how context evolved."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "commit_a": {
                        "type": "string",
                        "description": "First commit hash (defaults to parent of commit_b).",
                    },
                    "commit_b": {
                        "type": "string",
                        "description": "Second commit hash (defaults to HEAD).",
                    },
                },
            },
            handler=lambda commit_a=None, commit_b=None: _handle_diff(
                tract, commit_a, commit_b
            ),
        ),
        # 7. compress
        ToolDefinition(
            name="compress",
            description=(
                "Compress a range of commits into a summary to reduce token usage. "
                "Pinned commits are preserved verbatim. Requires an LLM client to be "
                "configured via tract.configure_llm() unless manual content is provided."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_tokens": {
                        "type": "integer",
                        "description": "Target token count for the compressed summary.",
                    },
                    "from_commit": {
                        "type": "string",
                        "description": "Start of range (inclusive).",
                    },
                    "to_commit": {
                        "type": "string",
                        "description": "End of range (inclusive).",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Additional instructions for LLM-based compression.",
                    },
                },
            },
            handler=lambda target_tokens=None, from_commit=None, to_commit=None, instructions=None: _handle_compress(
                tract, target_tokens, from_commit, to_commit, instructions
            ),
        ),
        # 8. branch
        ToolDefinition(
            name="branch",
            description=(
                "Create a new branch from the current position or a specified "
                "commit. Use branches to explore alternative conversation paths "
                "or organize context by topic."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the new branch.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Commit hash to branch from (defaults to HEAD).",
                    },
                    "switch": {
                        "type": "boolean",
                        "description": "Whether to switch to the new branch (default true).",
                    },
                },
                "required": ["name"],
            },
            handler=lambda name, source=None, switch=True: _handle_branch(
                tract, name, source, switch
            ),
        ),
        # 9. switch
        ToolDefinition(
            name="switch",
            description=(
                "Switch to a different branch. Changes the active context to "
                "the target branch's history. Only accepts branch names (not "
                "commit hashes)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Name of the branch to switch to.",
                    },
                },
                "required": ["target"],
            },
            handler=lambda target: _handle_switch(tract, target),
        ),
        # 10. merge
        ToolDefinition(
            name="merge",
            description=(
                "Merge a branch into the current branch. Combines context from "
                "two branches. Requires an LLM client to be configured via "
                "tract.configure_llm() for semantic conflict resolution."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Name of the branch to merge into current.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional merge commit message.",
                    },
                },
                "required": ["source"],
            },
            handler=lambda source, message=None: _handle_merge(tract, source, message),
        ),
        # 11. reset
        ToolDefinition(
            name="reset",
            description=(
                "Reset HEAD to a previous commit. Use this to undo recent "
                "changes by moving the branch pointer backward. The original "
                "HEAD is saved as ORIG_HEAD for recovery."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Commit hash, branch name, or prefix to reset to.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["soft", "hard"],
                        "description": "Reset mode (both behave identically in tract). Default 'soft'.",
                    },
                },
                "required": ["target"],
            },
            handler=lambda target, mode="soft": _handle_reset(tract, target, mode),
        ),
        # 12. checkout
        ToolDefinition(
            name="checkout",
            description=(
                "Checkout a specific commit for read-only inspection. Puts HEAD "
                "in detached state. Use this to examine historical context without "
                "modifying the current branch."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Commit hash, branch name, prefix, or '-' for previous position.",
                    },
                },
                "required": ["target"],
            },
            handler=lambda target: _handle_checkout(tract, target),
        ),
        # 13. gc
        ToolDefinition(
            name="gc",
            description=(
                "Run garbage collection to remove unreachable commits. Frees "
                "storage by deleting orphaned commits that are no longer part "
                "of any branch."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "min_age_hours": {
                        "type": "integer",
                        "description": "Minimum age in hours before a commit is eligible for GC (converted to days internally). Default 168 (7 days).",
                    },
                    "keep_pinned": {
                        "type": "boolean",
                        "description": "Whether to keep pinned commits even if unreachable (default true).",
                    },
                    "keep_branches": {
                        "type": "boolean",
                        "description": "Whether to keep all branch tips reachable (default true).",
                    },
                },
            },
            handler=lambda min_age_hours=168, keep_pinned=True, keep_branches=True: _handle_gc(
                tract, min_age_hours, keep_pinned, keep_branches
            ),
        ),
        # 14. list_branches
        ToolDefinition(
            name="list_branches",
            description=(
                "List all branches in the tract with their HEAD commits. Shows "
                "which branch is currently active."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=lambda: _handle_list_branches(tract),
        ),
        # 15. get_commit
        ToolDefinition(
            name="get_commit",
            description=(
                "Get detailed information about a specific commit including "
                "its content type, operation, token count, and metadata."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "commit_hash": {
                        "type": "string",
                        "description": "Hash of the commit to retrieve.",
                    },
                },
                "required": ["commit_hash"],
            },
            handler=lambda commit_hash: _handle_get_commit(tract, commit_hash),
        ),
    ]


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------
# Separated from lambdas for readability. Each handler converts complex
# return values to human-readable strings.


def _handle_commit(
    tract: Tract,
    content: dict,
    operation: str,
    message: str | None,
    response_to: str | None,
    metadata: dict | None,
    generation_config: dict | None,
) -> str:
    from tract.models.commit import CommitOperation

    op = CommitOperation(operation)
    info = tract.commit(
        content,
        operation=op,
        message=message,
        response_to=response_to,
        metadata=metadata,
        generation_config=generation_config,
    )
    return (
        f"Committed {info.commit_hash[:8]} "
        f"({info.content_type}, {info.operation.value}, {info.token_count} tokens)"
    )


def _handle_compile(tract: Tract) -> str:
    result = tract.compile()
    return (
        f"Compiled context: {len(result.messages)} messages, "
        f"{result.token_count} tokens ({result.token_source})"
    )


def _handle_annotate(
    tract: Tract, target_hash: str, priority: str, reason: str | None
) -> str:
    from tract.models.annotations import Priority

    prio = Priority(priority)
    annotation = tract.annotate(target_hash, prio, reason=reason)
    return f"Annotated {target_hash[:8]} as {annotation.priority.value}"


def _handle_status(tract: Tract) -> str:
    status = tract.status()
    head_short = status.head_hash[:8] if status.head_hash else "None"
    branch = status.branch_name or "(detached)"
    budget_pct = ""
    if status.token_budget_max and status.token_budget_max > 0:
        pct = (status.token_count / status.token_budget_max) * 100
        budget_pct = f", budget {pct:.1f}%"
    return (
        f"Branch: {branch} | HEAD: {head_short} | "
        f"{status.commit_count} commits, {status.token_count} tokens{budget_pct}"
    )


def _handle_log(tract: Tract, limit: int, op_filter: str | None) -> str:
    from tract.models.commit import CommitOperation

    op = CommitOperation(op_filter) if op_filter else None
    entries = tract.log(limit=limit, op_filter=op)
    if not entries:
        return "No commits found."
    lines = []
    for entry in entries:
        msg = entry.message or ""
        lines.append(
            f"  {entry.commit_hash[:8]} {entry.operation.value:6s} "
            f"{entry.content_type:12s} {entry.token_count:5d}t  {msg}"
        )
    return f"Log ({len(entries)} commits):\n" + "\n".join(lines)


def _handle_diff(tract: Tract, commit_a: str | None, commit_b: str | None) -> str:
    result = tract.diff(commit_a=commit_a, commit_b=commit_b)
    return (
        f"Diff: {result.stat.messages_added} added, "
        f"{result.stat.messages_removed} removed, "
        f"{result.stat.messages_modified} modified | "
        f"Token delta: {result.stat.total_token_delta:+d}"
    )


def _handle_compress(
    tract: Tract,
    target_tokens: int | None,
    from_commit: str | None,
    to_commit: str | None,
    instructions: str | None,
) -> str:
    result = tract.compress(
        target_tokens=target_tokens,
        from_commit=from_commit,
        to_commit=to_commit,
        instructions=instructions,
        auto_commit=True,
    )
    # CompressResult has original_tokens, compressed_tokens, source_commits, summary_commits
    return (
        f"Compressed: {result.original_tokens} -> {result.compressed_tokens} tokens "
        f"({len(result.summary_commits)} summaries, {len(result.source_commits)} source commits)"
    )


def _handle_branch(
    tract: Tract, name: str, source: str | None, switch: bool
) -> str:
    commit_hash = tract.branch(name, source=source, switch=switch)
    action = "Created and switched to" if switch else "Created"
    return f"{action} branch '{name}' at {commit_hash[:8]}"


def _handle_switch(tract: Tract, target: str) -> str:
    commit_hash = tract.switch(target)
    return f"Switched to branch '{target}' at {commit_hash[:8]}"


def _handle_merge(tract: Tract, source: str, message: str | None) -> str:
    # Use auto_commit for simplicity in tool context
    result = tract.merge(source, auto_commit=True)
    if result.merge_type == "fast_forward":
        return f"Fast-forward merge of '{source}' into {result.target_branch}"
    return (
        f"Merged '{source}' into {result.target_branch} "
        f"({result.merge_type}, {len(result.conflicts)} conflicts)"
    )


def _handle_reset(tract: Tract, target: str, mode: str) -> str:
    resolved = tract.reset(target, mode=mode)
    return f"Reset HEAD to {resolved[:8]} (mode={mode})"


def _handle_checkout(tract: Tract, target: str) -> str:
    commit_hash = tract.checkout(target)
    return f"Checked out {commit_hash[:8]}"


def _handle_gc(
    tract: Tract,
    min_age_hours: int,
    keep_pinned: bool,
    keep_branches: bool,
) -> str:
    # Convert hours to days for the gc API
    retention_days = max(1, min_age_hours // 24)
    result = tract.gc(orphan_retention_days=retention_days)
    return (
        f"GC complete: {result.commits_removed} commits removed, "
        f"{result.blobs_removed} blobs removed"
    )


def _handle_list_branches(tract: Tract) -> str:
    branches = tract.list_branches()
    if not branches:
        return "No branches found."
    lines = []
    for b in branches:
        marker = "* " if b.is_current else "  "
        lines.append(f"{marker}{b.name} ({b.commit_hash[:8]})")
    return "Branches:\n" + "\n".join(lines)


def _handle_get_commit(tract: Tract, commit_hash: str) -> str:
    info = tract.get_commit(commit_hash)
    if info is None:
        return f"Commit {commit_hash} not found."
    meta = json.dumps(info.metadata) if info.metadata else "none"
    gen_cfg = json.dumps(info.generation_config) if info.generation_config else "none"
    return (
        f"Commit {info.commit_hash[:8]}: "
        f"type={info.content_type}, op={info.operation.value}, "
        f"tokens={info.token_count}, message={info.message or 'none'}, "
        f"metadata={meta}, generation_config={gen_cfg}"
    )
