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
        List of 25+ ToolDefinition objects.
    """
    tools = [
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
                    "edit_target": {
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
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags to attach to the commit.",
                    },
                },
                "required": ["content"],
            },
            handler=lambda content, operation="append", message=None, edit_target=None, metadata=None, generation_config=None, tags=None: _handle_commit(
                tract, content, operation, message, edit_target, metadata, generation_config, tags
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
                    "min_age_days": {
                        "type": "integer",
                        "description": "Minimum age in days before a commit is eligible for GC. Default 7.",
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
            handler=lambda min_age_days=7, keep_pinned=True, keep_branches=True: _handle_gc(
                tract, min_age_days, keep_pinned, keep_branches
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
        # 16. configure_model
        ToolDefinition(
            name="configure_model",
            description=(
                "Change LLM model or temperature for a specific operation or "
                "tract-wide. Use this to switch models mid-conversation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g. 'gpt-4o', 'gpt-3.5-turbo').",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["chat", "merge", "compress", "orchestrate", "summarize"],
                        "description": "Operation to configure. Omit to set tract-wide default.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature override.",
                    },
                },
            },
            handler=lambda model=None, operation=None, temperature=None: _handle_configure_model(
                tract, model, operation, temperature
            ),
        ),
        # 17. tag
        ToolDefinition(
            name="tag",
            description=(
                "Add a mutable tag annotation to a commit. Tags can be added "
                "or removed after commit creation, unlike immutable commit tags."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "commit_hash": {
                        "type": "string",
                        "description": "Hash of the commit to tag.",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Tag name to add.",
                    },
                },
                "required": ["commit_hash", "tag"],
            },
            handler=lambda commit_hash, tag: _handle_tag(tract, commit_hash, tag),
        ),
        # 18. untag
        ToolDefinition(
            name="untag",
            description=(
                "Remove a mutable tag annotation from a commit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "commit_hash": {
                        "type": "string",
                        "description": "Hash of the commit to untag.",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Tag name to remove.",
                    },
                },
                "required": ["commit_hash", "tag"],
            },
            handler=lambda commit_hash, tag: _handle_untag(tract, commit_hash, tag),
        ),
        # 19. query_by_tags
        ToolDefinition(
            name="query_by_tags",
            description=(
                "Find commits that have specific tags. Returns commit hashes "
                "matching the given tag criteria."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to search for.",
                    },
                    "match": {
                        "type": "string",
                        "enum": ["any", "all"],
                        "description": "Match mode: 'any' (OR) or 'all' (AND). Default 'any'.",
                    },
                },
                "required": ["tags"],
            },
            handler=lambda tags, match="any": _handle_query_by_tags(tract, tags, match),
        ),
        # 20. register_tag
        ToolDefinition(
            name="register_tag",
            description=(
                "Register a custom tag name in the tag registry. Required before "
                "using the tag in strict mode (the default). Optionally provide a "
                "description explaining what the tag means."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Tag name to register.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of what this tag means.",
                    },
                },
                "required": ["name"],
            },
            handler=lambda name, description=None: _handle_register_tag(
                tract, name, description
            ),
        ),
        # 21. get_tags
        ToolDefinition(
            name="get_tags",
            description=(
                "Get all tags on a commit (both immutable auto-classified tags "
                "and mutable annotation tags, merged and deduplicated)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "commit_hash": {
                        "type": "string",
                        "description": "Hash of the commit to get tags for.",
                    },
                },
                "required": ["commit_hash"],
            },
            handler=lambda commit_hash: _handle_get_tags(tract, commit_hash),
        ),
        # 22. list_tags
        ToolDefinition(
            name="list_tags",
            description=(
                "List all registered tags with descriptions and usage counts. "
                "Shows both auto-created base tags and custom-registered tags."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=lambda: _handle_list_tags(tract),
        ),
        # 23. register_trigger
        ToolDefinition(
            name="register_trigger",
            description=(
                "Register a built-in trigger by type name and configuration. "
                "Available types: compress, pin, branch, merge, rebase, gc, archive."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "trigger_type": {
                        "type": "string",
                        "enum": ["compress", "pin", "branch", "merge", "rebase", "gc", "archive"],
                        "description": "Type of built-in trigger to register.",
                    },
                    "config": {
                        "type": "object",
                        "description": "Configuration dict passed to the trigger constructor.",
                    },
                },
                "required": ["trigger_type"],
            },
            handler=lambda trigger_type, config=None: _handle_register_trigger(
                tract, trigger_type, config
            ),
        ),
        # 21. unregister_trigger
        ToolDefinition(
            name="unregister_trigger",
            description=(
                "Remove a registered trigger by name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "trigger_name": {
                        "type": "string",
                        "description": "Name of the trigger to remove.",
                    },
                },
                "required": ["trigger_name"],
            },
            handler=lambda trigger_name: _handle_unregister_trigger(tract, trigger_name),
        ),
        # 22. toggle_triggers
        ToolDefinition(
            name="toggle_triggers",
            description=(
                "Pause or resume all trigger evaluation. Use to temporarily "
                "disable triggers during bulk operations."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to resume triggers, False to pause.",
                    },
                },
                "required": ["enabled"],
            },
            handler=lambda enabled: _handle_toggle_triggers(tract, enabled),
        ),
    ]

    # Append tools for dynamic operations
    for op_name in sorted(tract._operation_registry.operation_names):
        spec = tract._operation_registry.get_spec(op_name)
        if spec is not None:
            tools.append(ToolDefinition(
                name=f"fire_{op_name}",
                description=spec.description,
                parameters=_spec_fields_to_json_schema(spec.fields),
                handler=lambda fields=None, _name=op_name: tract.fire(_name, fields=fields),
            ))

    return tools


def _spec_fields_to_json_schema(field_specs: dict) -> dict:
    """Convert dynamic operation field specs to JSON Schema format."""
    _type_to_schema: dict[str, str] = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "list[str]": "array",
        "list[int]": "array",
        "dict[str, str]": "object",
    }

    properties: dict = {}
    for fname, fdef in field_specs.items():
        ftype = fdef.get("type", "string")
        schema_type = _type_to_schema.get(ftype, "string")
        prop: dict = {"type": schema_type}
        if "description" in fdef:
            prop["description"] = fdef["description"]
        properties[fname] = prop

    return {
        "type": "object",
        "properties": properties,
    }


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
    edit_target: str | None,
    metadata: dict | None,
    generation_config: dict | None,
    tags: list[str] | None = None,
) -> str:
    from tract.models.commit import CommitOperation

    op = CommitOperation(operation)
    info = tract.commit(
        content,
        operation=op,
        message=message,
        edit_target=edit_target,
        metadata=metadata,
        generation_config=generation_config,
        tags=tags,
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
    result = tract.merge(source, auto_commit=True, message=message)
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
    min_age_days: int,
    keep_pinned: bool,
    keep_branches: bool,
) -> str:
    result = tract.gc(orphan_retention_days=max(1, min_age_days))
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
    gen_cfg = json.dumps(info.generation_config.to_dict()) if info.generation_config else "none"
    tags = tract.get_tags(info.commit_hash)
    tags_str = ", ".join(tags) if tags else "none"
    return (
        f"Commit {info.commit_hash[:8]}: "
        f"type={info.content_type}, op={info.operation.value}, "
        f"tokens={info.token_count}, message={info.message or 'none'}, "
        f"tags={tags_str}, "
        f"metadata={meta}, generation_config={gen_cfg}"
    )


def _handle_configure_model(
    tract: Tract,
    model: str | None,
    operation: str | None,
    temperature: float | None,
) -> str:
    from tract.models.config import LLMConfig

    if operation is not None:
        config = LLMConfig(model=model, temperature=temperature)
        tract.configure_operations(**{operation: config})
        parts = []
        if model:
            parts.append(f"model={model}")
        if temperature is not None:
            parts.append(f"temperature={temperature}")
        return f"Configured {operation}: {', '.join(parts)}"
    else:
        # Set tract-wide default
        config = LLMConfig(model=model, temperature=temperature)
        tract._default_config = config
        parts = []
        if model:
            parts.append(f"model={model}")
        if temperature is not None:
            parts.append(f"temperature={temperature}")
        return f"Set tract-wide default: {', '.join(parts)}"


def _handle_register_tag(
    tract: Tract, name: str, description: str | None
) -> str:
    tract.register_tag(name, description)
    desc = f" ({description})" if description else ""
    return f"Registered tag '{name}'{desc}"


def _handle_get_tags(tract: Tract, commit_hash: str) -> str:
    tags = tract.get_tags(commit_hash)
    if not tags:
        return f"No tags on {commit_hash[:8]}"
    return f"Tags on {commit_hash[:8]}: {', '.join(tags)}"


def _handle_list_tags(tract: Tract) -> str:
    entries = tract.list_tags()
    if not entries:
        return "No tags registered."
    lines = []
    for entry in entries:
        kind = "auto" if entry["auto_created"] else "custom"
        desc = entry["description"] or ""
        lines.append(
            f"  {entry['name']:20s} count={entry['count']}  ({kind})  {desc}"
        )
    return f"Tags ({len(entries)} registered):\n" + "\n".join(lines)


def _handle_tag(tract: Tract, commit_hash: str, tag: str) -> str:
    tract.tag(commit_hash, tag)
    return f"Tagged {commit_hash[:8]} with '{tag}'"


def _handle_untag(tract: Tract, commit_hash: str, tag: str) -> str:
    removed = tract.untag(commit_hash, tag)
    if removed:
        return f"Removed tag '{tag}' from {commit_hash[:8]}"
    return f"Tag '{tag}' not found on {commit_hash[:8]}"


def _handle_query_by_tags(tract: Tract, tags: list[str], match: str) -> str:
    results = tract.query_by_tags(tags, match=match)
    if not results:
        return f"No commits found with tags {tags} (match={match})"
    short = [r.commit_hash[:8] for r in results]
    return f"Found {len(results)} commits: {', '.join(short)}"


def _handle_register_trigger(
    tract: Tract, trigger_type: str, config: dict | None,
) -> str:
    from tract.triggers.builtin import (
        CompressTrigger, PinTrigger, BranchTrigger,
        MergeTrigger, RebaseTrigger, GCTrigger, ArchiveTrigger,
    )

    _TRIGGER_MAP = {
        "compress": CompressTrigger,
        "pin": PinTrigger,
        "branch": BranchTrigger,
        "merge": MergeTrigger,
        "rebase": RebaseTrigger,
        "gc": GCTrigger,
        "archive": ArchiveTrigger,
    }

    cls = _TRIGGER_MAP.get(trigger_type)
    if cls is None:
        return f"Unknown trigger type: {trigger_type}"

    try:
        trigger = cls(**(config or {}))
    except Exception as exc:
        return f"Failed to create {trigger_type} trigger: {exc}"

    tract.register_trigger(trigger)
    return f"Registered {trigger_type} trigger as '{trigger.name}'"


def _handle_unregister_trigger(tract: Tract, trigger_name: str) -> str:
    tract.unregister_trigger(trigger_name)
    return f"Unregistered trigger '{trigger_name}'"


def _handle_toggle_triggers(tract: Tract, enabled: bool) -> str:
    if enabled:
        tract.resume_all_triggers()
        return "Triggers resumed"
    else:
        tract.pause_all_triggers()
        return "Triggers paused"
