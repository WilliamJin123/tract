"""Hand-crafted tool definitions for all Tract operations.

Each tool definition includes a clear, action-oriented description,
proper JSON Schema parameters, and a handler lambda bound to a specific
Tract instance.  Handler lambdas use explicit parameter whitelisting
(no ``**kwargs`` passthrough) to prevent hallucinated arguments.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tract.middleware import VALID_EVENTS
from tract.toolkit.models import ToolDefinition

if TYPE_CHECKING:
    from tract.tract import Tract


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
                "Save content to your context history. Use after generating a "
                "response, receiving input, or completing a reasoning step. "
                "Content dict requires a 'content_type' field. "
                "Tags are auto-registered if needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "description": (
                            "Content dict. Required field: content_type. "
                            "Fields per type:\n"
                            "- dialogue: role ('user'|'assistant'|'system'|'tool'), text (str)\n"
                            "- instruction: text (str), optional name (str, for dedup)\n"
                            "- tool_io: tool_name (str), direction ('call'|'result'), payload (dict), optional status ('success'|'error')\n"
                            "- reasoning: text (str), optional format ('parsed'|'raw'|'think_tags'|'anthropic')\n"
                            "- artifact: artifact_type (str), content (str), optional language (str)\n"
                            "- output: text (str), optional format ('text'|'markdown'|'json')\n"
                            "- freeform: payload (dict)"
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
            handler=lambda content=None, operation="append", message=None, edit_target=None, metadata=None, generation_config=None, tags=None, **extra: _handle_commit(
                tract, content, operation, message, edit_target, metadata, generation_config, tags, extra
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
                "Set priority on a commit. Use 'pinned' to protect content "
                "you'll need later (prevents compression), 'skip' to hide "
                "resolved/irrelevant content from compilation. Check log "
                "first — instruction/system commits are already pinned."
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
                "Check current state: branch, HEAD, token count, budget usage. "
                "Use before deciding whether to compress, branch, or commit."
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
                "Summarize older context to free up token budget. Use when "
                "status shows budget above 70% or when context feels bloated "
                "with resolved details."
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
                    "content": {
                        "type": "string",
                        "description": (
                            "Manual summary text. When provided, bypasses the LLM "
                            "and uses this text as the compression summary."
                        ),
                    },
                    "preserve": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of commit hashes to preserve verbatim "
                            "(treated as PINNED during compression)."
                        ),
                    },
                },
            },
            handler=lambda target_tokens=None, from_commit=None, to_commit=None, instructions=None, content=None, preserve=None: _handle_compress(
                tract, target_tokens, from_commit, to_commit, instructions, content, preserve
            ),
        ),
        # 8. branch
        ToolDefinition(
            name="branch",
            description=(
                "Create a new branch to work on something independently. Use "
                "BEFORE starting a second option, alternative approach, or "
                "tangent — anything that should not influence other work. "
                "Branch first, then commit to the branch."
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
                "Switch to an existing branch. Use to resume work on another "
                "line of thought or return to main. For stage-based workflow "
                "transitions (where you want context handoff), use transition instead."
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
                "Combine a branch into your current branch. Use after finishing "
                "independent work on a branch and wanting to bring results "
                "back together."
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
                        "enum": ["chat", "merge", "compress", "message"],
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
                "Pre-register a tag with an optional description. Not required for "
                "commit (tags are auto-registered there), but useful to attach a "
                "human-readable description to a tag before it is used."
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
        # 23. configure
        ToolDefinition(
            name="configure",
            description=(
                "Set config key-value pairs on the DAG. Well-known keys: "
                "model, temperature, max_tokens, max_commit_tokens, "
                "auto_compress_threshold, compact_tools, compile_strategy, "
                "compile_strategy_k, handoff_summary_k. Unknown keys pass through."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "description": "Key-value config settings (e.g. {model: 'gpt-4o', temperature: 0.7}).",
                    },
                },
                "required": ["settings"],
            },
            handler=lambda settings: _handle_configure(tract, settings),
        ),
        # 24. create_metadata
        ToolDefinition(
            name="create_metadata",
            description=(
                "Create or update a metadata entry. Metadata stores structured "
                "data (file trees, project plans, configs) alongside the context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "description": "Freeform label (e.g. 'file_tree', 'project_plan').",
                    },
                    "data": {
                        "type": "object",
                        "description": "Structured content dict.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional filesystem path for export/sync.",
                    },
                },
                "required": ["kind", "data"],
            },
            handler=lambda kind, data, path=None: _handle_create_metadata(
                tract, kind, data, path
            ),
        ),
        # 25. get_config
        ToolDefinition(
            name="get_config",
            description=(
                "Resolve a config value from the DAG. Uses DAG precedence "
                "(closest config to HEAD wins). Returns the value or null."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Config key to resolve.",
                    },
                },
                "required": ["key"],
            },
            handler=lambda key: _handle_get_config(tract, key),
        ),
        # 26. transition
        ToolDefinition(
            name="transition",
            description=(
                "Move to the next stage in a workflow (e.g., research -> drafting -> review). "
                "Creates the target branch if it doesn't exist, switches to it, and optionally "
                "commits a context handoff so the new stage has relevant prior context. "
                "Use this instead of switch when moving between workflow stages. "
                "Runs pre/post_transition middleware."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Target branch name (stage) to transition to. Created if it doesn't exist.",
                    },
                    "handoff": {
                        "type": "string",
                        "description": (
                            "What context to carry to the new stage. "
                            "'none' (default): switch only, no context carried. "
                            "'summary': compile an adaptive summary of current context. "
                            "'full': compile entire current context verbatim. "
                            "Or pass a custom string to use as the handoff text."
                        ),
                        "default": "none",
                    },
                },
                "required": ["target"],
            },
            handler=lambda target, handoff="none": _handle_transition(
                tract, target, handoff
            ),
        ),
        # 27. directive
        ToolDefinition(
            name="directive",
            description=(
                "Create a named standing instruction that persists in the LLM context. "
                "Directives are deduplicated by name: if you create two directives with "
                "the same name, only the latest one appears in compiled context. "
                "Use this to set behavioral rules like tone, format, or safety guidelines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique directive name (e.g. 'tone', 'format', 'safety'). Same name = override.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The instruction text that will appear in the LLM context.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["pinned", "normal", "skip"],
                        "description": "Priority level. Default 'pinned' (survives compression).",
                    },
                },
                "required": ["name", "text"],
            },
            handler=lambda name, text, priority=None: _handle_directive(
                tract, name, text, priority
            ),
        ),
        # 28. create_middleware
        ToolDefinition(
            name="create_middleware",
            description=(
                "Create a middleware handler from Python code. The code must define a "
                "function called `handler(ctx)` that receives a MiddlewareContext with "
                "attributes: event, commit, tract, branch, head, target, pending. "
                "To block an operation, raise BlockedError(event, reason). "
                "Available in code: BlockedError, re, json, len, str, int, float, "
                "bool, list, dict, set, tuple, range, enumerate, zip, sorted, "
                "min, max, sum, any, all, isinstance, hasattr, getattr, print. "
                "Example: 'def handler(ctx):\\n    if len(ctx.commit.message or \"\") > 500:\\n"
                "        raise BlockedError(\"pre_commit\", \"Message too long\")'"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "event": {
                        "type": "string",
                        "enum": sorted(VALID_EVENTS),
                        "description": "Event to hook into.",
                    },
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code defining a `handler(ctx)` function. "
                            "Raise BlockedError(event, reasons) to block."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what this middleware does.",
                    },
                },
                "required": ["event", "code"],
            },
            handler=lambda event, code, description="": _handle_create_middleware(
                tract, event, code, description
            ),
        ),
        # 29. remove_middleware
        ToolDefinition(
            name="remove_middleware",
            description=(
                "Remove a previously created middleware handler by its ID. "
                "Use the handler_id returned by create_middleware."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "handler_id": {
                        "type": "string",
                        "description": "The handler ID returned by create_middleware.",
                    },
                },
                "required": ["handler_id"],
            },
            handler=lambda handler_id: _handle_remove_middleware(tract, handler_id),
        ),
    ]

    return tools


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------
# Separated from lambdas for readability. Each handler converts complex
# return values to human-readable strings.


def _parse_str_to_obj(value: str) -> Any:
    """Try json.loads, then ast.literal_eval, to parse LLM-produced strings."""
    import ast

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        result = ast.literal_eval(value)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass
    return None  # could not parse


def _handle_commit(
    tract: Tract,
    content: dict | str | None,
    operation: str,
    message: str | None,
    edit_target: str | None,
    metadata: dict | None,
    generation_config: dict | None,
    tags: list[str] | None = None,
    extra: dict | None = None,
) -> str:
    from tract.models.commit import CommitOperation

    # LLMs sometimes pass content fields as flat top-level args instead of nesting
    # in a content dict (e.g. content_type="artifact", text="..." instead of
    # content={"content_type": "artifact", "text": "..."}).  Reconstruct.
    _CONTENT_KEYS = {"content_type", "text", "role", "payload", "artifact_type",
                     "tool_name", "direction", "status", "format", "language",
                     "name", "content"}
    if content is None and extra:
        flat = {k: v for k, v in extra.items() if k in _CONTENT_KEYS}
        if "content_type" in flat:
            # "content" inside a flat arg set means the text body of an artifact
            if "content" in flat and flat.get("content_type") == "artifact":
                flat.setdefault("text", flat.pop("content"))
            content = flat

    if content is None:
        raise ValueError(
            "Missing 'content' parameter. Pass a dict with at least "
            "'content_type' and the relevant fields (text, role, payload, etc.)."
        )

    # LLMs sometimes pass content as a JSON string instead of a dict — parse it.
    # Small models (e.g. llama-3.1-8b) may use Python repr (single quotes) instead
    # of valid JSON, so we also try ast.literal_eval.
    if isinstance(content, str):
        parsed = _parse_str_to_obj(content)
        if isinstance(parsed, dict):
            content = parsed
        else:
            # Treat plain text as a dialogue assistant message
            content = {"content_type": "dialogue", "role": "assistant", "text": content}

    # LLMs may also pass metadata/generation_config/tags as stringified JSON/repr
    if isinstance(metadata, str):
        parsed = _parse_str_to_obj(metadata)
        metadata = parsed if isinstance(parsed, dict) else None

    if isinstance(generation_config, str):
        parsed = _parse_str_to_obj(generation_config)
        generation_config = parsed if isinstance(parsed, dict) else None

    if isinstance(tags, str):
        parsed = _parse_str_to_obj(tags)
        if isinstance(parsed, list):
            tags = [str(t) for t in parsed]
        else:
            # Single tag name as a string
            tags = [tags] if tags.strip() else None

    # Normalize empty strings to None for optional fields
    if not message:
        message = None
    if not edit_target:
        edit_target = None

    # Resolve short hash prefixes to full hashes (LLMs use 8-char prefixes)
    if edit_target:
        edit_target = tract.resolve_commit(edit_target)

    # Auto-register unknown tags so commit never fails on unregistered tags
    if tags:
        for tag_name in tags:
            if (
                tract._tag_registry_repo is not None
                and not tract._tag_registry_repo.is_registered(
                    tract._tract_id, tag_name
                )
            ):
                tract.register_tag(tag_name)

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

    resolved = tract.resolve_commit(target_hash)
    prio = Priority(priority)
    annotation = tract.annotate(resolved, prio, reason=reason)
    return f"Annotated {resolved[:8]} as {annotation.priority.value}"


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
        pri = entry.effective_priority or "normal"
        pri_tag = f" [{pri}]" if pri != "normal" else ""
        lines.append(
            f"  {entry.commit_hash[:8]} {entry.operation.value:6s} "
            f"{entry.content_type:12s} {entry.token_count:5d}t{pri_tag}  {msg}"
        )
    return f"Log ({len(entries)} commits):\n" + "\n".join(lines)


def _handle_diff(tract: Tract, commit_a: str | None, commit_b: str | None) -> str:
    if commit_a is not None:
        commit_a = tract.resolve_commit(commit_a)
    if commit_b is not None:
        commit_b = tract.resolve_commit(commit_b)
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
    content: str | None = None,
    preserve: list[str] | None = None,
) -> str:
    if from_commit is not None:
        from_commit = tract.resolve_commit(from_commit)
    if to_commit is not None:
        to_commit = tract.resolve_commit(to_commit)
    if preserve is not None:
        preserve = [tract.resolve_commit(h) for h in preserve]
    result = tract.compress(
        target_tokens=target_tokens,
        from_commit=from_commit,
        to_commit=to_commit,
        instructions=instructions,
        content=content,
        preserve=preserve,
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
    commit_hash = tract.resolve_commit(commit_hash)
    info = tract.get_commit(commit_hash)
    if info is None:
        return f"Commit {commit_hash} not found."
    # Enrich with effective priority
    enriched = tract._enrich_with_priorities([info])
    info = enriched[0] if enriched else info
    priority = info.effective_priority or "normal"
    meta = json.dumps(info.metadata) if info.metadata else "none"
    gen_cfg = json.dumps(info.generation_config.to_dict()) if info.generation_config else "none"
    tags = tract.get_tags(info.commit_hash)
    tags_str = ", ".join(tags) if tags else "none"
    return (
        f"Commit {info.commit_hash[:8]}: "
        f"type={info.content_type}, op={info.operation.value}, "
        f"priority={priority}, "
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
    import dataclasses

    from tract.models.config import LLMConfig

    # Build overlay with only the fields the caller provided
    overlay = LLMConfig(model=model, temperature=temperature)
    non_none = overlay.non_none_fields()

    if operation is not None:
        # Merge with existing operation config (if any)
        existing = getattr(tract.operation_configs, operation, None)
        if existing is not None and non_none:
            merged = dataclasses.replace(existing, **non_none)
        else:
            merged = overlay
        tract.configure_operations(**{operation: merged})
        parts = []
        if model:
            parts.append(f"model={model}")
        if temperature is not None:
            parts.append(f"temperature={temperature}")
        return f"Configured {operation}: {', '.join(parts)}"
    else:
        # Merge with existing tract-wide default (preserves fields not overridden)
        existing = tract.default_config
        if existing is not None and non_none:
            merged = dataclasses.replace(existing, **non_none)
        else:
            merged = overlay
        tract._llm_state.default_config = merged
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
    commit_hash = tract.resolve_commit(commit_hash)
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
    commit_hash = tract.resolve_commit(commit_hash)
    tract.tag(commit_hash, tag)
    return f"Tagged {commit_hash[:8]} with '{tag}'"


def _handle_untag(tract: Tract, commit_hash: str, tag: str) -> str:
    commit_hash = tract.resolve_commit(commit_hash)
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


def _handle_configure(tract: Tract, settings: dict) -> str:
    info = tract.configure(**settings)
    keys = ", ".join(settings.keys())
    return f"Configured {keys} ({info.commit_hash[:8]})"


def _handle_create_metadata(
    tract: Tract,
    kind: str,
    data: dict,
    path: str | None,
) -> str:
    info = tract.metadata(kind, data, path=path)
    return f"Created metadata '{kind}' ({info.commit_hash[:8]})"


def _handle_get_config(tract: Tract, key: str) -> str:
    value = tract.get_config(key)
    if value is None:
        return f"Config '{key}': not set"
    return f"Config '{key}': {value}"


def _handle_transition(tract: Tract, target: str, handoff: str = "none") -> str:
    result = tract.transition(target, handoff=handoff)
    if result is None:
        return f"Transitioned to '{target}' (no handoff)"
    return f"Transitioned to '{target}' ({result.commit_hash[:8]})"


def _handle_directive(
    tract: Tract, name: str, text: str, priority: str | None
) -> str:
    from tract.models.annotations import Priority

    prio = Priority(priority) if priority else None
    info = tract.directive(name, text, priority=prio)
    prio_label = priority or "pinned"
    return f"Directive '{name}' set ({info.commit_hash[:8]}, priority={prio_label})"


def _handle_create_middleware(
    tract: Tract, event: str, code: str, description: str = ""
) -> str:
    """Compile LLM-generated Python into a middleware handler.

    Uses compile() + restricted globals for safety:
    - Only safe builtins (no __import__, exec, eval, open, etc.)
    - BlockedError, re, json available
    - Code must define handler(ctx)
    """
    import re as _re
    import json as _json

    from tract.exceptions import BlockedError as _BlockedError

    # Safe builtins whitelist (inspired by smolagents LocalPythonExecutor)
    _SAFE_BUILTINS = {
        "len": len, "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "range": range, "enumerate": enumerate, "zip": zip,
        "sorted": sorted, "min": min, "max": max, "sum": sum,
        "any": any, "all": all, "isinstance": isinstance,
        "hasattr": hasattr, "getattr": getattr, "print": print,
        "True": True, "False": False, "None": None,
        "ValueError": ValueError, "TypeError": TypeError,
        "KeyError": KeyError, "RuntimeError": RuntimeError,
    }

    restricted_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "BlockedError": _BlockedError,
        "re": _re,
        "json": _json,
    }

    try:
        compiled = compile(code, "<middleware>", "exec")
    except SyntaxError as exc:
        return f"ERROR: Syntax error in middleware code: {exc}"

    namespace: dict = {}
    try:
        exec(compiled, restricted_globals, namespace)  # noqa: S102
    except Exception as exc:
        return f"ERROR: Failed to execute middleware code: {exc}"

    handler_fn = namespace.get("handler")
    if handler_fn is None:
        return "ERROR: Code must define a `handler(ctx)` function"
    if not callable(handler_fn):
        return "ERROR: `handler` must be callable"

    handler_id = tract.use(event, handler_fn)
    desc = f" ({description})" if description else ""
    return f"Middleware registered on '{event}': {handler_id}{desc}"


def _handle_remove_middleware(tract: Tract, handler_id: str) -> str:
    try:
        tract.remove_middleware(handler_id)
        return f"Middleware {handler_id} removed"
    except ValueError as exc:
        return f"ERROR: {exc}"


