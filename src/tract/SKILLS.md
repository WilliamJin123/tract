# Tract — Context Version Control

Tract gives you git-like version control over your own context window. Every piece of content you commit is stored in a DAG; you can branch, merge, compress, tag, and compile your history into LLM-ready messages. Use it to organize long conversations, isolate topics on branches, and stay within token budgets.

## Core Workflow

**commit** content, **compile** to get messages, check **status** for token usage, **branch** to isolate topics, **switch** between branches.

## Content Types

Every commit requires a `content` dict with a `content_type` field. The seven types an agent will use:

| content_type | Required fields | Use for |
|---|---|---|
| `dialogue` | `role` (user/assistant/system/tool), `text` | Conversation turns |
| `instruction` | `text`, optional `name` | Standing instructions (name deduplicates) |
| `tool_io` | `tool_name`, `direction` (call/result), `payload` (dict) | Tool calls and results |
| `reasoning` | `text` | Chain-of-thought |
| `artifact` | `artifact_type`, `content`, optional `language` | Code, documents, configs |
| `output` | `text` | Final deliverables |
| `freeform` | `payload` (dict) | Anything else |

Content is always a dict, never a bare string.

## Operations Reference

**Context**: `commit(content, operation, message, edit_target, metadata, tags)` — operation is `"append"` (default) or `"edit"` (must set `edit_target` to a commit hash). `compile()` — returns token/message counts. `status()` — branch, HEAD, tokens, budget. `log(limit, op_filter)` — history. `diff(commit_a, commit_b)` — compare commits. `get_commit(commit_hash)` — details. `compress(target_tokens, from_commit, to_commit, instructions, content, preserve)` — reduce tokens; pinned commits survive.

**Branching**: `branch(name, source, switch)` — create branch. `switch(target)` — change branch. `merge(source, message)` — merge into current. `reset(target, mode)` — mode is `"soft"` (default). `checkout(target)` — read-only detached HEAD. `list_branches()`. `transition(target, handoff)` — switch with hooks; handoff is `"full"`, `"summary"`, `"none"`, or custom text.

**Annotations**: `annotate(target_hash, priority, reason)` — priority is `"pinned"` (survives compression), `"normal"`, or `"skip"` (excluded from compile). `directive(name, text, priority)` — named standing instruction, deduplicated by name, default pinned.

**Tags**: `tag(commit_hash, tag)`, `untag(commit_hash, tag)`, `query_by_tags(tags, match)` — match is `"any"` or `"all"`. `register_tag(name, description)`, `get_tags(commit_hash)`, `list_tags()`.

**Config**: `configure(settings)` — keys: model, temperature, max_tokens, max_commit_tokens, auto_compress_threshold, compile_strategy, compile_strategy_k. `configure_model(model, operation, temperature)`. `get_config(key)`. `create_metadata(kind, data, path)`.

**Middleware**: `create_middleware(event, code, description)` — events: pre_commit, post_commit, pre_compile, pre_compress, pre_merge, pre_gc, pre_transition, post_transition. Code must define `handler(ctx)`. `remove_middleware(handler_id)`.

## Common Patterns

- **New topic**: `branch("topic-name")` then `commit(...)` — isolates context.
- **Running low on tokens**: `compress(target_tokens=4000)` — pinned content survives.
- **Persistent instruction**: `directive("style", "Always respond in bullet points")` — survives branches, deduplicated by name.
- **Mark content for exclusion**: `annotate(hash, "skip")` — invisible to compile.
- **Protect important content**: `annotate(hash, "pinned")` — survives compression.
- **Edit a previous commit**: `commit(content, operation="edit", edit_target=hash)`.

## Key Gotchas

- `operation` values: `"append"` / `"edit"` — these are commit operations.
- `priority` values: `"pinned"` / `"normal"` / `"skip"` — these are annotations. Do not confuse with operations.
- `content` must be a dict with `content_type`, not a string.
- `tool_io` uses `payload` (dict), not `text`.
- `freeform` uses `payload` (dict), not `text`.
