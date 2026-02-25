# Tool Result Management API

The key utilities for working with tool call results in tract. Three layers:
primitives for manual control, query/selection for targeting, and
configuration for declarative per-tool behavior.

---

## Layer 1: Primitives

### `tool_result(edit=)` -- Edit a tool result in place

The atomic operation. Replace a verbose tool result with a leaner version.
Original preserved in history via the edit system (full provenance).

```python
# Commit verbose tool result
ci = t.tool_result("call_1", "grep", verbose_200_lines)

# Agent processes result, knows what mattered
t.tool_result("call_1", "grep", "config.py:42: DATABASE_URL=...", edit=ci.commit_hash)
```

The edit preserves:
- `role="tool"` (stays a tool message)
- `tool_call_id` linkage (still points to the same assistant tool call)
- `name` metadata (still "grep")
- Original content (in commit history, queryable via `t.log()`)

### `compress()` with `TOOL_SUMMARIZE_SYSTEM` -- Batch compression

For collapsing entire tool sequences after the fact. Uses the tool-aware
system prompt instead of the generic conversation summarizer.

```python
t.compress(
    commits=tool_hashes,
    system_prompt=TOOL_SUMMARIZE_SYSTEM,
    instructions="preserve the final config values found",
    preserve=[answer_hash],
)
```

### `compress_tool_calls()` -- Convenience wrapper

Auto-detects the final answer in a commit list and preserves it.
Defaults to `TOOL_SUMMARIZE_SYSTEM`.

```python
all_hashes = intermediate_hashes + [answer_hash]
t.compress_tool_calls(all_hashes, target_tokens=100)
```

---

## Layer 2: Query / Selection

Tools for finding and targeting tool-related commits. These return commit
hashes that feed into `edit=`, `compress(commits=)`, or any other operation.

### `find_tool_results(name=, after=, before=)` -- Find tool result commits

```python
# All tool results
results = t.find_tool_results()
# [CommitInfo(role=tool, name=grep, ...), CommitInfo(role=tool, name=read_file, ...), ...]

# Just grep results
grep_results = t.find_tool_results(name="grep")

# Grep results after a certain point
recent = t.find_tool_results(name="grep", after=some_hash)
```

Returns `list[CommitInfo]` for tool result commits matching the filter.
Detection: `metadata.tool_call_id` exists.

### `find_tool_calls(name=)` -- Find assistant tool-call commits

```python
# All assistant messages that requested tool calls
calls = t.find_tool_calls()

# Just the ones that called grep
grep_calls = t.find_tool_calls(name="grep")
```

Returns `list[CommitInfo]` for assistant commits with `metadata.tool_calls`.
When `name=` is specified, filters to commits where at least one tool call
matches the name.

### `find_tool_turns(name=)` -- Find paired (call, result) commits

```python
turns = t.find_tool_turns(name="grep")
# [ToolTurn(call=CommitInfo(...), results=[CommitInfo(...)]), ...]
```

A "tool turn" is an assistant tool-call commit paired with its tool result
commit(s). Matching is by `tool_call_id`. Returns a list of `ToolTurn`
dataclasses.

```python
@dataclass(frozen=True)
class ToolTurn:
    call: CommitInfo          # The assistant message with tool_calls
    results: list[CommitInfo] # The tool result(s) for this call
    tool_names: list[str]     # Names of tools called in this turn

    @property
    def all_hashes(self) -> list[str]:
        """All commit hashes in this turn (call + results)."""
        return [self.call.commit_hash] + [r.commit_hash for r in self.results]

    @property
    def result_hashes(self) -> list[str]:
        """Just the result commit hashes."""
        return [r.commit_hash for r in self.results]
```

### Composition with existing operations

```python
# Edit all grep results to be leaner
for ci in t.find_tool_results(name="grep"):
    t.tool_result(
        ci.metadata["tool_call_id"], "grep",
        extract_filenames(t.get_content(ci)),
        edit=ci.commit_hash,
    )

# Compress all tool turns into summaries
turns = t.find_tool_turns()
all_tool_hashes = [h for turn in turns for h in turn.all_hashes]
t.compress(commits=all_tool_hashes, system_prompt=TOOL_SUMMARIZE_SYSTEM)

# Compress just the grep turns, preserve everything else
grep_hashes = [h for turn in t.find_tool_turns(name="grep") for h in turn.all_hashes]
t.compress(commits=grep_hashes, instructions="summarize to matching filenames only")
```

### Agent-authored code note

Capable agents can write equivalent logic using `t.log()` + metadata
inspection directly, without these convenience methods:

```python
for entry in t.log():
    ci = t.get_commit(entry.commit_hash)
    meta = ci.metadata or {}
    if meta.get("name") == "grep" and "tool_call_id" in meta:
        t.tool_result(meta["tool_call_id"], "grep", summarized, edit=ci.commit_hash)
```

The query API is a convenience layer, not a requirement. Models with
strong coding capabilities can compose the primitives programmatically.

---

## Layer 3: Per-Tool Configuration

Declarative configuration for tool-specific summarization behavior.
Two modes: manual (instructions map) and automatic (threshold policy).

### `configure_tool_summarization(config)` -- Per-tool instructions

```python
t.configure_tool_summarization({
    "grep": "summarize to matching filenames and line numbers only",
    "read_file": "keep first 20 lines, summarize the rest",
    "bash": "preserve exit code, stderr, and last 10 lines of stdout",
})
```

When a tool result hook fires, the handler checks this config for
tool-specific instructions. If a match exists, the instructions are
passed to the summarization LLM. If no match, the result passes through
unmodified.

This is syntactic sugar over writing a hook handler with if-statements.
The config is stored on the Tract instance and used by the default
`tool_result` hook handler.

### Threshold-based auto-summarization

```python
t.configure_tool_summarization(
    auto_threshold=1000,  # tokens
    instructions={
        "grep": "summarize to matching filenames and line numbers only",
    },
)
```

When `auto_threshold` is set, tool results exceeding the token count are
automatically summarized (using the per-tool instructions if available,
or the generic TOOL_SUMMARIZE_SYSTEM prompt). Results under the threshold
pass through unchanged.

### Hook integration

The configuration layer is built ON TOP of the hook system:

```python
# configure_tool_summarization() internally does:
def _default_tool_handler(pending):
    config = pending.tract._tool_summarization_config
    if config and pending.tool_name in config.instructions:
        # Summarize with tool-specific instructions
        pending.summarize(instructions=config.instructions[pending.tool_name])
    elif config and config.auto_threshold and pending.token_count > config.auto_threshold:
        # Auto-summarize over threshold
        pending.summarize()
    else:
        pending.approve()  # Pass through unchanged

t.on("tool_result", _default_tool_handler)
```

Users can override with their own handler:
```python
t.on("tool_result", my_custom_handler)  # Replaces the default
```

---

## Summary of API surface

| Method | Layer | Purpose |
|--------|-------|---------|
| `tool_result(edit=)` | Primitive | Edit a single tool result in place |
| `compress_tool_calls()` | Primitive | Batch compress tool sequences |
| `find_tool_results(name=)` | Query | Find tool result commits |
| `find_tool_calls(name=)` | Query | Find tool-call assistant commits |
| `find_tool_turns(name=)` | Query | Find paired call+result turns |
| `configure_tool_summarization()` | Config | Declarative per-tool behavior |
| `t.on("tool_result", handler)` | Hook | Manual interception |
