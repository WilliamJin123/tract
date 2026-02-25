# Tool Result Summarization -- Implementation Plan

## Context

Tool results are often the most verbose content in an LLM context window.
A single grep can return hundreds of lines when only a filename mattered.
Tract needs first-class support for managing tool result verbosity while
maintaining full provenance.

### What already exists (done this session)

- `TOOL_SUMMARIZE_SYSTEM` prompt in `prompts/summarize.py`
- `CONVERSATION_SUMMARIZE_SYSTEM` (old default moved to named variant)
- `DEFAULT_SUMMARIZE_SYSTEM` rewritten to be neutral (no "Previously in
  this conversation:" prefix)
- `compress_tool_calls()` on Tract (batch compression convenience)
- All three prompts exported from `tract.__init__`
- 10 new tests passing

### What needs to be built

1. `tool_result(edit=)` -- the atomic primitive
2. `PendingToolResult` hook -- automatic interception
3. Tool query API -- `find_tool_results()`, `find_tool_calls()`, `find_tool_turns()`
4. `ToolTurn` dataclass
5. Per-tool configuration -- `configure_tool_summarization()`
6. Update `compress_tool_calls()` to use the query API internally
7. Cookbook update -- `08_tool_calling.py` to demonstrate the feature
8. Tests

---

## Phase 1: `tool_result(edit=)` primitive

**Scope:** Add `edit=` parameter to `tool_result()`, matching the pattern
used by `system()`, `user()`, and `assistant()`.

**File:** `src/tract/tract.py`

**Change:**

```python
def tool_result(
    self,
    tool_call_id: str,
    name: str,
    content: str,
    *,
    edit: str | None = None,          # NEW
    message: str | None = None,
    metadata: dict | None = None,
) -> CommitInfo:
    meta = {**(metadata or {}), "tool_call_id": tool_call_id, "name": name}
    return self.commit(
        DialogueContent(role="tool", text=content),
        operation=CommitOperation.EDIT if edit else CommitOperation.APPEND,
        edit_target=edit,
        message=message or f"tool result: {name}",
        metadata=meta,
    )
```

**Tests:**
- `test_tool_result_edit_basic` -- edit replaces content, original in history
- `test_tool_result_edit_preserves_metadata` -- tool_call_id and name survive
- `test_tool_result_edit_compiled_shows_new` -- compile() serves edited version
- `test_tool_result_edit_log_shows_both` -- log() shows both versions

**Depends on:** Nothing. Can be built immediately.

---

## Phase 2: Tool query API

**Scope:** Methods to find tool-related commits by role and tool name.

**Files:**
- `src/tract/tract.py` -- new methods
- `src/tract/protocols.py` -- `ToolTurn` dataclass

### 2a: `ToolTurn` dataclass

```python
@dataclass(frozen=True)
class ToolTurn:
    """A paired tool-call assistant message and its tool result(s)."""
    call: CommitInfo
    results: list[CommitInfo]
    tool_names: list[str]

    @property
    def all_hashes(self) -> list[str]:
        return [self.call.commit_hash] + [r.commit_hash for r in self.results]

    @property
    def result_hashes(self) -> list[str]:
        return [r.commit_hash for r in self.results]

    @property
    def total_tokens(self) -> int:
        return self.call.token_count + sum(r.token_count for r in self.results)
```

### 2b: Query methods on Tract

```python
def find_tool_results(
    self,
    name: str | None = None,
    after: str | None = None,
) -> list[CommitInfo]:
    """Find tool result commits on the current branch.

    Detection: commit metadata contains "tool_call_id".
    Filters: name= matches metadata "name" field. after= includes only
    commits after the given hash (exclusive).
    """

def find_tool_calls(
    self,
    name: str | None = None,
) -> list[CommitInfo]:
    """Find assistant commits that requested tool calls.

    Detection: commit metadata contains "tool_calls".
    Filters: name= matches if any tool_call in the list has that name.
    """

def find_tool_turns(
    self,
    name: str | None = None,
) -> list[ToolTurn]:
    """Find paired tool-call + tool-result commit groups.

    Walks the branch, matches tool results to their originating
    assistant tool-call message by tool_call_id. Returns ToolTurn
    instances with the call and its result(s) grouped together.
    """
```

**Implementation strategy:** Walk `t.log()`, inspect metadata on each
commit. Build an index of tool_call_ids to results, then match calls
to results. This is O(n) in commit count -- acceptable for typical
conversations (< 1000 commits).

**Tests:**
- `test_find_tool_results_all` -- finds all tool results
- `test_find_tool_results_by_name` -- filters by tool name
- `test_find_tool_results_after` -- filters by position
- `test_find_tool_calls_all` -- finds assistant tool-call messages
- `test_find_tool_calls_by_name` -- filters by tool name in tool_calls list
- `test_find_tool_turns_groups_correctly` -- call + results paired
- `test_find_tool_turns_multi_tool` -- assistant calls 2 tools, both results grouped
- `test_find_tool_turns_by_name` -- filters to turns containing that tool
- `test_find_tool_turns_empty` -- no tool calls returns empty list

**Depends on:** Nothing. Can be built in parallel with Phase 1.

---

## Phase 3: `PendingToolResult` hook

**Scope:** New hookable operation for tool results. Fires when
`tool_result()` is called (if a handler is registered). Handler can
edit, summarize, or reject the result before it's committed.

**Files:**
- `src/tract/hooks/tool_result.py` -- new PendingToolResult class
- `src/tract/hooks/__init__.py` -- export
- `src/tract/tract.py` -- fire hook from `tool_result()`, add to _HOOKABLE_OPS

### 3a: `PendingToolResult` class

```python
@dataclass
class PendingToolResult(Pending):
    """A tool result that has been received but not yet committed.

    Handlers can inspect, edit, summarize, or reject the result
    before it enters the commit chain.
    """

    tool_call_id: str = ""
    tool_name: str = ""
    content: str = ""
    token_count: int = 0

    # Set by the summarize() method after LLM call
    original_content: str | None = None

    _public_actions: set[str] = field(
        default_factory=lambda: {
            "approve", "reject", "edit_result", "summarize",
        },
        repr=False,
    )

    def edit_result(self, new_content: str) -> None:
        """Replace the result content before commit."""
        self._require_pending()
        if self.original_content is None:
            self.original_content = self.content
        self.content = new_content

    def summarize(
        self,
        *,
        instructions: str | None = None,
        target_tokens: int | None = None,
    ) -> None:
        """Summarize the result content via LLM.

        Uses TOOL_SUMMARIZE_SYSTEM as the system prompt.
        The original content is preserved in original_content.
        """
        self._require_pending()
        if self.original_content is None:
            self.original_content = self.content

        # Use tract's LLM client for summarization
        from tract.prompts.summarize import TOOL_SUMMARIZE_SYSTEM, build_summarize_prompt
        llm = self.tract._resolve_llm_client("compress")
        prompt = build_summarize_prompt(
            f"[tool:{self.tool_name}]: {self.content}",
            target_tokens=target_tokens,
            instructions=instructions,
        )
        response = llm.chat([
            {"role": "system", "content": TOOL_SUMMARIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        self.content = response["choices"][0]["message"]["content"]
```

### 3b: Hook firing in `tool_result()`

```python
def tool_result(self, tool_call_id, name, content, *, edit=None, ...):
    # If this is an edit, skip the hook (user already decided what to write)
    if edit is not None:
        return self._commit_tool_result(tool_call_id, name, content, edit=edit, ...)

    # Check for hook
    has_hook = "tool_result" in self._hooks or "*" in self._hooks
    if has_hook and not self._in_hook:
        pending = PendingToolResult(
            operation="tool_result",
            tract=self,
            tool_call_id=tool_call_id,
            tool_name=name,
            content=content,
            token_count=self._token_counter.count(content),
        )
        pending._execute_fn = lambda p: self._commit_tool_result(
            p.tool_call_id, p.tool_name, p.content, metadata=metadata,
            message=message,
        )
        self._fire_hook(pending)

        if pending.status == "approved":
            return pending._result
        elif pending.status == "rejected":
            return None  # or raise? TBD
        return pending  # unresolved

    # No hook: commit directly
    return self._commit_tool_result(tool_call_id, name, content, edit=edit, ...)
```

### 3c: Provenance

When `summarize()` is called on PendingToolResult:
- `original_content` preserves the raw output
- The committed result has `metadata.summarized_from` = original content hash
- If using `edit=` instead, the edit system provides provenance automatically

When the hook fires and the handler edits/summarizes:
- The commit metadata includes `triggered_by` from the hook
- The original content is recoverable from `pending.original_content`

**Tests:**
- `test_hook_fires_on_tool_result` -- handler receives PendingToolResult
- `test_hook_edit_result` -- handler edits, commit has new content
- `test_hook_summarize` -- handler calls summarize(), LLM runs
- `test_hook_reject` -- handler rejects, no commit created
- `test_hook_no_fire_on_edit` -- `edit=` bypasses the hook
- `test_hook_passthrough` -- handler approves without changes
- `test_hook_carries_token_count` -- pending has token_count
- `test_no_hook_commits_directly` -- without handler, commits as before

**Depends on:** Phase 1 (needs `edit=` support for internal wiring).

---

## Phase 4: Per-tool configuration

**Scope:** Declarative config for tool-specific summarization.
Built on top of the hook system.

**Files:**
- `src/tract/tract.py` -- `configure_tool_summarization()` method
- `src/tract/models/config.py` -- `ToolSummarizationConfig` dataclass

### 4a: Config dataclass

```python
@dataclass(frozen=True)
class ToolSummarizationConfig:
    instructions: dict[str, str] = field(default_factory=dict)
    auto_threshold: int | None = None  # token count threshold
    default_instructions: str | None = None  # fallback for unlisted tools
```

### 4b: `configure_tool_summarization()` method

```python
def configure_tool_summarization(
    self,
    instructions: dict[str, str] | None = None,
    auto_threshold: int | None = None,
    default_instructions: str | None = None,
) -> None:
    """Configure automatic tool result summarization.

    Sets up a tool_result hook that summarizes results based on
    per-tool instructions and/or token count thresholds.
    """
    self._tool_summarization_config = ToolSummarizationConfig(
        instructions=instructions or {},
        auto_threshold=auto_threshold,
        default_instructions=default_instructions,
    )

    def _auto_handler(pending):
        config = self._tool_summarization_config
        tool_instructions = config.instructions.get(pending.tool_name)

        if tool_instructions:
            pending.summarize(instructions=tool_instructions)
        elif config.auto_threshold and pending.token_count > config.auto_threshold:
            instr = config.default_instructions
            pending.summarize(instructions=instr)
        else:
            pending.approve()

    self.on("tool_result", _auto_handler)
```

**Tests:**
- `test_config_per_tool_instructions` -- grep gets summarized with specific instructions
- `test_config_auto_threshold` -- results over threshold get summarized
- `test_config_under_threshold_passthrough` -- small results pass through
- `test_config_default_instructions` -- unlisted tools use default
- `test_config_override_with_custom_handler` -- user can replace

**Depends on:** Phase 3 (needs the hook system).

---

## Phase 5: Refine `compress_tool_calls()` and cookbook

**Scope:** Update `compress_tool_calls()` to use the query API internally.
Update cookbook `08_tool_calling.py` to demonstrate the full feature set.

**Changes to `compress_tool_calls()`:**
- Use `find_tool_turns()` internally when no explicit `commits` list given
- Accept `name=` filter to compress only specific tool types
- Keep current behavior when `commits` is explicit

**Cookbook updates:**
- Show `tool_result(edit=)` for single-result cleanup
- Show `compress_tool_calls()` for batch compression
- Show `configure_tool_summarization()` for automatic mode
- Show query API for inspecting tool history

**Depends on:** Phases 1-4.

---

## Implementation Order

```
Phase 1: tool_result(edit=)                    [small, no dependencies]
Phase 2: Query API + ToolTurn                  [small, no dependencies]
  -- Phases 1 and 2 can be built in parallel --
Phase 3: PendingToolResult hook                [medium, depends on Phase 1]
Phase 4: Per-tool configuration                [small, depends on Phase 3]
Phase 5: Refine + cookbook                      [small, depends on all]
```

Estimated: ~200-300 lines of new code + ~150-200 lines of tests.

---

## Design Principles

1. **Primitives first.** `edit=` on `tool_result()` is the foundation.
   Everything else builds on it.

2. **Hooks are optional.** Without hooks, `tool_result()` commits directly
   as before. Hooks add interception, not complexity.

3. **Agents can code.** The query API is convenience. Capable agents can
   write equivalent loops using `t.log()` + metadata inspection. The API
   should be clean enough that either path works.

4. **Guidance upfront.** `instructions=` on summarize/compress provides
   guidance to the first LLM call. No retry needed for the common case.
   GuidanceMixin's edit-then-retry is for interactive review.

5. **Provenance by default.** Every edit preserves the original. Every
   hook records `triggered_by`. Every compression tracks source commits.

6. **Configuration is sugar.** `configure_tool_summarization()` is a
   convenience wrapper that registers a hook handler. Users can always
   write their own handler for full control.
