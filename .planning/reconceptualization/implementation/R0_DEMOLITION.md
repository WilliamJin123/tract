# Phase R0: Demolition + Content Types + Compile Strategy

> **Status: SUPERSEDED by Phase 14 (Config + Directives + Middleware).** Rule engine was implemented (R0-R4, commit 7a86b94) then replaced (commit 23a89eb). Kept as historical reference.

## Goal

Remove the old hook/orchestrator/trigger systems and add the new content types
(RuleContent, MetadataContent) and compile strategy (full/messages/adaptive).

After this phase: substrate tests pass, new content types work, compile strategy
works. Cookbooks and hook/orchestrator tests are deleted.

## Task Breakdown

### Task 0.1: Delete Hook System

**Delete files:**
- `src/tract/hooks/__init__.py`
- `src/tract/hooks/pending.py`
- `src/tract/hooks/compress.py`
- `src/tract/hooks/gc.py`
- `src/tract/hooks/rebase.py`
- `src/tract/hooks/merge.py`
- `src/tract/hooks/trigger.py`
- `src/tract/hooks/generation.py`
- `src/tract/hooks/tool_result.py`
- `src/tract/hooks/event.py`
- `src/tract/hooks/validation.py`
- `src/tract/hooks/registry.py`
- `src/tract/hooks/guidance.py`
- `src/tract/hooks/introspection.py`
- `src/tract/hooks/dynamic.py`
- `src/tract/hooks/improve.py`
- `src/tract/hooks/retry.py`
- `src/tract/hooks/templates/` (entire dir)

**Delete tests:**
- `tests/test_hooks.py`
- `tests/test_hooks_serialization.py`
- `tests/test_hooks_display.py`
- `tests/test_hooks_guidance.py`
- `tests/test_hooks_orchestrator.py`
- `tests/test_hooks_merge.py`
- `tests/test_hooks_improve.py`
- `tests/test_hooks_trigger.py`
- `tests/test_hooks_retry_validate.py`
- `tests/test_hooks_two_stage.py`
- `tests/test_hooks_rebase.py`
- `tests/test_hooks_read_tools.py`
- `tests/test_hooks_gc.py`
- `tests/test_hook_stacking.py`
- `tests/test_hook_middleware_ordering.py`

**Delete cookbooks:**
- `cookbook/hooks/` (entire dir, 27 files)

### Task 0.2: Delete Orchestrator System

**Delete files:**
- `src/tract/orchestrator/__init__.py`
- `src/tract/orchestrator/config.py`
- `src/tract/orchestrator/models.py`
- `src/tract/orchestrator/loop.py`
- `src/tract/orchestrator/assessment.py`
- `src/tract/prompts/orchestrator.py`

**Delete tests:**
- `tests/test_orchestrator.py`
- `tests/test_orchestrator_models.py`

**Delete cookbooks:**
- `cookbook/getting_started/02_agent.py`
- `cookbook/agentic/sidecar/02_assessment_loop.py`
- `cookbook/agentic/sidecar/03_toolkit.py`
- `cookbook/e2e/autonomous_steering.py`

### Task 0.3: Delete Trigger System

**Delete files:**
- `src/tract/triggers/__init__.py`
- `src/tract/triggers/protocols.py`
- `src/tract/triggers/evaluator.py`
- `src/tract/triggers/builtin/` (entire dir)

**Delete tests:**
- `tests/test_trigger_new.py`
- `tests/test_trigger_evaluator.py`
- `tests/test_trigger_builtin.py`
- `tests/test_trigger_storage.py`
- `tests/test_trigger_integration.py`

**Delete cookbooks:**
- `cookbook/agentic/sidecar/01_triggers.py`
- `cookbook/agentic/sidecar/04_auto_tagger.py`

### Task 0.4: Delete Adjacent Dead Code

**Delete files:**
- `src/tract/retry.py` -- only imported from `__init__.py`, no internal usage
- `src/tract/prompts/guidance.py` -- hook guidance prompts
- `src/tract/prompts/improve.py` -- hook improvement prompts
- `src/tract/prompts/tagger.py` -- trigger auto-tagging prompts

**DO NOT DELETE:**
- `src/tract/formatting.py` -- used by protocols.py, models/commit.py,
  models/merge.py, operations/diff.py, operations/history.py, cli/.
  Only remove the `pprint_hooks()` function from it.

**Delete tests:**
- `tests/test_retry.py`
- `tests/test_retry_chat.py`
- `tests/test_retry_compression.py`
- `tests/test_dynamic_operations.py`
- `tests/test_profile_switching.py`
- `tests/test_improve.py` (entirely hooks-dependent)

**Delete cookbooks that depend heavily on deleted systems:**
- `cookbook/agentic/pending/` (entire dir, 6 files)
- `cookbook/agentic/self_managing/` (entire dir, 4 files)
- `cookbook/agentic/sidecar/` (entire dir, remaining files)
- `cookbook/e2e/` (audit which survive -- most depend on orchestrator)

### Task 0.4b: Rewrite `operations/compression.py`

**Problem:** `compress_range()` imports, constructs, and returns `PendingCompress`
from the hooks package. This is a substrate operation file that must survive, but
its return type is a hook object.

**Fix:** Create a plain frozen dataclass to replace `PendingCompress`:

```python
@dataclass(frozen=True)
class CompressRangeResult:
    """Result from compress_range() -- data needed to finalize compression."""
    summary_text: str
    summary_commits: list[str]  # hashes of new summary commits
    replaced_hashes: list[str]  # hashes being compressed
    pinned_hashes: list[str]    # preserved PINNED commits
    token_count: int            # tokens in summary
    generation_config: dict | None  # LLM config used (if any)
```

Modify `compress_range()` to return `CompressRangeResult` instead of
`PendingCompress`. Remove the import of `PendingCompress`. The `Tract.compress()`
method in `tract.py` will use `CompressRangeResult` directly to finalize
(the approve/reject/validation workflow is deleted with hooks).

### Task 0.4c: Fix Surviving Source File Imports

These source files import from deleted packages and must be surgically modified
in R0, not deferred to later phases:

- `src/tract/toolkit/__init__.py` -- line 17: `from tract.orchestrator.models import ToolCall`.
  Fix: Change to `from tract.protocols import ToolCall` (protocols.py already has
  a ToolCall class). Verify the two ToolCall classes are compatible or merge them.

- `src/tract/toolkit/definitions.py` -- line 997: `from tract.triggers.builtin import ...`.
  Fix: Remove trigger-related tool definitions entirely. They will be replaced
  by rule-based tool definitions in R3.

- `src/tract/cli/commands/merge.py` -- line 46: `from tract.hooks.merge import PendingMerge`.
  Fix: Remove the PendingMerge-based conflict review workflow. The merge CLI
  command should directly return MergeResult (conflicts raise MergeError as before
  the hook system existed).

### Task 0.5: Clean tract.py Facade

**Remove from `__init__` / constructor:**
- All `_hooks` state (`_hooks`, `_in_hook`, `_hook_log`)
- All trigger state (`_trigger_evaluator`, `_trigger_repo`, `_trigger_commit_count`, `_token_trigger_fired`)
- All orchestrator state (`_orchestrating`, `_orchestrator`, `_agent_loop`)
- `_operation_registry`, `_custom_hookable_ops`
- `_HookEntry` dataclass
- `_validate_dynamic_fields` helper

**Remove methods:**
- `on()`, `off()`, `hooks`, `hook_names`, `hook_log`, `list_hooks()`, `pprint_hooks()`, `print_hooks()`
- `_fire_hook()`, `_find_hook_entry_index()`
- `register_trigger()`, `evaluate_triggers()`, `trigger_evaluator` property
- `run_orchestrator()`, `orchestrate()`, `_check_orchestrator_triggers()`
- All dynamic operation methods (`register_operation()`, `fire_operation()`, etc.)

**Modify methods (remove hook/trigger calls):**
- `compile()` -- remove trigger evaluation, remove `_check_orchestrator_triggers`
- `compress()` -- remove `_fire_hook(PendingCompress(...))`, return result directly
- `gc()` -- remove `_fire_hook(PendingGC(...))`, return result directly
- `rebase()` -- remove hook wiring
- `merge()` -- remove hook wiring
- `commit()` -- remove trigger evaluation after commit
- `chat()` / `generate()` -- remove PendingGeneration hooks

**Keep methods:**
- All commit/compile/branch/merge/rebase/compress/gc core logic
- `as_tools()`, `as_callable_tools()` (toolkit kept, simplified in R3)
- `system()`, `user()`, `assistant()` shorthand
- `chat()`, `generate()` (remove hook parts only)
- All navigation: `log()`, `status()`, `diff()`, `head`, `branches`, etc.

### Task 0.6: Clean __init__.py

Remove all imports for deleted packages:
- Hook system: `Pending`, `PendingCompress`, `PendingToolResult`, `ValidationResult`, `HookRejection`
- Orchestrator: `Orchestrator`, `OrchestratorConfig`, `AutonomyLevel`, etc. (20+ symbols)
- Triggers: `Trigger`, `TriggerEvaluator`, all builtin triggers, `TriggerAction`, etc.
- Retry: `RetryResult`, `retry_with_steering`

Update `__all__` to remove all deleted symbols.

### Task 0.7: Add RuleContent and MetadataContent

**Modify `src/tract/models/content.py`:**

```python
class RuleContent(BaseModel):
    """A rule definition. Never compiled to LLM messages."""
    content_type: Literal["rule"] = "rule"
    name: str
    trigger: str
    condition: dict | None = None
    action: dict

class MetadataContent(BaseModel):
    """Structured workspace metadata. Never compiled to LLM messages."""
    content_type: Literal["metadata"] = "metadata"
    kind: str
    data: dict | str
    path: str | None = None
```

**Add to discriminated union:**
```python
ContentPayload = Annotated[
    Union[
        ...,  # existing 8 types
        RuleContent,
        MetadataContent,
    ],
    Field(discriminator="content_type"),
]
```

**Add to BUILTIN_CONTENT_TYPES set:**
```python
BUILTIN_CONTENT_TYPES: set[str] = {
    ...,  # existing 8
    "rule",
    "metadata",
}
```

**Add behavioral hints:**
```python
BUILTIN_TYPE_HINTS["rule"] = ContentTypeHints(
    default_priority="pinned",
    default_role="system",
    compression_priority=100,  # NEVER compress rules
)
BUILTIN_TYPE_HINTS["metadata"] = ContentTypeHints(
    default_priority="skip",     # not compiled to messages
    default_role="system",
    compression_priority=100,  # NEVER compress metadata
)
```

**Key design point:** Both types have `compression_priority=100` so the
compression engine preserves them. The compiler skips them via the `compilable`
flag on ContentTypeHints.

**Add a `compilable` flag to ContentTypeHints:**

```python
@dataclass(frozen=True)
class ContentTypeHints:
    default_priority: str = "normal"
    default_role: str = "assistant"
    compression_priority: int = 50
    aggregation_rule: str = "concatenate"
    compilable: bool = True  # NEW: if False, never included in compiled output
```

Then in `BUILTIN_TYPE_HINTS`:
```python
BUILTIN_TYPE_HINTS["rule"] = ContentTypeHints(
    default_priority="normal",
    default_role="system",
    compression_priority=100,
    compilable=False,  # never in LLM messages
)
BUILTIN_TYPE_HINTS["metadata"] = ContentTypeHints(
    default_priority="normal",
    default_role="system",
    compression_priority=100,
    compilable=False,  # never in LLM messages
)
```

**Modify compiler** (`engine/compiler.py`, `_build_effective_commits`):
```python
def _build_effective_commits(self, commits, edit_map, priority_map):
    effective = []
    for c in commits:
        if c.operation == CommitOperation.EDIT:
            continue
        if priority_map.get(c.commit_hash) == Priority.SKIP:
            continue
        # NEW: skip non-compilable content types
        hints = BUILTIN_TYPE_HINTS.get(c.content_type)
        if hints and not hints.compilable:
            continue
        effective.append(c)
    return effective
```

### Task 0.8: Add Compile Strategy

**Modify `engine/compiler.py`:**

Add `strategy` parameter to `compile()`:

```python
def compile(
    self,
    tract_id: str,
    head_hash: str,
    *,
    at_time: datetime | None = None,
    at_commit: str | None = None,
    include_edit_annotations: bool = False,
    include_reasoning: bool = False,
    strategy: str = "full",  # NEW: "full" | "messages" | "adaptive"
    strategy_k: int = 5,      # NEW: K for adaptive strategy
) -> CompiledContext:
```

Strategy implementations:
- `"full"` -- current behavior (all content)
- `"messages"` -- commit messages only, no content bodies. For each commit,
  use the commit `message` field instead of blob content. Free (no blob reads).
  **Fallback for empty/None message**: use `f"[{commit.content_type}] {commit.commit_hash[:8]}"`.
- `"adaptive"` -- last K commits at full detail, everything before at messages-only.
  Non-destructive read-time lens.

For "messages" strategy, `_build_messages` returns Message objects where content
is the commit's `message` field (the auto-generated or user-provided commit
message stored on CommitRow).

For "adaptive", split effective_commits into two groups:
- `tail = effective_commits[-k:]` -- full content
- `head = effective_commits[:-k]` -- messages only
Build messages for each group with appropriate detail level.

**Cache design:** The compile cache key stays as `head_hash` (unchanged). The cache
always stores the `"full"` strategy snapshot. Other strategies (`"messages"`,
`"adaptive"`) are derived at read time from the full snapshot:

- `"messages"` — iterate the full snapshot's commits, replace content with commit
  messages. No separate cache entry.
- `"adaptive(k)"` — split the full snapshot at index `-k`: tail keeps full content,
  head replaces with commit messages. No separate cache entry.

This preserves the existing O(1) incremental patching (`extend_for_append`,
`patch_for_edit`, `patch_for_annotate`) without modification. The derivation
functions live in the compiler, not the cache:

```python
def _apply_strategy(self, snapshot: CompileSnapshot, strategy: str, k: int) -> CompiledContext:
    """Derive a strategy-specific view from the canonical full snapshot."""
    if strategy == "full":
        return self._snapshot_to_context(snapshot)
    elif strategy == "messages":
        return self._derive_messages_only(snapshot)
    elif strategy == "adaptive":
        return self._derive_adaptive(snapshot, k)
    raise ValueError(f"Unknown compile strategy: {strategy!r}")
```

**Modify `tract.py` compile():**

Thread `strategy` and `strategy_k` through to compiler:

```python
def compile(self, *, strategy: str = "full", strategy_k: int = 5, ...):
    ...
    result = self._compiler.compile(
        ...,
        strategy=strategy,
        strategy_k=strategy_k,
    )
```

### Task 0.9: Clean Storage Schema (Optional)

Decide: remove HookWiringRow, TriggerLogRow, ConfigChangeRow, DynamicOpSpecRow
from `storage/schema.py`?

**Recommendation:** Remove them. They're only referenced by deleted code.
But check if any surviving code references them first. The schema version
needs to bump.

### Task 0.10: Delete Remaining Dead Cookbooks

After tasks 0.1-0.4, audit remaining cookbooks for broken imports:
```bash
cd cookbook && grep -rl "from tract.hooks\|from tract.orchestrator\|from tract.triggers" .
```

Delete any files that import deleted modules. Keep files that only use
substrate operations (commit, compile, branch, merge, etc.).

---

## Test Plan

### Tests Requiring Surgical Edits (keep file, remove specific tests)

These files are NOT deleted but contain some tests that import deleted modules.
Remove only the affected test functions, keep the rest.

- `test_toolkit.py` -- line 582: `from tract.triggers.builtin.compress import CompressTrigger`
  (remove the specific test that uses CompressTrigger)
- `test_compression_storage.py` -- lines 808/828: `from tract.hooks.compress import PendingCompress`
  (remove tests that use PendingCompress)
- `test_operation_config.py` -- lines 551+: `from tract.orchestrator.*`
  (remove orchestrator-related config tests, ~80 lines)
### Surviving Tests (must pass)
- `test_models/test_content.py` (+ new tests for RuleContent, MetadataContent)
- `test_storage/test_schema.py`
- `test_storage/test_repositories.py`
- `test_engine/test_commit.py`
- `test_engine/test_compiler.py` (+ new tests for compile strategy)
- `test_engine/test_hashing.py`
- `test_engine/test_tokens.py`
- `test_tract.py` (will need modifications to remove hook/trigger tests)
- `test_navigation.py`
- `test_operations.py`
- `test_branch.py`
- `test_merge.py`
- `test_rebase.py`
- `test_compression.py`
- `test_compression_storage.py`
- `test_compression_lifecycle.py`
- `test_gc.py`
- `test_reorder.py`
- `test_llm.py`
- `test_conversation.py`
- `test_operation_clients.py`
- `test_operation_config.py`
- `test_format_shorthand.py`
- `test_auto_message.py`
- `test_tool_calls.py`
- `test_integration_multiagent.py`
- `test_spawn.py`
- `test_spawn_storage.py`
- `test_session.py`
- `test_persistence.py`
- `test_tags.py`
- `test_edit_history.py`
- `test_toolkit.py` (may need modifications if orchestrator refs removed)
- `test_important_priority.py`
- `test_token_tolerance.py`
- `test_tool_tracking.py`
- `test_tool_query.py`
- `test_reasoning.py`
- `test_pprint.py`
- `test_compile_records.py`
- `test_config_provenance.py`
- `test_config_resolution.py`
- `test_operation_prompts.py`

### New Tests

**`tests/test_content_new_types.py` (~20 tests):**
- `test_rule_content_creation` -- RuleContent validates with all fields
- `test_rule_content_minimal` -- name + trigger + action only, condition=None
- `test_rule_content_discriminator` -- detected via content_type="rule"
- `test_metadata_content_dict_data` -- kind + data as dict
- `test_metadata_content_str_data` -- kind + data as string
- `test_metadata_content_with_path` -- path field populated
- `test_metadata_content_no_path` -- path=None default
- `test_validate_rule_content` -- through validate_content()
- `test_validate_metadata_content` -- through validate_content()
- `test_rule_not_compiled` -- rule commits excluded from compile output
- `test_metadata_not_compiled` -- metadata commits excluded from compile output
- `test_rule_not_compressed` -- compression_priority=100 preserved
- `test_metadata_not_compressed` -- compression_priority=100 preserved
- `test_compilable_hint_flag` -- ContentTypeHints.compilable defaults True
- `test_compilable_false_skips` -- commits with compilable=False excluded

**`tests/test_compile_strategy.py` (~15 tests):**
- `test_full_strategy_default` -- strategy="full" is same as current
- `test_messages_strategy` -- only commit messages, no content
- `test_messages_strategy_empty_message` -- fallback for commits without message
- `test_adaptive_strategy_k5` -- last 5 full, rest messages
- `test_adaptive_strategy_k1` -- only last commit full
- `test_adaptive_strategy_k_larger_than_commits` -- all full when K > N
- `test_adaptive_strategy_k0` -- all messages (edge case)
- `test_strategy_through_facade` -- t.compile(strategy="adaptive", strategy_k=3)
- `test_strategy_does_not_mutate_dag` -- DAG unchanged after compile with any strategy
- `test_strategy_with_edits` -- edit resolution works with messages strategy
- `test_strategy_with_pinned` -- PINNED commits always full regardless of strategy
- `test_strategy_ignores_rules` -- rule commits excluded regardless of strategy
- `test_strategy_ignores_metadata` -- metadata commits excluded regardless
- `test_strategy_with_time_travel` -- at_time + strategy composable
- `test_strategy_cache_key` -- different strategies produce different cache keys

---

## Acceptance Criteria

1. `python -m pytest tests/` passes with 0 failures (deleted tests don't exist)
2. No import of `tract.hooks`, `tract.orchestrator`, or `tract.triggers` anywhere
   in surviving source code
3. `RuleContent` and `MetadataContent` round-trip through commit/compile
4. Rule and metadata commits are excluded from compiled output
5. `compile(strategy="messages")` returns commit-message-only content
6. `compile(strategy="adaptive", strategy_k=3)` returns last 3 full, rest messages
7. No orphaned files in `src/tract/`

## Execution Order

1. Tasks 0.1-0.4 in parallel (all deletions are independent)
2. Task 0.5 (tract.py surgery -- depends on knowing what's deleted)
3. Task 0.6 (init.py cleanup -- depends on 0.5)
4. Run surviving tests -- fix any broken imports
5. Task 0.7 (content types -- independent of deletions)
6. Task 0.8 (compile strategy -- depends on 0.7 for compilable flag)
7. Task 0.9 (optional schema cleanup)
8. Task 0.10 (cookbook audit)
9. Run full test suite including new tests
