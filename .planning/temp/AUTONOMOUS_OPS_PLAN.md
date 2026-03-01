# Autonomous Operations Plan

## Context

Discussion identified 4 gaps + 1 rename needed for tract to fully support autonomous LLM agents managing their own context. The core mental model:

- **Trigger** = watches conditions, fires an operation when threshold is met
- **Hook** = intercepts the fired operation, gates/modifies/approves it
- Flow: `Trigger evaluates → fires operation → Hook intercepts → approve/reject/modify → execute`

---

## Gap 1: Semantic Tags + Auto-Classify

**Problem:** No structured way for an agent to categorize commits. Metadata is freeform, content_type is single-valued, and neither supports multi-label queryable classification.

**Why not content_type + metadata?** Tags provide multi-label classification (a commit can be both `tool_call` AND `reasoning`), queryable sets (`query_by_tags(match="all")`), and enforced vocabulary with strict mode — capabilities that single-valued `content_type` and unstructured `metadata` cannot express.

**Distinction from Priority annotations:** Tags = classification (what the content *is*). Priority annotations = compilation behavior (what to *do with* the content during compile). A commit can have `tag="instruction"` AND `priority=PINNED` — these are orthogonal.

### Changes

#### 1a. `tags: list[str]` on CommitInfo (immutable at commit time)
- Add `tags` field to `CommitInfo` (default `[]`)
- Add `tags_json` JSON column to `CommitRow`
- Schema migration (next version)
- Tags describe what the content *is* — factual, set at creation

#### 1b. Mutable annotation tags via `t.tag()`
- New method: `t.tag(hash, "dead_end")` for mutable retrospective tags
- Separate from `t.annotate()` which remains unchanged for priority annotations
- Agent can tag old commits: `t.tag(hash, "dead_end")`
- These describe what the agent *thinks about* the content — evolves over time
- Storage: new `TagAnnotationRow` table (tract_id, target_hash, tag, created_at)

#### 1c. TagRegistry (strict enforcement)
- `TagRegistry` class on Tract tracking known tags with descriptions
- Pre-seeded base tags from heuristics:
  - `"instruction"` (system messages)
  - `"tool_call"` (has tool_calls in metadata)
  - `"tool_result"` (has tool_call_id in metadata)
  - `"reasoning"` (assistant without tool_calls)
  - `"revision"` (EDIT operations)
  - `"observation"` (user messages with data)
  - `"decision"` (assistant with explicit choices)
  - `"summary"` (compression output)
- `t.register_tag(name, description)` — agent registers new tags explicitly
- `t.list_tags()` — returns all registered tags with counts and descriptions
- **Strict mode:** passing an unregistered tag raises an error
- Store registry in SQLite (tag_name, description, created_at, auto_created)

#### 1d. `_auto_classify()` replaces `_auto_message()`
- Returns both `message` and `tags: list[str]`
- Heuristic-based, no LLM call:
  - Inspects content_type, metadata keys, operation type
  - Only assigns tags from the registered vocabulary
- Agent can override/append: `t.assistant("...", tags=["hypothesis"])`
- Auto-tags + explicit tags merge (deduplicated)

#### 1e. Tag-based queries
- `t.log(tags=["reasoning", "decision"])` — filter log by tags
- `t.query_by_tags(tags, match="any"|"all")` — flexible query
- Enables Gap 4 (selective sub-agent context)

---

## Gap 2: Policy → Trigger Rename + Symmetry

**Problem:** "Policy" implies permission/restriction. "Trigger" says what it does — initiates operations when conditions are met. Also, only compress has a trigger; other operations are missing theirs.

### Changes

#### 2a. Rename policy → trigger throughout codebase
- `Policy` ABC → `Trigger` ABC
- `.trigger` property → `.fires_on` (avoids `Trigger.trigger` tautology; values: `"compile"` or `"commit"`)
- `PolicyEvaluator` → `TriggerEvaluator`
- `PolicyAction` → `TriggerAction`
- `CompressPolicy` → `CompressTrigger`
- `PinPolicy` → `PinTrigger`
- `BranchPolicy` → `BranchTrigger`
- `ArchivePolicy` (builtin) → `ArchiveTrigger`
- `PendingPolicy` → `PendingTrigger`
- `t.configure_policies()` → `t.configure_triggers()`
- `t.register_policy()` → `t.register_trigger()`
- `t.unregister_policy()` → `t.unregister_trigger()`
- `t.pause_all_policies()` → `t.pause_all_triggers()`
- `t.resume_all_policies()` → `t.resume_all_triggers()`
- Directory: `src/tract/policy/` → `src/tract/triggers/`
- Test files: `test_policy_*` → `test_trigger_*`
- Models: `models/policy.py` → `models/trigger.py`
- Hookable op string: `"policy"` → `"trigger"` in `_HOOKABLE_OPS`
- `_trace_meta` key: `"policy_config"` → `"trigger_config"`
- All internal references, imports, docstrings

#### 2b. Add missing triggers (specs deferred to phase planning)
- **`RebaseTrigger`** — evaluates branch divergence, fires rebase when threshold exceeded
  - Config: target branch, divergence_commits threshold, divergence_tokens threshold
  - Detailed evaluation logic, parameter passing, and interaction with existing `check_rebase_safety()` to be specified during phase planning
- **`GCTrigger`** — evaluates dead commit count or storage size, fires GC
  - Config: max_dead_commits, max_storage_bytes
  - Detailed interaction with existing `plan_gc()` retention policies to be specified during phase planning
- **`MergeTrigger`** — evaluates branch completion criteria, fires merge
  - Config: target branch, completion heuristic (all commits tagged "complete"?)
  - Depends on Gap 1 tags; detailed evaluation logic to be specified during phase planning

#### 2c. Ensure symmetry: every trigger has a hook, every hookable op has a trigger path
- Verify all hookable operations can be triggered
- Verify annotate has a hook if PinTrigger fires it

---

## Gap 3: File-Based Persistence

**Problem:** Agent-built hooks, triggers, dynamic ops, and configs vanish on process death. Dynamic ops can serialize but nobody writes to disk.

### Changes

#### 3a. `.tract/` directory structure
```
.tract/
  hooks/
    compress_guard.py  # agent-written hook handler
    rebase_check.py    # agent-written hook handler
  triggers/
    auto_rebase.py     # agent-written trigger
  workflows/
    README.md          # agent-generated docs
    research_flow.py   # agent-defined composite workflow
```

#### 3b. Extend existing SQLite database (no separate config.db)
- Add tables to the existing tract SQLite database:
  - Tag registry table (name, description, created_at)
  - Trigger configs table (trigger_name, operation, params_json, enabled)
  - Hook wiring table (operation, handler_file, handler_function, priority)
  - Dynamic operation specs (already serializable via to_config)
  - Operation configs (LLMConfig, OperationConfigs serialized)
- One DB, one migration system, no split-brain risk

#### 3c. File-based code persistence
- Agent writes Python files to `.tract/hooks/` and `.tract/triggers/`
- Files follow a convention (module-level `handler` function or `trigger` class)
- On `Tract.open()`, scan directories, import modules, wire based on SQLite config
- Same trust model as dynamic ops `compile_action()` (local file execution, analogous to `.git/hooks/`)

#### 3d. Auto-load on startup with error recovery
- `Tract.open()` checks for `.tract/` directory
- If exists, loads tag registry, trigger configs, hook wiring
- Imports and registers code from hooks/ and triggers/
- **Error recovery:** broken/corrupt Python files are logged and skipped, never crash `Tract.open()`. Quarantined modules listed in warning log so agent can fix them.
- Agent's previous session state is fully restored (minus quarantined modules)

#### 3e. Save API
- `t.save_hook(name, code, operation)` — writes file + registers in SQLite
- `t.save_trigger(name, code, operation, config)` — writes file + registers
- `t.save_workflow(name, code, description)` — writes file + markdown docs

---

## Gap 4: Sub-Agent Deployment with Curated Context

**Problem:** `session.spawn(inheritance="head_snapshot")` gives a flat snapshot. No way to give a child a real branch with curated history.

### Changes

#### 4a. `session.deploy()` high-level operation
- New method on the existing `Session` class (alongside `session.spawn()` and `session.collapse()`)
- Composes: branch → curate → spawn
- Main agent stays on its branch (never moves)
- Creates a new branch, curates it, hands to sub-agent

```python
child = session.deploy(
    parent=t,
    purpose="research X",
    branch_name="research-x",
    curate={
        "drop": [hash1, hash2],        # remove irrelevant commits
        "compact_before": hash3,         # compress old history
        "reorder": [(h1, h2), ...],     # reorder commits
        "keep_tags": ["instruction", "decision"],  # filter by tags (Gap 1)
    }
)
```

#### 4b. Branch-based inheritance mode
- New inheritance mode: `"branch"` (vs existing `"head_snapshot"` and `"full_clone"`)
- Child gets a real branch with real commit history
- Can log, inspect, annotate, continue building on structured history
- Parent can merge child branch back when done

#### 4c. Curation pipeline (defined execution order)
Before handing off, apply curation operations in this fixed order:

1. **`keep_tags`** — Filter commits to only those with matching tags (narrowest working set first, reduces work for later steps)
2. **`drop`** — Remove explicitly listed commit hashes
   - ERROR if hash is `edit_target` of a remaining commit
   - ERROR if hash not in working set after filtering
3. **`compact_before`** — Compress all commits before the marker (operates on already-filtered, already-dropped set)
4. **`reorder`** — Reorder remaining commits
   - WARNING if EDIT commit comes before its target (uses existing `check_reorder_safety()`)

All operations execute on the child branch. Parent branch is untouched.

#### 4d. Merge-back workflow
- When sub-agent finishes, parent can:
  - `session.collapse()` (existing — summarize and commit to parent)
  - `t.merge(child_branch)` (new option — full merge of child commits into parent)
  - Peek first: `child.log()`, `child.status()` for review

---

## Execution Order

Each gap is an independent phase with its own success criteria and verification:

1. **Gap 2a (rename)** — do first since it touches the most files and is a pure rename. Doing it before adding new code avoids writing code with the old names.
2. **Gap 1 (tags)** — builds on the renamed trigger system, adds schema + TagRegistry + auto-classify
3. **Gap 2b (new triggers)** — now add RebaseTrigger, GCTrigger, MergeTrigger using the new naming
4. **Gap 3 (persistence)** — depends on tags and triggers existing to persist them
5. **Gap 4 (deploy)** — depends on tags for filtering, triggers/hooks for automation

## Verification

### Gap 2a (rename): Success Criteria
- All existing tests pass with new naming (zero regressions)
- No references to old names (`Policy`, `PolicyEvaluator`, etc.) remain in source or tests
- `.fires_on` property replaces `.trigger` on all trigger subclasses

### Gap 1 (tags): Success Criteria
- Tag CRUD roundtrip: create, query, update, delete tags
- `_auto_classify()` assigns ≥1 tag to every known content type
- `query_by_tags(match="any")` and `query_by_tags(match="all")` return correct results
- `t.tag()` creates mutable annotation tags separate from priority annotations
- Strict mode rejects unregistered tags

### Gap 2b (new triggers): Success Criteria
- Each trigger fires when its condition is met and not otherwise
- Each trigger's action is routable through the hook system
- Trigger parameters pass correctly to downstream operations

### Gap 3 (persistence): Success Criteria
- Tag registry, trigger configs, hook wiring survive process restart
- Broken Python files in `.tract/hooks/` are logged and skipped, not crash
- `t.save_hook()` / `t.save_trigger()` roundtrip correctly

### Gap 4 (deploy): Success Criteria
- `session.deploy()` creates child branch with curated history
- Curation pipeline executes in order: filter → drop → compact → reorder
- Drop of edit_target raises error
- Parent branch is unmodified after deploy
- Merge-back produces correct commit history
