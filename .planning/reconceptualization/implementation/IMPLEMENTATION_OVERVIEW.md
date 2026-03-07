# Reconceptualization Implementation Plan

> **Status: SUPERSEDED by Phase 14 (Config + Directives + Middleware).** Rule engine was implemented (R0-R4, commit 7a86b94) then replaced (commit 23a89eb). Kept as historical reference.

Status: Ready for execution
Date: 2026-03-05

## Guiding Principles

1. **Clean break** -- nobody is using this framework yet. No backward compat.
2. **Substrate is sacred** -- graph primitives (commit, branch, merge, compress,
   rebase, compile) stay untouched. The rule system is layered on top.
3. **Tests before trust** -- each phase has acceptance criteria that must pass
   before moving on. No "I'll test later" debt.
4. **Cookbooks are the spec** -- if a POC cookbook can't demonstrate the feature
   on a cheap model (Cerebras/Groq), the design is wrong.

## Codebase Impact Summary

### What Gets Deleted (~5,200 lines source, ~912 tests)

| Package | Lines | Tests | Reason |
|---------|-------|-------|--------|
| `src/tract/hooks/` | 3,642 | 536 | Replaced by rule system |
| `src/tract/orchestrator/` | 956 | 142 | Replaced by default loop + host app |
| `src/tract/triggers/` | 622 | 234 | Replaced by rule triggers |

Also deleted:
- `src/tract/retry.py` (orchestrator dependency)
- Storage rows: HookWiringRow, TriggerLogRow, ConfigChangeRow, DynamicOpSpecRow
- Related imports from `__init__.py` and `tract.py`

**DO NOT DELETE** `src/tract/formatting.py` -- used by protocols.py, models/commit.py,
models/merge.py, operations/diff.py, operations/history.py, cli/. Only remove
`pprint_hooks()` and `list_hooks()` from it.

**Note:** `src/tract/policy/` was already removed in a prior cleanup. Nothing to
delete or keep.

**Requires rewrite (not deletion):**
- `src/tract/operations/compression.py` -- `compress_range()` returns `PendingCompress`
  from hooks. Must be rewritten to return a plain `CompressResult` dataclass.

### What Gets Kept (untouched)

| Package | Lines | Notes |
|---------|-------|-------|
| `src/tract/storage/` | 3,168 | Add new rows, keep existing |
| `src/tract/engine/` | 1,573 | Modify compiler only |
| `src/tract/operations/` | 3,934 | All graph primitives stay |
| `src/tract/llm/` | 648 | Client stays as-is |
| `src/tract/models/` | 1,369 | Add new content types |
| `src/tract/protocols.py` | 409 | Unchanged |
| `src/tract/session.py` | 1,026 | Multi-agent stays |

### What Gets Created (~4,500 lines source, ~450 tests)

| File/Package | Est. Lines | Phase |
|---|---|---|
| `src/tract/models/content.py` (modified) | +80 | R0 |
| `src/tract/engine/compiler.py` (modified) | +120 | R0 |
| `src/tract/rules/__init__.py` | 30 | R1 |
| `src/tract/rules/models.py` | 200 | R1 |
| `src/tract/rules/index.py` | 250 | R1 |
| `src/tract/rules/conditions.py` | 300 | R1 |
| `src/tract/rules/engine.py` | 500 | R2 |
| `src/tract/rules/actions.py` | 400 | R2 |
| `src/tract/rules/registries.py` | 200 | R4 |
| `src/tract/loop.py` | 250 | R3 |
| `src/tract/tract.py` (modified) | net -1,500 | R0-R3 |
| `src/tract/__init__.py` (modified) | net -80 | R0 |
| `src/tract/toolkit/` (modified) | net -200 | R3 |
| Cookbook rewrites | ~2,000 | R4 |

## Phase Dependency Graph

```
R0 (Demolition + Content Types)
 |
 v
R1 (Rule Engine Core)
 |
 v
R2 (Event Processing + Actions)
 |
 v
R3 (Default Loop + Toolkit Rewire)
 |
 v
R4 (Cookbooks + Validation + Registries)
```

Strictly sequential -- each phase builds on the previous.

## Per-Phase Plans

| Phase | Plan File | Goal |
|-------|-----------|------|
| R0 | [R0_DEMOLITION.md](R0_DEMOLITION.md) | Delete old systems, add content types, add compile strategy |
| R1 | [R1_RULE_ENGINE_CORE.md](R1_RULE_ENGINE_CORE.md) | Rule index, conditions, config resolution, t.rule() API |
| R2 | [R2_EVENT_PROCESSING.md](R2_EVENT_PROCESSING.md) | Event triggers, action handlers, execution pipeline |
| R3 | [R3_DEFAULT_LOOP.md](R3_DEFAULT_LOOP.md) | Dumb loop, toolkit rewire, transition mechanics |
| R4 | [R4_COOKBOOKS_VALIDATION.md](R4_COOKBOOKS_VALIDATION.md) | Rewrite cookbooks, POC on cheap models, registries |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| tract.py is 6,890 lines; surgery is error-prone | High | R0 does deletion only, R1-R3 add incrementally. Run surviving tests after each edit. |
| Compile strategy changes break cache | Medium | Cache key must include `(head_hash, strategy, strategy_k)` as a composite key. Current cache is head_hash-only. Task 0.8 specifies the fix. |
| Rule index rebuild on switch/merge is expensive | Low | Same invalidation pattern as compile cache. Already proven fast enough. |
| Cheap models can't follow rule-heavy system prompts | Medium | Rules are evaluated by the engine, not the LLM. LLM conditions are opt-in. |
| Cookbook rewrite scope is large (107 files) | Medium | Triage: rewrite 10 core cookbooks, delete the rest. New cookbooks for rules. |

## Success Criteria (End-to-End)

1. All surviving substrate tests pass (models, storage, engine, operations, merge, rebase, compression, GC, session, LLM)
2. Rule engine: create rules, evaluate conditions, resolve configs, process events
3. Default loop: compile -> LLM -> tools -> repeat, clean exit on block
4. Compile strategy: full/messages/adaptive(k) all work, orthogonal to compression
5. POC cookbook on Cerebras/Groq demonstrates a multi-stage workflow using rules
6. No orphaned imports, no dead code, clean `python -m pytest` on the full suite
