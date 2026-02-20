# Phase 12: LLMConfig Cleanup & Tightening - Research

**Researched:** 2026-02-19
**Domain:** Python dataclass refactoring, LLM config resolution, internal API tightening
**Confidence:** HIGH

## Summary

Phase 12 is a pure internal refactoring phase. No new libraries are needed, no external APIs are being integrated, and no schema migrations are required. The work is entirely about tightening the existing LLMConfig plumbing that was built incrementally across Phases 9-11.

The codebase is well-structured with clear patterns established in prior phases: frozen dataclasses for config, `dataclasses.replace()` for mutation safety, Pydantic auto-coercion at boundaries, and a three-level resolution chain (call > operation > tract default). Phase 12 extends these patterns -- it does not introduce new ones.

All 8 issues identified in the design document (`docs/plans/2026-02-19-phase-12-llmconfig-cleanup-design.md`) map cleanly to localized code changes in 4-6 files. The existing test suite (1011 tests) provides strong regression coverage. The risk is low because each change is independently verifiable.

**Primary recommendation:** Implement as a single plan with 8 tasks (one per issue), ordered by dependency. Issues 1-2 are foundational (other changes build on them), Issue 4 (from_dict aliases) is independent, and Issues 5-8 are downstream consumers.

## Standard Stack

No new libraries are needed. This phase uses only what is already in the codebase.

### Core (already present)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| dataclasses (stdlib) | Python 3.12+ | Frozen config dataclasses | Already used for LLMConfig |
| types.MappingProxyType (stdlib) | Python 3.12+ | Immutable dict wrapper for extra | Already used in LLMConfig.__post_init__ |
| dataclasses.replace (stdlib) | Python 3.12+ | Mutation-safe config updates | Already used in orchestrate() |

### Supporting (already present)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | 2.x | CommitInfo coercion boundaries | Already handles dict->LLMConfig at boundaries |
| pytest | 8.x | Test infrastructure | All existing tests |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| frozen dataclass for OperationConfigs | Pydantic BaseModel | Unnecessary -- LLMConfig is already a dataclass, keep consistent. TractConfig is Pydantic but OperationConfigs is runtime-only, not persisted. Prior decision 10-01-D1 locks this. |

**Installation:**
No new dependencies required.

## Architecture Patterns

### Current File Layout (no new files needed)
```
src/tract/
  models/config.py          # LLMConfig + NEW: OperationConfigs
  tract.py                  # Tract facade (most changes here)
  operations/compression.py # Thread generation_config through summary commits
  orchestrator/loop.py      # Forward full config in _call_llm()
  orchestrator/config.py    # Add max_tokens + extra_llm_kwargs to OrchestratorConfig
  __init__.py               # Export OperationConfigs
```

### Pattern 1: Frozen Dataclass with All-None Defaults
**What:** OperationConfigs follows the same pattern as LLMConfig -- all fields default to None, frozen for safety.
**When to use:** For config objects that are runtime-only (not persisted) and need IDE autocomplete.
**Example:**
```python
# Source: existing LLMConfig pattern in models/config.py
@dataclass(frozen=True)
class OperationConfigs:
    chat: LLMConfig | None = None
    merge: LLMConfig | None = None
    compress: LLMConfig | None = None
    orchestrate: LLMConfig | None = None
```

### Pattern 2: Four-Level Resolution Chain
**What:** Extends the existing three-level chain (call > operation > tract) by adding `llm_config=` between sugar params and operation config.
**When to use:** Every LLM-powered operation on Tract.
**Example:**
```python
# Resolution order for each field:
# 1. Sugar param (model=, temperature=, max_tokens=) -- highest priority
# 2. llm_config.field (if llm_config provided and field is not None)
# 3. operation_configs.{operation}.field
# 4. _default_config.field (tract-level default)
# 5. Not set (omitted from output dict)
```

### Pattern 3: Alias-Aware from_dict() Pipeline
**What:** Apply aliases, drop ignored keys, then route to known fields or extra.
**When to use:** In LLMConfig.from_dict() to handle cross-framework field names.
**Example:**
```python
# Pipeline: raw dict -> apply aliases -> drop ignored -> split known/extra
_ALIASES = {"stop": "stop_sequences", "max_completion_tokens": "max_tokens"}
_IGNORED = frozenset({"messages", "tools", "tool_choice", "stream", ...})
```

### Anti-Patterns to Avoid
- **String-keyed config dicts for known fields:** The whole point of Issue #1 is replacing `dict[str, LLMConfig]` with a typed dataclass. Do NOT keep string keys for operation names.
- **Mutating caller-provided config objects:** Always use `dataclasses.replace()`. The existing orchestrate() method already does this correctly -- maintain that pattern everywhere.
- **Partial field capture:** The whole point of Issue #2 is capturing ALL resolved fields. Do NOT add individual field handling -- capture the full resolved dict.
- **Silent failures on misconfiguration:** Issue #7 requires raising errors when LLM params are passed without an LLM client. Do NOT silently drop config.

## Don't Hand-Roll

This phase is purely internal refactoring -- there are no external problems that need off-the-shelf solutions. However, there are patterns already established in the codebase that MUST be reused:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Immutable config merging | Manual dict merging with mutation | `dataclasses.replace()` | Already established pattern (10-01-D3) |
| Dict->LLMConfig coercion | Manual type checking at every boundary | Pydantic `field_validator` on CommitInfo | Already handles this (models/commit.py:47-52) |
| Immutable extra dict | Plain dict with defensive copying | `types.MappingProxyType` | Already established in LLMConfig.__post_init__ |
| Frozen dataclass field validation | Manual `__post_init__` checks | Dataclass frozen=True + type hints | Python stdlib handles TypeError on typos |

**Key insight:** Every pattern needed for Phase 12 already exists in the codebase. The work is extending existing patterns to new locations, not inventing new ones.

## Common Pitfalls

### Pitfall 1: Breaking Backward Compatibility of configure_operations()
**What goes wrong:** Changing `configure_operations(**kwargs)` to only accept `OperationConfigs` breaks all existing code that uses keyword arguments.
**Why it happens:** Natural reflex to make a clean break when introducing the typed class.
**How to avoid:** Support BOTH calling conventions: `configure_operations(chat=LLMConfig(...))` (kwargs, backward compat) AND `configure_operations(OperationConfigs(chat=LLMConfig(...)))` (positional, new style). Detect which is being used by checking if the first positional arg is an OperationConfigs instance.
**Warning signs:** Existing tests in `test_operation_config.py::TestConfigureOperations` start failing.

### Pitfall 2: Property Return Type Change for operation_configs
**What goes wrong:** `operation_configs` property currently returns `dict[str, LLMConfig]`. Changing to `OperationConfigs` is a minor breaking change.
**Why it happens:** Callers may be indexing with `t.operation_configs["chat"]` which won't work on a dataclass.
**How to avoid:** Test files use `t.operation_configs["chat"]`. The return type MUST change (that's the purpose), but update all tests that index by string. Alternatively, could add `__getitem__` to OperationConfigs for backward compat -- but this adds complexity and defeats the purpose of typed access.
**Warning signs:** Tests with `configs["chat"]` syntax failing with TypeError.

### Pitfall 3: Circular Resolution in _resolve_llm_config
**What goes wrong:** Adding `llm_config` parameter to `_resolve_llm_config` while also having sugar params creates ambiguity about precedence.
**Why it happens:** Four levels of config are complex to reason about.
**How to avoid:** Implement field-by-field with explicit priority comments. Test each level individually. The design doc is explicit: sugar params > llm_config > operation > tract default.
**Warning signs:** Tests where sugar param doesn't override llm_config field.

### Pitfall 4: Compression Pipeline Threading
**What goes wrong:** `compress_range()` has a complex call chain (`compress_range` -> `_summarize_group` -> `_commit_compression`). Threading `llm_kwargs` to the commit creation site requires modifying multiple function signatures.
**Why it happens:** The compression pipeline was designed before generation_config tracking existed.
**How to avoid:** Add `generation_config: dict | None = None` parameter to `_commit_compression()` and thread it through from `compress_range()`. The commit site is in `_commit_compression()` at the `commit_engine.create_commit()` call (line ~665).
**Warning signs:** Summary commits still have no generation_config after the change.

### Pitfall 5: OrchestratorConfig Is Not Frozen
**What goes wrong:** OrchestratorConfig is a regular (mutable) dataclass, not frozen. Adding fields to it is safe, but the orchestrate() method must still use `dataclasses.replace()` to avoid mutating caller-provided objects.
**Why it happens:** OrchestratorConfig was intentionally made mutable ("users may adjust settings between runs" per its docstring).
**How to avoid:** Continue using `replace()` in `tract.orchestrate()` when merging operation-level config into the OrchestratorConfig. The existing code already does this (tract.py:2610-2616).
**Warning signs:** Original config object mutated after orchestrate() call.

### Pitfall 6: from_dict() Alias Collision
**What goes wrong:** If a dict has BOTH `stop` and `stop_sequences`, the alias mapping could lose data.
**Why it happens:** Cross-framework dicts may have both the native and aliased key.
**How to avoid:** Only apply alias if the canonical key is NOT already present: `if alias_target not in d and alias_source in d: d[alias_target] = d.pop(alias_source)`.
**Warning signs:** Test with both `stop` and `stop_sequences` in same dict losing one value.

## Code Examples

### OperationConfigs Dataclass (models/config.py)
```python
# Source: design doc + existing LLMConfig pattern
@dataclass(frozen=True)
class OperationConfigs:
    """Per-operation LLM configuration defaults.

    Each field corresponds to an LLM-powered operation on Tract.
    None means 'no operation-level override -- use tract default.'
    """
    chat: LLMConfig | None = None
    merge: LLMConfig | None = None
    compress: LLMConfig | None = None
    orchestrate: LLMConfig | None = None
```

### Updated _resolve_llm_config (tract.py)
```python
# Source: design doc section 3
def _resolve_llm_config(
    self,
    operation: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    llm_config: LLMConfig | None = None,
    **kwargs: object,
) -> dict:
    """Resolve effective LLM config: sugar > llm_config > operation > tract default."""
    op_config = getattr(self._operation_configs, operation, None)
    default = self._default_config

    resolved: dict = {}

    # For each known field, apply 4-level priority
    for field_name in ("model", "temperature", "max_tokens", "top_p",
                       "frequency_penalty", "presence_penalty", "top_k",
                       "seed", "stop_sequences"):
        # Level 1: sugar params (only model, temperature, max_tokens)
        sugar_val = locals().get(field_name)  # Won't work -- use explicit checks
        # ... explicit per-field resolution
    # ... (see full implementation in plan)
```

### Updated _build_generation_config (tract.py)
```python
# Source: design doc section 5
def _build_generation_config(self, response: dict, *, resolved: dict) -> dict:
    """Build generation_config from the full resolved LLM kwargs."""
    config = dict(resolved)
    # Response model is authoritative
    if "model" in response:
        config["model"] = response["model"]
    return config
```

### from_dict() with Aliases (models/config.py)
```python
# Source: design doc section 4
_ALIASES: dict[str, str] = {
    "stop": "stop_sequences",
    "max_completion_tokens": "max_tokens",
}

_IGNORED: frozenset[str] = frozenset({
    "messages", "tools", "tool_choice", "stream",
    "response_format", "n", "logprobs", "top_logprobs",
    "functions", "function_call",
    "system", "metadata",
})

@classmethod
def from_dict(cls, d: dict | None) -> LLMConfig | None:
    if d is None:
        return None
    # Step 1: copy to avoid mutating input
    d = dict(d)
    # Step 2: apply aliases (only if canonical key not present)
    for alias, canonical in _ALIASES.items():
        if alias in d and canonical not in d:
            d[canonical] = d.pop(alias)
        elif alias in d:
            del d[alias]  # canonical already present, drop alias
    # Step 3: drop ignored keys
    for key in _IGNORED:
        d.pop(key, None)
    # Step 4: split known/extra (existing logic)
    known = {f.name for f in dc_fields(cls)} - {"extra"}
    ...
```

### Compression with generation_config (operations/compression.py)
```python
# Source: design doc section 6
# In _commit_compression(), at the summary commit creation site (~line 665):
info = commit_engine.create_commit(
    content=summary_content,
    message=f"Compressed {n_commits} commits",
    generation_config=generation_config,  # NEW: thread from compress_range()
)
```

### Orchestrator Full Config Forwarding (orchestrator/loop.py)
```python
# Source: design doc section 7
# In _call_llm(), replace:
#   model=self._config.model, temperature=self._config.temperature
# with:
kwargs = {}
if self._config.model:
    kwargs["model"] = self._config.model
if self._config.temperature is not None:
    kwargs["temperature"] = self._config.temperature
if self._config.max_tokens is not None:
    kwargs["max_tokens"] = self._config.max_tokens
if self._config.extra_llm_kwargs:
    kwargs.update(self._config.extra_llm_kwargs)
return client.chat(messages, **kwargs, tools=tools)
```

## State of the Art

No external library changes apply. This is internal refactoring.

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `dict[str, LLMConfig]` for operation configs | `OperationConfigs` frozen dataclass | Phase 12 | Typos caught at construction, IDE autocomplete |
| `_default_model: str \| None` | `_default_config: LLMConfig \| None` | Phase 12 | All 9 fields available as tract defaults |
| 3-field `_build_generation_config` | Full resolved dict capture | Phase 12 | query_by_config works for all fields |
| No generation_config on compression summaries | Summary commits record LLM config | Phase 12 | Audit trail for compression |

## Open Questions

### 1. Should configure_operations() accept both OperationConfigs and **kwargs?
- **What we know:** Existing code uses `configure_operations(chat=LLMConfig(...))`. Tests rely on this.
- **What's unclear:** Should we support both `configure_operations(OperationConfigs(...))` and `configure_operations(chat=...)` or deprecate one?
- **Recommendation:** Support both. Check if first positional arg is OperationConfigs; if so, use it directly. If kwargs provided, construct OperationConfigs from them. This maintains backward compat and adds the typed path.

### 2. Should operation_configs property change return type?
- **What we know:** Currently returns `dict[str, LLMConfig]`. Tests index with string keys.
- **What's unclear:** Is `OperationConfigs` sufficient, or do callers need dict-like access?
- **Recommendation:** Return `OperationConfigs` directly. It is more useful for type checking. Tests that use string indexing (6 instances in test_operation_config.py) need updating. Could add `__getitem__` to OperationConfigs as a bridge, but it adds complexity.

### 3. Should from_obj() use __dict__, dataclass fields, or Pydantic model_dump?
- **What we know:** Design says "extracts from any object with `__dict__` or dataclass fields."
- **What's unclear:** How robust does cross-framework extraction need to be?
- **Recommendation:** Use `dataclasses.fields()` if it is a dataclass, `model_dump()` if it is a Pydantic model, otherwise `vars(obj)`. Then pipe through `from_dict()`. Keep it simple -- this is a convenience utility, not a full serialization framework.

### 4. Should Tract.open() accept default_config= in addition to model=?
- **What we know:** Design says if both provided, raise ValueError.
- **What's unclear:** Is the complexity of two paths worth it?
- **Recommendation:** Yes, add `default_config: LLMConfig | None = None`. The model= sugar is too convenient to remove (most users just want to set the model). The ValueError on conflict is the right guard.

## Detailed Change Inventory

### File: src/tract/models/config.py
1. Add `OperationConfigs` frozen dataclass (4 fields)
2. Add `_ALIASES` and `_IGNORED` module-level constants
3. Update `from_dict()` to apply aliases and drop ignored keys
4. Add `from_obj()` classmethod

### File: src/tract/tract.py
1. Replace `_default_model: str | None` with `_default_config: LLMConfig | None`
2. Replace `_operation_configs: dict[str, LLMConfig]` with `_operation_configs: OperationConfigs`
3. Update `Tract.open()` to accept `default_config=` and create `_default_config` from `model=`
4. Update `Tract.open()` to accept `operations=OperationConfigs` (keep `operation_configs=dict` for compat)
5. Rewrite `_resolve_llm_config()` with 4-level chain and `llm_config=` parameter
6. Rewrite `_build_generation_config()` to capture full resolved dict
7. Add `llm_config=` parameter to `generate()`, `chat()`, `merge()`, `compress()`
8. Update `configure_operations()` to support both OperationConfigs and **kwargs
9. Update `operation_configs` property to return OperationConfigs
10. Add error in `compress()` when explicit LLM params provided without LLM client
11. Update `orchestrate()` to forward all config fields

### File: src/tract/operations/compression.py
1. Thread `generation_config` parameter through `compress_range()` -> `_commit_compression()`
2. Set `generation_config=` on summary commit creation in `_commit_compression()`

### File: src/tract/orchestrator/config.py
1. Add `max_tokens: int | None = None` to OrchestratorConfig
2. Add `extra_llm_kwargs: dict | None = None` to OrchestratorConfig

### File: src/tract/orchestrator/loop.py
1. Update `_call_llm()` to forward max_tokens and extra_llm_kwargs

### File: src/tract/__init__.py
1. Export `OperationConfigs`

### File: tests/test_operation_config.py
1. Update all tests using `operation_configs["chat"]` dict syntax
2. Add tests for OperationConfigs dataclass
3. Add tests for from_dict() aliases and ignored keys
4. Add tests for from_obj()
5. Add tests for llm_config= parameter on chat/generate/merge/compress
6. Add tests for full generation_config capture
7. Add tests for compression summary generation_config
8. Add tests for orchestrator full config forwarding
9. Add tests for compress() error on LLM params without client

## Dependency Order

The 8 issues have the following dependencies:

```
Issue 1 (OperationConfigs)  ---> Issue 3 (_resolve_llm_config accepts llm_config=)
Issue 2 (Consolidate default) ---> Issue 5 (_build_generation_config captures all)
Issue 4 (from_dict aliases)  ---> independent
Issue 5 (full capture)       ---> depends on Issues 1, 2
Issue 6 (compression config) ---> depends on Issue 5 (for consistency)
Issue 7 (orchestrator fixes) ---> depends on Issues 1, 5
Issue 8 (compress error)     ---> depends on Issue 3 (llm_config= exists)
```

**Recommended execution order:**
1. Issue 4 (from_dict aliases) -- independent, no dependencies
2. Issue 1 (OperationConfigs) -- foundational for typed access
3. Issue 2 (consolidate default) -- foundational for config resolution
4. Issue 3 (call-level llm_config=) -- extends resolution chain
5. Issue 5 (full capture) -- uses resolved chain from Issues 1-3
6. Issue 6 (compression config) -- downstream consumer
7. Issue 7 (orchestrator fixes) -- downstream consumer
8. Issue 8 (compress error) -- simple guard, depends on llm_config= existing

## Sources

### Primary (HIGH confidence)
- `src/tract/models/config.py` -- Current LLMConfig implementation (read directly)
- `src/tract/tract.py` -- Current Tract facade, all affected methods (read directly)
- `src/tract/operations/compression.py` -- Compression pipeline (read directly)
- `src/tract/orchestrator/loop.py` -- Orchestrator _call_llm() (read directly)
- `src/tract/orchestrator/config.py` -- OrchestratorConfig (read directly)
- `tests/test_operation_config.py` -- Existing tests for LLMConfig and operations (read directly)
- `docs/plans/2026-02-19-phase-12-llmconfig-cleanup-design.md` -- Approved design document (read from git)

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md` -- Phase 12 requirements and success criteria (read directly)

### Tertiary (LOW confidence)
None -- all findings are from direct code inspection and the approved design document.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries needed, all patterns exist in codebase
- Architecture: HIGH -- design document specifies exact changes, code inspected
- Pitfalls: HIGH -- identified from direct code analysis of affected files
- Code examples: HIGH -- based on design document and existing patterns

**Research date:** 2026-02-19
**Valid until:** N/A -- this is internal refactoring, not subject to external library changes
