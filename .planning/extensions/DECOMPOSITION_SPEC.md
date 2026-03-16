# Tract Sub-Object Decomposition Spec

## Overview

Decompose the 8,278-line `Tract` class (~160 methods) into 15 sub-objects accessed via composition: `t.tags.add(...)`, `t.branches.create(...)`, `t.llm.chat(...)`, etc. Each sub-object is a real class with explicit constructor dependencies, testable in isolation.

---

## Sub-Object Table

| Accessor | Class | File | Method Count | Est. Lines |
|---|---|---|---|---|
| `t.tags` | `TagManager` | `managers/tags.py` | 9 | ~300 |
| `t.branches` | `BranchManager` | `managers/branches.py` | 7 | ~250 |
| `t.annotations` | `AnnotationManager` | `managers/annotations.py` | 4 | ~150 |
| `t.routing` | `RoutingManager` | `managers/routing.py` | 8 (+1 async) | ~350 |
| `t.search` | `SearchManager` | `managers/search.py` | 16 | ~700 |
| `t.llm` | `LLMManager` | `managers/llm.py` | 22 (+7 async) | ~1200 |
| `t.compression` | `CompressionManager` | `managers/compression.py` | 11 (+2 async) | ~800 |
| `t.middleware` | `MiddlewareManager` | `managers/middleware.py` | 8 | ~350 |
| `t.config` | `ConfigManager` | `managers/config.py` | 14 | ~500 |
| `t.tools` | `ToolManager` | `managers/tools.py` | 9 | ~350 |
| `t.toolkit` | `ToolkitManager` | `managers/toolkit.py` | 10 | ~350 |
| `t.intelligence` | `IntelligenceManager` | `managers/intelligence.py` | 5 (+5 async) | ~300 |
| `t.templates` | `TemplateManager` | `managers/templates.py` | 10 | ~250 |
| `t.persistence` | `PersistenceManager` | `managers/persistence.py` | 12 | ~500 |
| `t.spawn` | `SpawnManager` | `managers/spawn.py` | 4 | ~120 |

**Total sub-object lines: ~5,970** (vs 8,278 current — remainder stays on Tract)

---

## Per Sub-Object Detail

### 1. TagManager (`t.tags`)

**Methods:**
- `add(target_hash, tag_name)` — was `tag()`
- `remove(target_hash, tag_name)` — was `untag()`
- `get(target_hash)` — was `get_tags()`
- `register(name, description)` — was `register_tag()`
- `list()` — was `list_tags()`
- `query(tags, match, limit)` — was `query_by_tags()`
- `_seed_base()` — was `_seed_base_tags()`
- `_validate(tags)` — was `_validate_tags()`
- `_classify(content_type, role, operation, metadata)` — was `_classify_tags()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    tag_annotation_repo: SqliteTagAnnotationRepository,
    tag_registry_repo: SqliteTagRegistryRepository,
    commit_repo: SqliteCommitRepository,
    blob_repo: SqliteBlobRepository,
    annotation_repo: SqliteAnnotationRepository,
    parent_repo: SqliteCommitParentRepository | None,
    strict_tags: bool,
    check_open: Callable,
):
```

**State owned:** `_strict_tags: bool`

**Cross-deps:** None (leaf node). Called BY `Tract.commit()` for auto-classification.

---

### 2. BranchManager (`t.branches`)

**Methods:**
- `create(name, source)` — was `branch()`
- `switch(target)` — was `switch()`
- `checkout(target)` — was `checkout()`
- `reset(target)` — was `reset()`
- `list()` — was `list_branches()`
- `delete(name, force)` — was `delete_branch()`
- `resolve(ref_or_prefix)` — was `resolve_commit()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    ref_repo: SqliteRefRepository,
    commit_repo: SqliteCommitRepository,
    parent_repo: SqliteCommitParentRepository | None,
    cache: CacheManager,
    check_open: Callable,
    commit_session: Callable,
):
```

**State owned:** None (stateless delegation to repos/operations).

**Cross-deps:** None (leaf node). Called BY `Tract.transition()`, `SearchManager`, `PersistenceManager`.

---

### 3. AnnotationManager (`t.annotations`)

**Methods:**
- `set(target_hash, priority, reason)` — was `annotate()`
- `get(target_hash)` — was `get_annotations()`
- `counts(limit)` — was `annotation_counts()`
- `_enrich_with_priorities(entries)` — was `_enrich_with_priorities()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    annotation_repo: SqliteAnnotationRepository,
    commit_repo: SqliteCommitRepository,
    check_open: Callable,
    commit_session: Callable,
):
```

**State owned:** None.

**Cross-deps:** None (leaf node). Called BY `SearchManager._enrich`, `LLMManager` (SKIP annotations on retries).

---

### 4. RoutingManager (`t.routing`)

**Methods:**
- `add(name, description, route_type, keywords, pattern)` — was `add_route()`
- `remove(name)` — was `remove_route()`
- `route(query, router, apply)` — was `route()`
- `aroute(query, router, apply)` — was `aroute()`
- `_fallback(query)` — was `_route_fallback()`
- `_apply_result(result, apply)` — was `_route_apply()`
- `_ensure_table()` — was `_ensure_routing_table()`
- `_apply(route)` — was `_apply_route()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    ref_repo: SqliteRefRepository,
    commit_repo: SqliteCommitRepository,
    config_index: Callable[[], ConfigIndex],  # lazy accessor
    check_open: Callable,
    # Callbacks for apply actions:
    switch_branch: Callable,
    apply_stage: Callable,
    commit_session: Callable,
):
```

**State owned:** `_routing_table: RoutingTable | None`

**Cross-deps:** Calls `BranchManager.switch()` and `TemplateManager.apply_stage()` via callbacks.

---

### 5. SearchManager (`t.search`)

**Methods:**
- `log(limit, branch, include_metadata, content_types, since, until)` — was `log()`
- `find(content_types, tags, metadata, predicate, branch, limit, since, until)` — was `find()`
- `find_one(...)` — was `find_one()`
- `query_by_config(key, value, branch, limit)` — was `query_by_config()`
- `diff(from_ref, to_ref, content_types, branch)` — was `diff()`
- `compare(source_branch, target_branch)` — was `compare()`
- `skipped(limit)` — was `skipped()`
- `pinned(limit)` — was `pinned()`
- `status()` — was `status()`
- `health()` — was `health()`
- `manifest(max_log_entries)` — was `manifest()`
- `edit_history(commit_hash)` — was `edit_history()`
- `restore(commit_hash, message, tags)` — was `restore()`
- `get_commit(commit_hash)` — was `get_commit()`
- `get_content(commit_or_hash)` — was `get_content()`
- `get_metadata(commit_or_hash)` — was `get_metadata()`
- `show(commit_or_hash)` — was `show()`

Note: `_compile_at()` moves here too (used by `diff`/`compare`).

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    commit_repo: SqliteCommitRepository,
    blob_repo: SqliteBlobRepository,
    ref_repo: SqliteRefRepository,
    annotation_repo: SqliteAnnotationRepository,
    parent_repo: SqliteCommitParentRepository | None,
    event_repo: SqliteOperationEventRepository | None,
    token_counter: TokenCounter,
    compiler: ContextCompiler,
    config: TractConfig,
    config_index: Callable[[], ConfigIndex],  # lazy
    custom_type_registry: dict,
    check_open: Callable,
    # For _enrich_with_priorities:
    enrich: Callable,
    # For _compile_at:
    compile_fn: Callable,
):
```

**State owned:** None (reads only).

**Cross-deps:** Calls `AnnotationManager._enrich_with_priorities()` via callback. Uses `compile()` for diff/compare.

---

### 6. LLMManager (`t.llm`)

**Methods:**
- `chat(prompt, ...)` — was `chat()`
- `achat(prompt, ...)` — was `achat()`
- `generate(...)` — was `generate()`
- `agenerate(...)` — was `agenerate()`
- `revise(target_hash, ...)` — was `revise()`
- `arevise(target_hash, ...)` — was `arevise()`
- `run(prompt, ...)` — was `run()`
- `arun(prompt, ...)` — was `arun()`
- `_generate_once(...)` — was `_generate_once()`
- `_agenerate_once(...)` — was `_agenerate_once()`
- `_generate_once_pre(...)` — was `_generate_once_pre()`
- `_generate_once_post(...)` — was `_generate_once_post()`
- `_generate_pre()` — was `_generate_pre()`
- `_generate_validate_loop(...)` — was `_generate_validate_loop()`
- `_resolve_llm_config(...)` — was `_resolve_llm_config()`
- `_build_generation_config(...)` — was `_build_generation_config()`
- `_extract_content(...)` — was `_extract_content()`
- `_extract_usage(...)` — was `_extract_usage()`
- `_revise_post(...)` — was `_revise_post()`
- `_resolve_llm_client(operation)` — was `_resolve_llm_client()`
- `_has_llm_client(operation)` — was `_has_llm_client()`
- `_resolve_resolver(resolver, operation)` — was `_resolve_resolver()`
- `_auto_message(content_type, text)` — was `_auto_message()`
- `_improve_commit(...)` — was `_improve_commit()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    ref_repo: SqliteRefRepository,
    commit_repo: SqliteCommitRepository,
    blob_repo: SqliteBlobRepository,
    annotation_repo: SqliteAnnotationRepository,
    parent_repo: SqliteCommitParentRepository | None,
    config: TractConfig,
    token_counter: TokenCounter,
    # Shared LLM state (owned, mutable):
    llm_state: LLMState,  # see below
    # Callbacks to Tract:
    check_open: Callable,
    commit_fn: Callable,  # Tract.commit
    system_fn: Callable,  # Tract.system
    user_fn: Callable,    # Tract.user
    assistant_fn: Callable, # Tract.assistant
    compile_fn: Callable, # Tract.compile
    annotate_fn: Callable, # AnnotationManager.set
    run_middleware: Callable,
    record_usage: Callable,
    get_tools: Callable,  # ToolManager.get_tools
    commit_session: Callable,
):
```

**LLMState (shared mutable bag):**
```python
@dataclass
class LLMState:
    llm_client: LLMClient | None
    default_config: LLMConfig | None
    operation_configs: OperationConfigs
    operation_prompts: OperationPrompts
    operation_clients: OperationClients
    retry_config: RetryConfig | None
    default_resolver: ResolverCallable | None
    commit_reasoning: bool
    auto_message_enabled: bool
    tool_summarization_config: ToolSummarizationConfig | None
    owns_llm_client: bool
```

**Cross-deps:** Heavy. Calls `Tract.commit/system/user/assistant/compile`, `AnnotationManager.set`, `MiddlewareManager._run`, `ToolManager.get_tools`, `ConfigManager.record_usage`. This is the most interconnected sub-object.

---

### 7. CompressionManager (`t.compression`)

**Methods:**
- `compress(...)` — was `compress()`
- `acompress(...)` — was `acompress()`
- `compress_tool_calls(...)` — was `compress_tool_calls()`
- `acompress_tool_calls(...)` — was `acompress_tool_calls()`
- `gc(...)` — was `gc()`
- `_compress_pre(...)` — was `_compress_pre()`
- `_compress_finalize(...)` — was `_compress_finalize()`
- `_compress_sliding_window(...)` — was `_compress_sliding_window()`
- `_compress_tool_calls_pre(...)` — was `_compress_tool_calls_pre()`
- `_compress_tool_calls_post(...)` — was `_compress_tool_calls_post()`
- `record_usage(...)` — was `record_usage()`

Note: `record_usage` goes here because it's tightly coupled with compression tracking.

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    commit_repo: SqliteCommitRepository,
    blob_repo: SqliteBlobRepository,
    ref_repo: SqliteRefRepository,
    annotation_repo: SqliteAnnotationRepository,
    parent_repo: SqliteCommitParentRepository | None,
    event_repo: SqliteOperationEventRepository | None,
    compile_record_repo: SqliteCompileRecordRepository | None,
    token_counter: TokenCounter,
    commit_engine: CommitEngine,
    compiler: ContextCompiler,
    config: TractConfig,
    cache: CacheManager,
    llm_state: LLMState,  # shared with LLMManager
    check_open: Callable,
    commit_fn: Callable,
    compile_fn: Callable,
    run_middleware: Callable,
    commit_session: Callable,
    resolve_llm_config: Callable,  # from LLMManager
    resolve_llm_client: Callable,  # from LLMManager
    has_llm_client: Callable,      # from LLMManager
):
```

**Cross-deps:** Calls `LLMManager._resolve_llm_config/client`, `Tract.commit`, `Tract.compile`, `MiddlewareManager._run`.

---

### 8. MiddlewareManager (`t.middleware`)

**Methods:**
- `add(event, handler)` — was `add_middleware()`; also aliased as `use()`
- `remove(handler_id)` — was `remove_middleware()`
- `gate(name, event, criteria, ...)` — was `gate()`
- `remove_gate(name)` — was `remove_gate()`
- `list_gates()` — was `list_gates()`
- `maintain(name, event, criteria, actions, ...)` — was `maintain()`
- `remove_maintainer(name)` — was `remove_maintainer()`
- `list_maintainers()` — was `list_maintainers()`
- `_run(event, **kwargs)` — was `_run_middleware()`

**Constructor:**
```python
def __init__(
    self,
    check_open: Callable,
):
```

**State owned:**
- `_middleware: dict[str, list[tuple[str, Callable]]]`
- `_in_middleware_events: set[str]`
- `_gates: dict[str, str]`
- `_maintainers: dict[str, str]`

**Cross-deps:** None outgoing. Called BY almost everything (pre/post events). Pure event dispatch.

---

### 9. ConfigManager (`t.config`)

**Methods:**
- `get(key, default)` — was `get_config()`
- `get_all()` — was `get_all_configs()`
- `set(**settings)` — was `configure()`
- `configure_llm(client, resolver)` — was `configure_llm()`
- `configure_operations(...)` — was `configure_operations()`
- `configure_clients(...)` — was `configure_clients()`
- `configure_prompts(...)` — was `configure_prompts()`
- `configure_tool_summarization(...)` — was `configure_tool_summarization()`
- `history(key, limit)` — was `config_history()`
- `_log_change(...)` — was `_log_config_change()`
- `_serialize_operation_configs()` — was `_serialize_operation_configs()`
- `_serialize_prompts()` — was `_serialize_prompts()`

**Properties:**
- `operation_configs` — was `operation_configs`
- `operation_prompts` — was `operation_prompts`
- `operation_clients` — was `operation_clients`
- `llm_client` — was `llm_client`
- `default_config` — was `default_config`
- `retry_config` — was `retry_config`
- `commit_reasoning` — was `commit_reasoning`
- `tool_summarization_config` — was `tool_summarization_config`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    commit_engine: CommitEngine,
    ref_repo: SqliteRefRepository,
    event_repo: SqliteOperationEventRepository | None,
    config: TractConfig,
    config_index: ConfigIndex | None,
    llm_state: LLMState,  # shared mutable bag
    check_open: Callable,
    commit_session: Callable,
):
```

**State owned:** Mutates `LLMState` (shared with LLM/Compression managers), owns `_config_index`.

**Cross-deps:** None outgoing. Mutates shared `LLMState`. Called BY `open()`, `LLMManager`, `CompressionManager`.

---

### 10. ToolManager (`t.tools`)

**Methods:**
- `set(tools)` — was `set_tools()`
- `get()` — was `get_tools()`
- `get_for_commit(commit_hash)` — was `get_commit_tools()`
- `find_results(name, limit, branch)` — was `find_tool_results()`
- `find_calls(name, limit, branch)` — was `find_tool_calls()`
- `find_turns(limit, branch)` — was `find_tool_turns()`
- `drop_failed_turns(branch, limit)` — was `drop_failed_tool_turns()`
- `_store_and_link(commit_hash, tools)` — was `_store_and_link_tools()`
- `_gather_for_compile()` — was `_gather_tools_for_compile()`
- `_inject(result)` — was `_inject_tools()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    commit_repo: SqliteCommitRepository,
    blob_repo: SqliteBlobRepository,
    ref_repo: SqliteRefRepository,
    parent_repo: SqliteCommitParentRepository | None,
    annotation_repo: SqliteAnnotationRepository,
    tool_schema_repo: SqliteToolSchemaRepository | None,
    check_open: Callable,
    commit_session: Callable,
):
```

**State owned:** `_active_tools: list[dict] | None`

**Cross-deps:** None outgoing. Called BY `Tract.commit()` (tool linking), `Tract.compile()` (tool injection), `LLMManager` (tool gathering for run loop).

---

### 11. ToolkitManager (`t.toolkit`)

**Methods:**
- `as_tools(profile, tool_names, overrides, format)` — was `as_tools()`
- `as_callable_tools(profile, tool_names, overrides)` — was `as_callable_tools()`
- `switch_profile(profile)` — was `switch_profile()`
- `lock(tool_name)` — was `lock_tool()`
- `unlock(tool_name)` — was `unlock_tool()`
- `register(fn, name, description)` — was `tool()` decorator
- `remove(tool_name)` — was `remove_tool()`
- `custom_tools` (property) — was `custom_tools`
- `_resolve(profile, tool_names, overrides)` — was `_resolve_tools()`

**Constructor:**
```python
def __init__(
    self,
    tract: Tract,  # needed by get_all_tools/get_compact_tools
    check_open: Callable,
):
```

Note: `get_all_tools(tract)` and `get_compact_tools(tract)` in `toolkit/definitions.py` and `toolkit/compact.py` take the full tract reference. This is the ONE sub-object that needs a back-reference to `Tract` (or we refactor those functions to take a protocol).

**State owned:**
- `_tool_profile: str | ToolProfile | None`
- `_tool_executor: ToolExecutor | None`
- `_custom_tools: dict[str, Any]`
- `_tool_result_format: str`

**Cross-deps:** Depends on `Tract` (back-ref for toolkit definitions). Called BY `LLMManager.run()`.

---

### 12. IntelligenceManager (`t.intelligence`)

**Methods:**
- `cherry_pick(source, commit_hash, ...)` — was `cherry_pick()`
- `acherry_pick(...)` — was `acherry_pick()`
- `deduplicate(...)` — was `deduplicate()`
- `adeduplicate(...)` — was `adeduplicate()`
- `auto_split(commit_hash, ...)` — was `auto_split()`
- `aauto_split(...)` — was `aauto_split()`
- `auto_rebase(...)` — was `auto_rebase()`
- `aauto_rebase(...)` — was `aauto_rebase()`
- `auto_branch(context, ...)` — was `auto_branch()`
- `aauto_branch(...)` — was `aauto_branch()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    commit_repo: SqliteCommitRepository,
    blob_repo: SqliteBlobRepository,
    ref_repo: SqliteRefRepository,
    annotation_repo: SqliteAnnotationRepository,
    parent_repo: SqliteCommitParentRepository | None,
    commit_engine: CommitEngine,
    config: TractConfig,
    check_open: Callable,
    resolve_llm_client: Callable,
    resolve_llm_config: Callable,
    commit_session: Callable,
):
```

**Cross-deps:** Calls `LLMManager._resolve_llm_client/config` via callbacks.

---

### 13. TemplateManager (`t.templates`)

**Methods:**
- `apply(name, **params)` — was `apply_template()`
- `register(template)` — was `register_template()`
- `get(name)` — was `get_template()`
- `list()` — was `list_templates()`
- `load_profile(name, apply_directives)` — was `load_profile()`
- `apply_stage(stage_name)` — was `apply_stage()`
- `active_profile` (property) — was `active_profile`
- `register_profile(profile)` — was `register_profile()`
- `get_profile(name)` — was `get_profile()`
- `list_profiles()` — was `list_profiles()`

**Constructor:**
```python
def __init__(
    self,
    check_open: Callable,
    directive_fn: Callable,   # Tract.directive
    configure_fn: Callable,   # ConfigManager.set
):
```

**State owned:**
- `_template_registry: dict`
- `_profile_registry: dict`
- `_active_profile: WorkflowProfile | None`

**Cross-deps:** Calls `Tract.directive()` and `ConfigManager.set()` via callbacks.

---

### 14. PersistenceManager (`t.persistence`)

**Methods:**
- `snapshot(label, metadata)` — was `snapshot()`
- `list_snapshots()` — was `list_snapshots()`
- `restore_snapshot(tag_or_label, create_branch)` — was `restore_snapshot()`
- `export_state(include_blobs)` — was `export_state()`
- `load_state(state)` — was `load_state()`
- `compile_records(limit)` — was `compile_records()`
- `compile_record_commits(record_id)` — was `compile_record_commits()`
- `token_checkpoints(limit)` — was `token_checkpoints()`
- `persist_behavioral_spec(...)` — was `persist_behavioral_spec()`
- `load_behavioral_specs(...)` — was `load_behavioral_specs()`
- `list_behavioral_specs(...)` — was `list_behavioral_specs()`
- `remove_behavioral_spec(...)` — was `remove_behavioral_spec()`
- `save_workflow(...)` — was `save_workflow()`
- `_load_persisted_state()` — was `_load_persisted_state()`
- `_ensure_tract_dir(subdir)` — was `_ensure_tract_dir()`
- `tract_dir` (property) — was `tract_dir`
- `quarantined` (property) — was `quarantined`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    commit_repo: SqliteCommitRepository,
    blob_repo: SqliteBlobRepository,
    ref_repo: SqliteRefRepository,
    annotation_repo: SqliteAnnotationRepository,
    parent_repo: SqliteCommitParentRepository | None,
    event_repo: SqliteOperationEventRepository | None,
    compile_record_repo: SqliteCompileRecordRepository | None,
    persistence_repo: SqlitePersistenceRepository | None,
    behavioral_spec_repo: SqliteBehavioralSpecRepository | None,
    config: TractConfig,
    custom_type_registry: dict,
    db_path: str,
    check_open: Callable,
    # Callbacks for snapshot/restore:
    commit_fn: Callable,
    log_fn: Callable,
    branch_fn: Callable,
    switch_fn: Callable,
    reset_fn: Callable,
    register_tag_fn: Callable,
    annotate_fn: Callable,
    list_branches_fn: Callable,
):
```

**State owned:**
- `_persistence_repo`
- `_behavioral_spec_repo`
- `_quarantined: list[str]`
- `_db_path: str`

**Cross-deps:** Calls `BranchManager.create/switch/reset/list`, `TagManager.register`, `AnnotationManager.set`, `Tract.commit`, `SearchManager.log`.

---

### 15. SpawnManager (`t.spawn`)

**Methods:**
- `parent()` — was `parent()`
- `children()` — was `children()`
- `send_to_child(child_tract_id, content, **kwargs)` — was `send_to_child()`
- `transition(target, handoff, **kwargs)` — was `transition()`

**Constructor:**
```python
def __init__(
    self,
    tract_id: str,
    spawn_repo: SqliteSpawnPointerRepository | None,
    check_open: Callable,
    session_owner: object | None,
    # Callbacks for transition:
    compile_fn: Callable,
    get_config_fn: Callable,
    list_branches_fn: Callable,
    switch_fn: Callable,
    system_fn: Callable,
    run_middleware: Callable,
    commit_session: Callable,
):
```

**State owned:** `_spawn_repo` reference.

**Cross-deps:** `transition()` calls `BranchManager.switch/list`, `Tract.compile`, `Tract.system`, `MiddlewareManager._run`.

---

## What Stays on Tract

These methods remain directly on the `Tract` class because they are lifecycle/cross-cutting:

### Lifecycle
- `__init__()`, `open()`, `from_components()`, `close()`, `aclose()`
- `__enter__`, `__exit__`, `__aenter__`, `__aexit__`, `__repr__`
- `_check_open()`

### Properties
- `tract_id`, `head`, `config`, `is_detached`, `current_branch`, `config_index`
- `spawn_repo` (internal, used by Session)

### Core Commit Operations (cross-cutting — touch tags, tools, middleware, config, annotations)
- `commit(content, ...)` — orchestrates tags, tools, middleware, auto-message
- `system(content, ...)`, `user(content, ...)`, `assistant(content, ...)` — convenience wrappers
- `reasoning(content, ...)`, `tool_result(content, ...)`
- `metadata(kind, data, ...)`
- `_commit_dialogue(role, content, ...)` — shared logic for system/user/assistant

### Complex Cross-Cutting Operations
- `compile(...)` — touches compiler, tools, annotations, middleware, records
- `merge(...)` — touches branches, LLM, middleware, commit engine
- `commit_merge(...)` — merge commit finalization
- `rebase(...)` — branches, commit engine, cache
- `import_commit(...)` — cross-branch commit import
- `directive(...)` — creates config commits, used by templates

### Infrastructure
- `batch()` — context manager for deferred session commits
- `register_content_type(name, model)` — custom type registry
- `_commit_session()` — flush SQLAlchemy session
- `_get_merge_aware_ancestors(...)` — DAG walking helper used by compile
- `_reorder_compiled(...)` — compile post-processing
- `_save_compile_record(...)` — compile record bookkeeping
- `_normalize_usage_dict(...)` — usage normalization

**Estimated Tract-retained lines: ~2,300** (lifecycle + commit + compile + merge/rebase + infrastructure)

---

## Cross-Dependency Map

```
                    ┌──────────┐
                    │  Tract   │ (lifecycle, commit, compile, merge, rebase)
                    └────┬─────┘
                         │ owns all sub-objects
         ┌───────────────┼───────────────────────────┐
         │               │                           │
    ┌────▼────┐   ┌──────▼──────┐   ┌───────────────▼──────────┐
    │  Tags   │   │  Branches   │   │     Middleware            │
    │ (leaf)  │   │  (leaf)     │   │  (leaf — event dispatch)  │
    └─────────┘   └─────────────┘   └──────────────────────────┘
         ▲               ▲                    ▲
         │               │                    │ called by nearly everything
    ┌────┴────┐   ┌──────┴──────┐   ┌────────┴────────┐
    │Annotate │   │   Config    │   │     Search       │
    │ (leaf)  │   │ (mutates    │   │  (read-only)     │
    └─────────┘   │  LLMState)  │   └─────────────────┘
         ▲        └─────────────┘
         │               ▲
    ┌────┴────────┐      │
    │    LLM      │──────┘  reads LLMState
    │ (heaviest)  │──────── calls Tract.commit/compile/annotate
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Compression │──── calls LLM._resolve_*, Tract.commit/compile
    └─────────────┘

    ┌───────────┐     ┌───────────┐     ┌─────────────┐
    │  Routing  │     │ Templates │     │Intelligence │
    │ → Branch  │     │ → Config  │     │ → LLM       │
    │ → Template│     │ → Tract   │     └─────────────┘
    └───────────┘     └───────────┘

    ┌───────────┐     ┌───────────┐     ┌─────────────┐
    │   Tools   │     │  Toolkit  │     │ Persistence │
    │  (leaf)   │     │ → Tract*  │     │ → Branch    │
    └───────────┘     └───────────┘     │ → Tag       │
                       *back-ref        │ → Search    │
                                        └─────────────┘

    ┌───────────┐
    │   Spawn   │ → Branch, Middleware, Tract.compile/system
    └───────────┘
```

### Dependency direction summary

| Sub-object | Depends on (calls into) |
|---|---|
| Tags | -- (leaf) |
| Branches | -- (leaf) |
| Annotations | -- (leaf) |
| Middleware | -- (leaf, event dispatch) |
| Tools | -- (leaf) |
| Config | LLMState (shared write) |
| Search | Annotations (enrich), Tract.compile |
| Templates | Tract.directive, Config.set |
| Routing | Branches.switch, Templates.apply_stage |
| Spawn | Branches, Middleware, Tract.compile/system |
| LLM | Tract.commit/system/user/assistant/compile, Annotations, Middleware, Tools, Config (LLMState read) |
| Compression | LLM (resolve), Tract.commit/compile, Middleware |
| Intelligence | LLM (resolve) |
| Toolkit | Tract (back-ref for definition generation) |
| Persistence | Branches, Tags, Annotations, Search.log, Tract.commit |

---

## Shared State: LLMState

The biggest challenge is LLM configuration state, which is read by `LLMManager`, `CompressionManager`, and `IntelligenceManager`, and written by `ConfigManager`. Rather than passing 10+ individual fields to each constructor, we extract a shared `LLMState` dataclass:

```python
@dataclass
class LLMState:
    """Shared mutable LLM configuration state."""
    llm_client: LLMClient | None = None
    default_config: LLMConfig | None = None
    operation_configs: OperationConfigs = field(default_factory=OperationConfigs)
    operation_prompts: OperationPrompts = field(default_factory=OperationPrompts)
    operation_clients: OperationClients = field(default_factory=OperationClients)
    retry_config: RetryConfig | None = None
    default_resolver: ResolverCallable | None = None
    commit_reasoning: bool = True
    auto_message_enabled: bool = False
    tool_summarization_config: ToolSummarizationConfig | None = None
    owns_llm_client: bool = False
```

This lives in `src/tract/managers/state.py` and is created by `Tract.__init__`, then passed to LLM/Compression/Intelligence/Config managers.

---

## Callback Pattern

Cross-dependencies are wired via **callbacks** (plain `Callable` references), not via sub-object-to-sub-object references. This:
1. Prevents circular imports
2. Makes dependencies explicit in constructors
3. Allows testing with stubs

Example wiring in `Tract.__init__`:
```python
self._llm = LLMManager(
    ...,
    commit_fn=self.commit,
    compile_fn=self.compile,
    annotate_fn=self.annotations.set,
    run_middleware=self.middleware._run,
    ...
)
```

---

## File Layout

```
src/tract/
  managers/
    __init__.py          # re-exports all manager classes
    state.py             # LLMState dataclass (~30 lines)
    tags.py              # TagManager
    branches.py          # BranchManager
    annotations.py       # AnnotationManager
    routing.py           # RoutingManager
    search.py            # SearchManager
    llm.py               # LLMManager
    compression.py       # CompressionManager
    middleware.py         # MiddlewareManager
    config.py            # ConfigManager
    tools.py             # ToolManager
    toolkit.py           # ToolkitManager
    intelligence.py      # IntelligenceManager
    templates.py         # TemplateManager
    persistence.py       # PersistenceManager
    spawn.py             # SpawnManager
  tract.py               # Tract class (~2,300 lines, down from 8,278)
```

---

## Implementation Order

1. **Leaf nodes first** (no outgoing deps): Tags, Branches, Annotations, Middleware, Tools
2. **Low-dep managers**: Config (writes LLMState), Search (read-only + enrich callback), Templates, Spawn
3. **High-dep managers**: LLM (heaviest cross-cutting), Compression (depends on LLM), Intelligence, Routing
4. **Back-ref manager**: Toolkit (needs Tract back-reference — may need protocol extraction later)
5. **Complex manager**: Persistence (many callbacks, but straightforward delegation)

Each step: extract methods, wire callbacks, run full test suite (2717 tests).
