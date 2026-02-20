# Requirements: Tract v3.0 — DX & API Overhaul

**Defined:** 2026-02-19
**Core Value:** Rich functionality through minimal interfaces. Easy for the common case, configurable for every edge case.

## v3 Requirements

### Core API Simplification (CORE)

- [x] **CORE-01**: commit() accepts shorthand args (role + text string) without requiring content model imports
- [x] **CORE-02**: compile() output provides clean path to LLM-ready dicts (no manual list comprehension)
- [x] **CORE-03**: Commit messages auto-generated when not provided (reduce ceremony)

### Conversation Layer (CONV)

- [x] **CONV-01**: t.chat(text) — one call does user commit → compile → LLM call → assistant commit → record usage
- [x] **CONV-02**: t.generate() — compile → LLM call → assistant commit → record usage (user msg already committed)
- [x] **CONV-03**: Response object from chat/generate with .text, .usage, .commit_info, .generation_config

### LLM Integration (LLM)

- [x] **LLM-01**: LLM configurable on Tract.open() — api_key, model, base_url as params
- [x] **LLM-02**: Auto generation_config capture from LLM request parameters
- [x] **LLM-03**: Auto usage recording from LLM response
- [x] **LLM-04**: Per-operation LLM configuration — each LLM-powered operation (chat, merge, compress, orchestrate) independently configurable with model, params, and custom client kwargs

### Format & Output (FMT)

- [x] **FMT-01**: CompiledContext.to_dicts() returns list[dict] with role/content keys
- [x] **FMT-02**: CompiledContext.to_openai() / .to_anthropic() for provider-specific formats

### Unified Config & Cookbook (CONFIG/QUERY/COOK)

- [x] **CONFIG-01**: LLMConfig frozen dataclass replaces LLMOperationConfig with typed fields for all standard LLM hyperparameters plus extra dict escape hatch
- [x] **QUERY-01**: query_by_config supports multi-field AND queries, IN operator, and whole-config matching via LLMConfig object
- [x] **COOK-01**: All 3 Tier 1 cookbook examples updated to use LLMConfig typed access (response.generation_config.model instead of response.generation_config.get("model"))

### LLMConfig Cleanup (CLEAN)

- [x] **CLEAN-01**: OperationConfigs frozen dataclass with typed chat/merge/compress/orchestrate fields — typos caught at construction, IDE autocomplete
- [x] **CLEAN-02**: Consolidated _default_config: LLMConfig replaces _default_model; open(model=...) is sugar
- [x] **CLEAN-03**: chat()/generate()/merge()/compress() accept llm_config: LLMConfig for full call-level override
- [x] **CLEAN-04**: LLMConfig.from_dict() handles cross-framework aliases and ignores API plumbing keys; from_obj() extracts from arbitrary objects
- [x] **CLEAN-05**: _build_generation_config captures ALL resolved fields (top_p, seed, frequency_penalty, etc.)
- [x] **CLEAN-06**: Compression gen_config threading, orchestrator full config forwarding, compress error guard

### Provenance & Schema (PROV)

- [x] **PROV-01**: Unified OperationEvent+OperationCommit 2-table model records compress, reorganize, and import operations
- [x] **PROV-02**: CompileRecord+CompileEffective tables persist compiled context; chat()/generate() auto-create records
- [x] **PROV-03**: Rebase creates "reorganize" events with source/result commit mappings
- [x] **PROV-04**: Cherry-pick dissolved into import_commit with "import" event recording
- [x] **PROV-05**: GC respects OperationCommitRow FKs — source commits protected from garbage collection
- [x] **PROV-06**: Schema v5→v6 migration with compression data migration and old table removal

## Future Requirements

- Cookbook-driven: additional requirements will be added as cookbook examples are rewritten against the new API
- Framework-specific adapters (Claude Code, OpenAI SDK, LangGraph) — after v3 API stabilizes
- Public API surface audit — reduce/reorganize __init__.py exports

## Out of Scope

| Feature | Reason |
|---------|--------|
| Async API | IO-bound but sync is simpler; async can be added in v4 |
| Removing existing low-level methods | Backward compat — commit(), compile(), etc. stay; they get smarter |
| GUI/visualization | Deferred from v2, still deferred |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 8 | Complete |
| CORE-02 | Phase 8 | Complete |
| CORE-03 | Phase 8 | Complete |
| CONV-01 | Phase 9 | Complete |
| CONV-02 | Phase 9 | Complete |
| CONV-03 | Phase 9 | Complete |
| LLM-01 | Phase 9 | Complete |
| LLM-02 | Phase 9 | Complete |
| LLM-03 | Phase 9 | Complete |
| LLM-04 | Phase 10 | Complete |
| FMT-01 | Phase 8 | Complete |
| FMT-02 | Phase 8 | Complete |
| CONFIG-01 | Phase 11 | Complete |
| QUERY-01 | Phase 11 | Complete |
| COOK-01 | Phase 11 | Complete |

| CLEAN-01 | Phase 12 | Complete |
| CLEAN-02 | Phase 12 | Complete |
| CLEAN-03 | Phase 12 | Complete |
| CLEAN-04 | Phase 12 | Complete |
| CLEAN-05 | Phase 12 | Complete |
| CLEAN-06 | Phase 12 | Complete |

| PROV-01 | Phase 13 | Complete |
| PROV-02 | Phase 13 | Complete |
| PROV-03 | Phase 13 | Complete |
| PROV-04 | Phase 13 | Complete |
| PROV-05 | Phase 13 | Complete |
| PROV-06 | Phase 13 | Complete |

**Coverage:**
- v3 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-20 after Phase 13 completion*
