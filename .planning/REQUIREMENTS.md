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

- [ ] **CONFIG-01**: LLMConfig frozen dataclass replaces LLMOperationConfig with typed fields for all standard LLM hyperparameters plus extra dict escape hatch
- [ ] **QUERY-01**: query_by_config supports multi-field AND queries, IN operator, and whole-config matching via LLMConfig object
- [ ] **COOK-01**: All 3 Tier 1 cookbook examples updated to use LLMConfig typed access (response.generation_config.model instead of response.generation_config.get("model"))

## Future Requirements

- Cookbook-driven: additional requirements will be added as cookbook examples are rewritten against the new API
- Framework-specific adapters (Claude Code, OpenAI SDK, LangGraph) — after v3 API stabilizes
- Public API surface audit — reduce/reorganize __init__.py exports

## Out of Scope

| Feature | Reason |
|---------|--------|
| Async API | IO-bound but sync is simpler; async can be added in v4 |
| Breaking storage schema | v3 is API-layer only; schema stays at v5 |
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

**Coverage:**
- v3 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-20 after Phase 10 completion*
