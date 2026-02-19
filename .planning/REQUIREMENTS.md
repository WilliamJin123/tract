# Requirements: Tract v3.0 — DX & API Overhaul

**Defined:** 2026-02-19
**Core Value:** Rich functionality through minimal interfaces. Easy for the common case, configurable for every edge case.

## v3 Requirements

### Core API Simplification (CORE)

- [ ] **CORE-01**: commit() accepts shorthand args (role + text string) without requiring content model imports
- [ ] **CORE-02**: compile() output provides clean path to LLM-ready dicts (no manual list comprehension)
- [ ] **CORE-03**: Commit messages auto-generated when not provided (reduce ceremony)

### Conversation Layer (CONV)

- [ ] **CONV-01**: t.chat(text) — one call does user commit → compile → LLM call → assistant commit → record usage
- [ ] **CONV-02**: t.generate() — compile → LLM call → assistant commit → record usage (user msg already committed)
- [ ] **CONV-03**: Response object from chat/generate with .text, .usage, .commit_info, .generation_config

### LLM Integration (LLM)

- [ ] **LLM-01**: LLM configurable on Tract.open() — api_key, model, base_url as params
- [ ] **LLM-02**: Auto generation_config capture from LLM request parameters
- [ ] **LLM-03**: Auto usage recording from LLM response
- [ ] **LLM-04**: Per-operation LLM configuration — each LLM-powered operation (chat, merge, compress, orchestrate, auto-commit-messages) independently configurable with model, params, and custom client kwargs

### Format & Output (FMT)

- [ ] **FMT-01**: CompiledContext.to_dicts() returns list[dict] with role/content keys
- [ ] **FMT-02**: CompiledContext.to_openai() / .to_anthropic() for provider-specific formats

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
| CORE-01 | TBD | Pending |
| CORE-02 | TBD | Pending |
| CORE-03 | TBD | Pending |
| CONV-01 | TBD | Pending |
| CONV-02 | TBD | Pending |
| CONV-03 | TBD | Pending |
| LLM-01 | TBD | Pending |
| LLM-02 | TBD | Pending |
| LLM-03 | TBD | Pending |
| LLM-04 | TBD | Pending |
| FMT-01 | TBD | Pending |
| FMT-02 | TBD | Pending |

**Coverage:**
- v3 requirements: 12 total
- Mapped to phases: 0
- Unmapped: 12 (pending roadmap)

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-19 after initial definition*
