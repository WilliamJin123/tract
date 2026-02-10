# Project Research Summary

**Project:** Trace — Git for Context
**Domain:** Python library for git-like version control of LLM context windows
**Researched:** 2026-02-10
**Confidence:** MEDIUM-HIGH

## Executive Summary

Trace enters a validated but nascent space. Multiple research systems (Git Context Controller, ContextBranch, Context Folding) demonstrate that git-inspired version control improves agent performance dramatically: GCC achieved 48% on SWE-Bench-Lite (outperforming 26 systems), ContextBranch reduced context size by 58%, and agents with version-controlled context spontaneously adopted disciplined behaviors like modularizing work and testing before committing. The technical approach is sound: content-addressable storage via SQLite, commit DAG with parent pointers, and pluggable LLM operations for semantic merge/compression.

The critical insight from research is where the git metaphor breaks: context is order-sensitive natural language, not order-independent files. This creates three high-risk areas: (1) semantic merge/rebase requires LLM mediation because textual merge fails for prose, (2) compression is inherently lossy and research shows 3-55% performance degradation when done naively, and (3) multi-agent coordination introduces a new surface for state corruption. The recommended mitigation is architectural discipline: isolate LLM operations behind protocols, make compression explicit rather than automatic, and design for concurrent access from Phase 0.

The recommended stack is modern Python with minimal dependencies: SQLAlchemy 2.0 for storage, tiktoken for token counting (pluggable), Pydantic for API models, Click+Rich for CLI, and httpx for LLM calls. The architecture follows repository pattern for storage abstraction, strategy pattern for pluggable operations (materializers, merge strategies), and event sourcing mindset (commits are the source of truth, materialized context is a projection). The path to v1 is clear: Phase 0 builds foundations (storage, commits, materialization), Phase 1 adds linear history operations (log, diff, reset), Phase 2 implements branching and compression (the high-value features), and Phase 3 tackles multi-agent coordination.

## Key Findings

### Recommended Stack

The stack centers on modern Python tooling with a focus on minimal dependencies and dual sync/async support. Python 3.10 is the floor (required by SQLAlchemy 2.1 and modern typing), with 3.11+ recommended for performance. uv replaces pip/virtualenv/build as the single package manager (10-100x faster). The src/ layout ensures proper isolation.

**Core technologies:**
- **SQLAlchemy 2.0.46+** (ORM + database abstraction) — Modern 2.0-style with Mapped[] types, full async support via aiosqlite. Proven for DAG storage via adjacency list pattern.
- **tiktoken 0.12.0+** (token counting) — OpenAI's official tokenizer, Rust-backed, covers primary use case. Pluggable protocol for Anthropic/Google users.
- **Pydantic 2.10+** (data validation) — API surface types (CommitInfo, BranchInfo). 5-50x faster than v1. Validation at SDK boundary, dataclasses for internals.
- **Click 8.1+ + Rich 14.0+** (CLI framework) — Battle-tested, rich terminal output. Rich-click for beautiful help. CLI is thin wrapper over SDK.
- **httpx 0.28.0+** (HTTP client) — Unified sync/async interface for LLM API calls. Better than requests (no async) or aiohttp (async-only).
- **pytest 8.0+ + hypothesis 6.150+** (testing) — Standard test runner. hypothesis critical for property-based testing of DAG invariants (merge correctness, compression preservation, token accounting).

**Architecture decision encoded in stack:** SQLite is the only storage backend for v1 (no Alembic, no migrations). Libraries manage their own schema via create_all(). SQLite with WAL mode handles multi-agent concurrency. No NetworkX dependency — Trace's DAG is simple enough to implement on SQLAlchemy relationships.

### Expected Features

Research reveals clear feature tiers. Table stakes are validated by every comparable system (GCC, ContextBranch, LangGraph). Differentiators are where Trace innovates beyond existing tools. Anti-features clarify what Trace explicitly avoids to remain focused.

**Must have (table stakes):**
- **Commit with message and metadata** — Core promise. Without commits, there is no version control. Three types (append/edit/pin) are Trace innovation.
- **Linear history (log/status)** — Developers expect to see what happened. Include token counts per commit.
- **Branch and switch** — Validated by GCC (+348% task resolution) and ContextBranch (58% context reduction). Copy-on-write semantics, cheap branching.
- **Reset (soft/hard)** — Undo is fundamental. Soft = move HEAD, keep content. Hard = discard forward commits.
- **Token counting** — Context management without token awareness is useless. On every commit and operation. Pluggable tokenizer protocol.
- **Materialize/render** — Producing the context window for LLM is the read operation. Simple concatenation default, pluggable for structured prompts.
- **Persistence (SQLite)** — In-memory-only is a toy. Sessions must survive crashes.
- **Merge (basic + semantic)** — Complement to branching. LLM-mediated merge for semantic conflicts. Simpler "inject" (cherry-pick) for cases where full merge is overkill.

**Should have (competitive differentiators):**
- **Typed commits (Append/Edit/Pin)** — Novel. Enables smarter compression (never summarize pins), smarter diffs, smarter materialization.
- **Token-budget-aware compression** — "Compress to fit N tokens while preserving pins." ACON shows naive summarization loses critical details (26-54% reduction but performance loss). Hierarchical compression (summarize summaries).
- **Pluggable materializers** — Decouple storage from rendering. Claude wants XML, ChatGPT wants message arrays. Framework-agnostic enabler.
- **Semantic merge with LLM mediation** — Merging prose is fundamentally different from merging code. Use LLM to resolve "contradictory information" conflicts.
- **Spawn/collapse for multi-agent** — Give subagent a context subset (spawn), get results as summary (collapse). Maps to git clone + squash merge. Enables coordination without shared mutable state.
- **Compression-resistant pinning** — Mark "never compress this." System prompts, critical facts must survive. High value, simple to implement.

**Defer (v2+):**
- **Semantic diff** — LLM-powered "what meaning changed." High complexity, uncertain v1 value.
- **Framework adapters** — LangChain/CrewAI integrations. Need stable SDK first.
- **GUI/visualization** — SDK must be proven before GUI. Twigg is GUI-first; Trace is SDK-first.
- **Rebase with semantic safety** — Reorder context history while verifying meaning preserved. High complexity, novel concept.

### Architecture Approach

The architecture follows clean layering with explicit boundaries. Public API never touches storage directly; Core Engine never imports LLM operations. This isolation enables testing, extensibility, and clean error boundaries.

**Major components:**
1. **Storage Layer** — SQLAlchemy models + repository pattern. Content-addressable blob table separate from commit metadata (enables fast queries without loading content). Commit DAG via association table (CommitParent) supporting multiple parents for merge commits. Refs table for branches/HEAD as mutable pointers. All in one SQLite file with repo_id scoping for multi-agent.

2. **Core Engine** — DAG operations (commit, branch, merge, reset, checkout, log, diff). Pure logic, no LLM calls, no storage details. Operates on repository interfaces. Event sourcing mindset: commits are source of truth, materialized context is projection.

3. **Materialization** — Converts commit DAG into context window. Pluggable strategies (simple concat default, template-based, custom). Handles edit commit resolution (override semantics: edit commits reference a target commit, materializer replaces target's content with edit's).

4. **Token Accounting** — tiktoken default, pluggable protocol. Prefer API-reported counts when available (post-compression, post-merge). Report counts as estimates with confidence indicator.

5. **LLM Operations** — Isolated layer. Compression, semantic merge, reorder safety checks. Accepts user-provided callables (LLMCallable protocol). Ships convenience client (httpx-based, OpenAI-compatible) but core never depends on it.

6. **Multi-Agent (Phase 3)** — OpenTelemetry-inspired trace hierarchy. session_id (shared across all agent repos), repo_id (per agent), SpawnPointer links parent commit to child repo. Collapse generates summary + provenance. All repos in one SQLite file for atomic operations and cross-repo queries.

**Patterns to follow:**
- **Repository pattern** for storage abstraction (CommitRepository, RefRepository, ContentRepository)
- **Strategy pattern** for pluggable operations (materializers, merge strategies, compression strategies)
- **Event sourcing** mindset (commit chain is event log, derived state is always re-computable)
- **Protocol types** for extensibility (TokenCounter, LLMCallable, Materializer)

### Critical Pitfalls

Research identified 14 pitfalls across critical/moderate/minor severity. Top 5 by impact:

1. **Over-extending git metaphor where it breaks** — Git is order-agnostic; context is order-sensitive. Moving a commit changes its effective weight in LLM reasoning (Liu et al., "Lost in the Middle"). Merge/rebase/reorder are not neutral operations. **Avoidance:** LLM-mediated merge from day one (no textual merge fallback), semantic safety checks for reorder, explicit documentation of where git analogy fails.

2. **Lossy compression that silently drops critical information** — LLM summarization inherently loses specifics (research shows 3-55% performance degradation). Most vulnerable: exact values, conditional logic, negations, cross-references. **Avoidance:** Pin commits survive compression verbatim with hash verification, compression validation step checks information retention, hierarchical compression (2-3 detail layers), expose compression metrics to user, never auto-compress.

3. **Token counting that disagrees with LLM provider** — tiktoken works for OpenAI but wrong for Anthropic/Google. Even within OpenAI, community reports discrepancies on o4-mini/gpt-4o-mini. System prompts add hidden tokens. **Avoidance:** Pluggable TokenCounter protocol from Phase 0, prefer API-reported counts when available, report estimates with confidence indicator, test against real APIs in CI.

4. **Multi-agent state corruption and orphaned traces** — Multi-agent systems have 41-86.7% failure rate in production (Cemri et al.). Coordination failures are 36.94% of all failures. SQLite single-writer + concurrent agents = "database is locked" errors or orphaned traces. **Avoidance:** WAL mode mandatory, NullPool for SQLite connections, atomic spawn/collapse operations, application-level write serialization, `trace repair` command for cleanup.

5. **Testing LLM-dependent operations is non-deterministic** — LLM outputs vary across runs. Traditional assert-exact-output fails. **Avoidance:** Layered testing (unit tests with mocks for deterministic logic, contract tests with fuzzy assertions for LLM ops, snapshot tests with human review), mock at LLM boundary (LLMClient protocol), use VCR.py-style cassettes for integration tests.

## Implications for Roadmap

Based on research, the roadmap should follow strict dependency ordering with risk-front-loading for LLM operations.

### Phase 0: Foundations
**Rationale:** Everything depends on storage and commit/read cycle. Get the data model right before adding complexity.
**Delivers:** Storage layer (SQLAlchemy models, content-addressable blobs, repository interfaces), Core Engine (commit, read commit chain), Token Accounting (tiktoken integration + pluggable protocol), Basic Materialization (simple concatenation), Public API (Repo.open(), commit(), materialize())
**Addresses:** Table stakes (commit with metadata, token counting, persistence, materialize)
**Avoids:** Pitfall 1 (design for order sensitivity from day one), Pitfall 3 (pluggable token counting), Pitfall 7 (separate metadata from content in schema)

### Phase 1: Linear History Operations
**Rationale:** Proves core model works for simplest case before branching complexity. Validates data model.
**Delivers:** log, status, diff (textual), reset (soft/hard), checkout (read-only), full Public API for linear history, CLI wrapper (thin, over SDK)
**Addresses:** Table stakes (log, status, reset, diff)
**Avoids:** Pitfall 11 (SDK-first development, CLI is thin wrapper), Pitfall 14 (resolve edit commit semantics: override commit, not in-place mutation)
**Uses:** Click + Rich for CLI, pytest for comprehensive unit tests (all operations are deterministic at this phase)

### Phase 2: Branching, Merging, Compression
**Rationale:** Branching is the high-value feature validated by GCC/ContextBranch research. LLM operations (merge, compress) are the highest-risk features and require careful testing.
**Delivers:** branch, switch, merge (fast-forward + LLM-mediated), LLM Operations layer (callable interface + built-in httpx client), compression (budget-aware, respects pins, hierarchical), commit reordering, garbage collection
**Addresses:** Differentiators (semantic merge, token-budget-aware compression, pinning)
**Avoids:** Pitfall 1 (LLM-mediated merge, semantic safety for reorder), Pitfall 2 (compression validation, hierarchical layers, expose metrics), Pitfall 9 (layered testing with mocks, contract tests, VCR cassettes)
**Research flag:** This phase needs `/gsd:research-phase` for semantic merge strategies (CHATMERGE, ConGra benchmarks) and compression validation approaches.

### Phase 3: Multi-Agent
**Rationale:** Multi-agent is repo-of-repos, requires stable single-repo operations first. Highest coordination complexity.
**Delivers:** SpawnPointer model, Session management (session_id, trace hierarchy), spawn/collapse/expand operations, cross-repo queries, garbage collection with reachability analysis
**Addresses:** Differentiators (spawn/collapse for multi-agent)
**Avoids:** Pitfall 4 (WAL mode, atomic spawn/collapse, write serialization, repair command), Pitfall 12 (GC reachability analysis, dry-run, quarantine)
**Uses:** OpenTelemetry trace/span model for session hierarchy
**Research flag:** This phase needs `/gsd:research-phase` for multi-agent coordination patterns and testing approaches.

### Phase Ordering Rationale

- **Phase 0 before all else:** Storage and commit model are load-bearing. Mistakes here require rewrites. The edit commit semantics decision (override vs in-place) must be resolved in Phase 0.
- **Phase 1 validates data model:** Linear history is the simplest case. If this doesn't work cleanly, branching will be a nightmare. CLI development in Phase 1 ensures SDK is the tested path.
- **Phase 2 front-loads risk:** Merge and compression are the highest-complexity features and the most likely to fail. Better to discover fundamental issues in Phase 2 than after multi-agent is built on top.
- **Phase 3 builds on proven primitives:** Multi-agent is "just" spawn/collapse operations that use the Phase 2 compression and merge capabilities. If Phase 2 works, Phase 3 is mostly plumbing.
- **Defer semantic diff and GUI to post-v1:** These are polish features that don't affect core functionality. Semantic diff has uncertain value; GUI requires stable SDK first.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Merge/Compression):** Semantic merge strategies (CHATMERGE approach, ConGra benchmarks, LLM-as-judge for conflict resolution). Compression validation techniques (information retention scoring, hierarchical summarization prompt engineering).
- **Phase 3 (Multi-Agent):** Concurrent access patterns for SQLite (write serialization approaches, connection pooling vs NullPool tradeoffs). Spawn pointer lifecycle management (when to GC, how to handle crashed agents).

Phases with standard patterns (skip research-phase):
- **Phase 0 (Foundations):** SQLAlchemy adjacency list for DAG is well-documented. Content-addressable storage is standard CS pattern. No novel research needed.
- **Phase 1 (Linear History):** Git-style log/status/reset are well-understood. Textual diff is standard. These operations are deterministic and testable without LLM complexity.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies verified on PyPI with version numbers. SQLAlchemy 2.0 patterns confirmed in official docs. uv as package manager is well-documented 2026 standard. Only MEDIUM on hatchling vs uv_build (uv_build is very new). |
| Features | MEDIUM-HIGH | Table stakes validated by multiple systems (GCC, ContextBranch, LangGraph). Differentiators (typed commits, semantic merge) are novel so confidence is lower — validated by project design but not by existing production systems. Anti-features list is well-reasoned but untested in practice. |
| Architecture | HIGH | Storage layer design follows SQLAlchemy official directed graph example. Repository pattern is standard. LLM operations isolation via protocol is proven Python pattern. Edit commit semantics (override) mirrors git's immutable objects. Only MEDIUM on multi-agent hierarchy (OpenTelemetry analogy is solid but untested for this use case). |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls (position sensitivity, compression loss, token counting mismatch, multi-agent coordination) are well-documented in research papers and production failure reports. Phase-specific warnings are inferred from research but not tested in Trace's specific context. |

**Overall confidence:** MEDIUM-HIGH

Research is strong on proven patterns (git object model, SQLAlchemy DAG, token counting challenges) and validated use cases (branching improves agent performance). Confidence is lower on novel features (semantic merge quality, compression validation effectiveness) and multi-agent coordination (known to be hard but specific solutions are untested).

### Gaps to Address

- **Semantic merge quality:** No existing system does LLM-mediated semantic merge at the level Trace proposes. CHATMERGE and ConGra research show it's hard even for code. For natural language context, quality is unknown. **Mitigation:** Phase 2 should include extensive A/B testing of merge strategies with real agent tasks. Expose merge confidence scores to users.

- **Compression information retention:** Research shows all compression is lossy (3-55% performance drop), but optimal compression strategies for LLM context are not established. Hierarchical compression and pinning are theoretical improvements. **Mitigation:** Phase 2 should benchmark multiple compression approaches (flat vs hierarchical, token-based vs semantic-based). Provide compression validation as first-class feature, not afterthought.

- **Multi-agent concurrency model:** SQLite with WAL mode theoretically handles concurrent writes, but actual performance and failure modes in multi-agent LLM scenarios are unknown. **Mitigation:** Phase 3 should include stress testing with 5-10 concurrent agents. Design application-level write serialization as fallback if SQLite locking becomes bottleneck.

- **Token counting accuracy for non-OpenAI models:** tiktoken only covers OpenAI. Anthropic and Google tokenizers are not publicly available as standalone libraries. **Mitigation:** Provide pluggable TokenCounter protocol, document limitations, prefer API-reported counts. Phase 1 testing should validate tiktoken accuracy against real OpenAI API calls.

- **Edit commit semantics edge cases:** Override commit approach handles basic case (edit replaces target during materialization), but edge cases are unresolved: editing a pinned commit, compressing a range with both original and edit, editing an edit (chained overrides). **Mitigation:** Phase 0 must define and document all edge cases with test cases for each.

## Sources

### Primary (HIGH confidence)
- Git Internals - Git Objects (official git documentation) — object model, content-addressable storage
- SQLAlchemy 2.0/2.1 Documentation (official) — Declarative mapping, async support, directed graph example
- SQLite Documentation (official) — WAL mode, blob performance, thread safety
- Liu et al., "Lost in the Middle" (2023, arXiv 2307.03172) — position bias in LLM context
- Understanding and Improving Information Preservation in Prompt Compression (2025, arXiv 2503.19114) — compression loss quantification
- Git Context Controller paper (2025, arXiv 2508.00031) — 48% SWE-Bench-Lite result, validates git metaphor
- ContextBranch paper (2025, arXiv 2512.13914) — 58% context reduction, validates branching
- Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (2025, arXiv 2503.13657) — MAST taxonomy, 41-86.7% failure rate
- OpenTelemetry Traces (official) — trace/span hierarchy for multi-agent architecture

### Secondary (MEDIUM confidence)
- uv Project Documentation (Astral) — package manager, build backend
- Pydantic Serialization Documentation (official) — model_dump patterns
- pytest-asyncio Documentation (official) — async test patterns
- Hypothesis Documentation (official) — property-based testing for DAG invariants
- HTTPX Documentation (official) — sync/async unified interface
- Rich Documentation (official) — terminal formatting
- MemGPT/Letta Documentation — validates structured memory management improves agents
- LangGraph Persistence Documentation — checkpointing validates state snapshots
- ACON paper (2025, arXiv 2510.00615) — 26-54% compression with guideline-based approach
- Context Folding paper (2025, arXiv 2510.11967) — 62% BrowseComp-Plus with fold/collapse

### Tertiary (LOW confidence, needs validation)
- OpenAI community forum: tiktoken discrepancy reports — anecdotal but multiple independent reports
- Twigg (Product Hunt) — visual tree for conversations, 30-60% token reduction claim (no peer review)
- Factory.ai blog: Compressing Context — industry perspective, not research
- Anthropic Context Engineering blog — referenced widely but could not verify directly

---
*Research completed: 2026-02-10*
*Ready for roadmap: yes*
