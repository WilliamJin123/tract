# Trace — Git for Context

## What This Is

Trace is a Python library that brings git-like version control to LLM context windows. It lets agent framework developers and AI power users commit, branch, merge, compress, and roll back context — giving agents clean, coherent, relevant context instead of ever-growing conversation logs that degrade performance. The library includes a policy engine for automatic context operations and a built-in orchestrator that completes the autonomy spectrum from manual through collaborative to fully autonomous context management. The primary interface is a Python SDK (`tract-ai` on PyPI); a CLI is provided for inspection and debugging.

## Core Value

1. Agents produce better outputs when their context is clean, coherent, and relevant. Trace makes context a managed, version-controlled resource instead of an append-only log.

2. Context operations exist on an autonomy spectrum. Every operation that transforms context (merge, rebase, cherry-pick, compression) supports three modes: **manual** (human only), **collaborative** (LLM resolves, human reviews and approves), and **autonomous** (LLM handles end-to-end). The default is collaborative — the LLM is a tool in the pipeline, not a gatekeeper. Humans can intercept at any point: before (guide via prompt), during (provide a custom resolver), or after (review and edit results before commit). This same pattern applies across all interfaces: the programmatic SDK, the human-facing CLI, and the agent toolkit.

## Requirements

### Validated

- Core data model: commits with content, message, token count, timestamp, parent pointers, and type (append/edit) — v1.0
- DAG-based commit history with branch and HEAD references — v1.0
- SQLite storage via SQLAlchemy — v1.0
- Python SDK as primary interface (framework-agnostic) — v1.0
- Token-aware operations: tiktoken default + API response extraction — v1.0
- Single-agent linear history: init, commit, log, status, diff, reset, checkout — v1.0
- Context materialization with pluggable materializers — v1.0
- CLI wrapper for inspection and debugging — v1.0
- Branching and merging: branch, switch, merge, rebase — v1.0
- LLM-mediated semantic merge strategy (built-in client + user-provided callable) — v1.0
- Compression: summarize commit ranges, respect pinned commits — v1.0
- Commit reordering with semantic safety checks — v1.0
- Multi-agent: spawn pointers, subagent traces, context collapse, expand for debugging — v1.0
- Session persistence and crash recovery — v1.0
- Session continuity and cross-session handoff — v1.0
- Garbage collection with retention policies — v1.0
- Published pip-installable package with docs and examples — v1.0
- Policy engine: declarative rules for auto-compress, auto-pin, auto-branch, auto-rebase — v2.0
- Context management agent toolkit with 15 tools, 3 profiles, and built-in orchestrator — v2.0
- Human override: every automatic operation can be intercepted, reviewed, or overridden — v2.0

### Active

**v3.0 — DX & API Overhaul**

- Convenience commit methods (t.system/user/assistant) to eliminate content type ceremony
- Integrated chat loop (t.chat/t.generate) that connects compile→LLM→commit→record_usage
- CompiledContext.to_dicts() and format-specific output methods
- Auto generation_config capture and usage recording when using built-in client
- LLM configuration on Tract.open() to eliminate separate configure_llm() step
- Ongoing: cookbook-driven discovery of additional API issues and design fixes

### Out of Scope

- Framework-specific adapters (Claude Code, OpenAI SDK, LangGraph) — build after autonomous features are proven in real use
- Web/Desktop GUI viewer ("GitHub of Trace") — deferred
- RAG integration / external file tracking — defer until core context model is proven
- Mobile or non-Python clients — Python-first
- Benchmarking framework (A/B managed vs unmanaged context) — defer until real-world usage data available
- Virtual context with tiered storage — future (VCTX-01, VCTX-02)
- Semantic diff (LLM-powered "what meaning changed") — future (SMDIFF-01)
- Trace blame and cross-agent audit queries — future (AUDIT-01, AUDIT-02)

## Context

Shipped v2.0 with 15,545 LOC Python (source) + 16,169 LOC Python (tests).
Tech stack: Python, SQLAlchemy, SQLite, tiktoken, httpx, tenacity, Pydantic, Click, Rich.
895 tests passing across 22 test files.
Published as `tract-ai` on PyPI (import as `tract`).

Architecture:
- Core: models, storage (SQLAlchemy/SQLite), engine (commit/compile), operations (navigation, branch, merge, rebase, compression)
- LLM: httpx-based client with OpenAI-compatible API, tenacity retry, pluggable resolver protocol
- CLI: Click + Rich for terminal inspection
- Policy: ABC-based policies with PolicyEvaluator sidecar, 4 built-in policies
- Toolkit: 15 tool definitions, 3 profiles, ToolExecutor
- Orchestrator: assess → LLM → tools → repeat loop with stop/pause lifecycle

The project addresses a real pain point in agentic engineering: longer context windows and low-signal tokens degrade agent performance. No existing tooling provides structured, version-controlled context management for agents.

## Constraints

- **Language**: Python — context management is IO-bound (LLM calls, storage), not compute-bound
- **Storage**: SQLite via SQLAlchemy — simple, sufficient through v2; schema version 5 with auto-migration chain
- **Token counting**: tiktoken as default tokenizer, with support for extracting token counts from API response payloads
- **LLM operations**: Built-in convenience client + accept user-provided callables for flexibility
- **API-first**: Primary consumers are agents/frameworks, not humans. SDK is the product.
- **Scope**: v1 complete (Phases 1-5), v2 complete (Phases 6-7). v3.0 focuses on DX/API overhaul.
- **Backward compat**: v3 may introduce breaking changes where the old API was genuinely bad. Convenience methods are always additive.
- **Cookbook-driven**: Every API change must make a cookbook example simpler. No changes for their own sake.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python-first, no premature Rust/C optimization | IO-bound workload, developer velocity matters more | Good — 895 tests run in 32s |
| Framework-agnostic core, no framework adapters | Avoid coupling, keep library clean, add adapters later | Good — clean SDK boundary |
| SQLite via SQLAlchemy for storage | Simple and sufficient; can swap backend later via SQLAlchemy | Good — 5 schema versions, auto-migration |
| tiktoken + API response extraction for token counting | Covers 90% of use cases without extra dependencies | Good — two-tier tracking works well |
| Pluggable materializers with simple concat default | Power users can customize; sane default for everyone else | Good — ContextCompiler protocol |
| Built-in LLM client + user-provided callables | Convenience for common case, flexibility for advanced use | Good — ResolverCallable protocol |
| Two operations (Append, Edit) + priority annotations | Operations are commit-level; annotations are lightweight metadata | Good — clean separation |
| Content-addressable hashing (SHA-256) | Dedup, integrity verification, immutable history | Good — compression provenance |
| Per-tract custom type registry | Extensible content types without global state | Good — SessionContent etc. |
| PolicyEvaluator as sidecar (not embedded in Tract) | Separation of concerns, optional feature | Good — clean integration |
| Orchestrator with autonomy ceiling | Global constraint on max automation level | Good — `trigger_autonomy` override |
| `tract-ai` distribution name, `tract` import name | Avoid PyPI name conflicts, avoid stdlib `trace` shadow | Good — no conflicts |

---
*Last updated: 2026-02-19 after v3.0 milestone start*
