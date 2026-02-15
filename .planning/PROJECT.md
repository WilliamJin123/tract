# Trace — Git for Context

## What This Is

Trace is a Python library that brings git-like version control to LLM context windows. It lets agent framework developers and AI power users commit, branch, merge, compress, and roll back context — giving agents clean, coherent, relevant context instead of ever-growing conversation logs that degrade performance. The primary interface is a Python SDK; a CLI is provided for inspection and debugging.

## Core Value

1. Agents produce better outputs when their context is clean, coherent, and relevant. Trace makes context a managed, version-controlled resource instead of an append-only log.

2. Context operations exist on an autonomy spectrum. Every operation that transforms context (merge, rebase, cherry-pick, compression) supports three modes: **manual** (human only), **collaborative** (LLM resolves, human reviews and approves), and **autonomous** (LLM handles end-to-end). The default is collaborative — the LLM is a tool in the pipeline, not a gatekeeper. Humans can intercept at any point: before (guide via prompt), during (provide a custom resolver), or after (review and edit results before commit). This same pattern applies across all interfaces: the programmatic SDK, the human-facing CLI, and future agent toolkits.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Core data model: commits with content, message, token count, timestamp, parent pointers, and type (append/edit/pin)
- [ ] DAG-based commit history with branch and HEAD references
- [ ] SQLite storage via SQLAlchemy
- [ ] Python SDK as primary interface (framework-agnostic, no coupling to any specific agent framework)
- [ ] Token-aware operations: tiktoken default + API response extraction when available
- [ ] Single-agent linear history: init, commit, log, status, diff, reset (soft/hard), checkout
- [ ] Context materialization with pluggable materializers (simple concatenation default, extensible)
- [ ] CLI wrapper for inspection and debugging
- [ ] Branching and merging: branch, switch, merge, rebase
- [ ] LLM-mediated semantic merge strategy (built-in LLM client + user-provided callable)
- [ ] Compression: summarize commit ranges, respect pinned commits, hierarchical summarization
- [ ] Commit reordering with semantic safety checks
- [ ] Multi-agent: spawn pointers, subagent traces, context collapse, expand for debugging
- [ ] Session persistence and crash recovery
- [ ] Garbage collection with retention policies
- [ ] Published pip-installable package with docs and examples

### Out of Scope

- Framework-specific adapters (Claude Code, OpenAI SDK, LangGraph) — build after core is stable
- Web/Desktop GUI viewer ("GitHub of Trace") — Phase 4 in future milestone
- Autonomous context management / policy engine — Phase 5 in future milestone
- RAG integration / external file tracking — defer until core context model is proven
- Mobile or non-Python clients — Python-first

## Context

- The project addresses a real pain point in agentic engineering: longer context windows and low-signal tokens degrade agent performance
- No existing tooling provides structured, version-controlled context management for agents
- The git mental model is familiar to the target audience (developers) and maps well to the problem domain
- Key insight: context is order-sensitive (unlike git file content), which affects merge, rebase, and reorder operations
- LLM-powered operations (compression, semantic merge) mean the version control system itself requires inference calls — different cost model than git
- Personal knowledge gaps to address via research: detailed git internals, CLI UX design patterns (OpenTelemetry as inspiration), decoupled architecture for future GUI

## Constraints

- **Language**: Python — context management is IO-bound (LLM calls, storage), not compute-bound
- **Storage**: SQLite via SQLAlchemy — simple, sufficient for v1
- **Token counting**: tiktoken as default tokenizer, with support for extracting token counts from API response payloads
- **LLM operations**: Built-in convenience client + accept user-provided callables for flexibility
- **API-first**: Primary consumers are agents/frameworks, not humans. SDK is the product.
- **Scope**: v1 milestone covers Phase 0 through Phase 3 (multi-agent). Phases 4-5 are future milestones.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python-first, no premature Rust/C optimization | IO-bound workload, developer velocity matters more | — Pending |
| Framework-agnostic core, no framework adapters in v1 | Avoid coupling, keep library clean, add adapters later | — Pending |
| SQLite via SQLAlchemy for storage | Simple and sufficient; can swap backend later via SQLAlchemy | — Pending |
| tiktoken + API response extraction for token counting | Covers 90% of use cases without extra dependencies | — Pending |
| Pluggable materializers with simple concat default | Power users can customize; sane default for everyone else | — Pending |
| Built-in LLM client + user-provided callables | Convenience for common case, flexibility for advanced use | — Pending |
| Edit commit semantics | TBD — in-place replacement vs override commit. Resolve in Phase 0 | — Pending |
| Storage model (structured rows vs blobs) | TBD — resolve during Phase 0 research/design | — Pending |
| Two operations (Append, Edit) + priority annotations (Skip/Normal/Pinned) | Operations are commit-level; annotations are lightweight metadata with provenance history | Decided |

---
*Last updated: 2026-02-10 after initialization*
