# Tract Reconceptualization: Substrate + Rules

> **Status: SUPERSEDED by Phase 14 (Config + Directives + Middleware).** Rule engine was implemented (R0-R4, commit 7a86b94) then replaced (commit 23a89eb). Kept as historical reference.

Date: 2026-03-04 (updated 2026-03-05)
Status: Design complete — ready for implementation scoping

## Build Order

1. **Content types + compile strategy** — RuleContent, MetadataContent,
   adaptive K-window compile. Fully unit-testable, no LLM needed.
2. **Rule engine core** — index, event processing, config resolution,
   built-in deterministic conditions (tag, pattern, threshold).
3. **Default loop** — dumb loop replacing orchestrator. compile → LLM →
   tools → repeat. Clean exit on block.
4. **Action handlers + transitions** — operation, block, require,
   compile_filter, set_config. Transition mechanics across branches.
   LLM conditions/actions can be mocked in tests.
5. **Registries + extensibility** — custom conditions, actions, metrics,
   triggers. Protocol-based registration.

Validation: POC cookbooks on cheap models (Cerebras/Groq free tiers).
Benchmarks (SWE-Bench) deferred until frontier model access available.

## Motivation

While building cookbooks and the hook system, we realized tract had grown into
two distinct things without a unified design for the second:

1. **Context as commits** — the fundamental substrate. Git-like DAG for LLM
   context. Well-designed, well-tested (phases 1-13 complete, 1087+ tests).

2. **A workflow/behavior layer** — hooks, triggers, policies, orchestrator.
   Built incrementally, scattered across four subsystems that overlap and
   don't compose cleanly.

The hook system works but conflates two concerns: rules (quality control within
an operation) and protocols (workflow progression across operations). A hook
handler might validate a compression summary AND decide to trigger a branch
operation. One mechanism doing two conceptually different jobs.

The reconceptualization separates tract into **substrate** (graph primitives)
and **applications** (common patterns built from substrate via rules).

## The Two Zoom Levels

Tract is one graph at two zoom levels:

- **Macro graph**: workflow stages as nodes, transitions as edges. Ecommerce
  example: product_research → lander_pages → ads → metric_analysis → loop.
- **Micro graph**: the commit DAG within each stage. Messages, edits, branches,
  compressions — all graph transformations on content nodes.

The same data structure at different resolutions. A stage transition is a
"merge + compress + branch" at the macro level. Operations are graph
transformations at the micro level.

## See Also

- [SUBSTRATE.md](SUBSTRATE.md) — fundamental graph primitives
- [RULES.md](RULES.md) — the unified rule system design
- [APPLICATIONS.md](APPLICATIONS.md) — common patterns built from substrate
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) — unresolved design decisions
- [PAPER_ANALYSIS_GCC.md](PAPER_ANALYSIS_GCC.md) — comparison with Git Context Controller paper
