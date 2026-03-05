# Tract Reconceptualization: Substrate + Rules

Date: 2026-03-04 (updated 2026-03-05)
Status: Design exploration — substrate, rules, compile strategy, and metadata settled

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
