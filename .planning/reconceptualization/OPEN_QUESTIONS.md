# Open Questions

## Resolved

- **Rule storage**: Rules are first-class commits (RuleContent type) on
  the branch they scope to. Gets versioning/editing/branching for free.
- **Rule schema**: 4 fields — name, trigger, condition, action. Scope via
  DAG placement. Provenance in commit metadata. Status via annotations.
- **Stage mechanics**: Stages are branches. No stack. Inheritance via DAG
  ancestry. The macro workflow graph emerges from branch/merge patterns.
- **Multi-agent**: Fan-out/fan-in on separate branches. Urgent broadcast
  via cherry-pick to sibling branches.
- **Human interaction**: Humans commit like any other actor (user-role
  commits with tags). Approval gates are rules checking for required tags.
- **Evaluation lifecycle**: Event-driven (commit-scoped on commit,
  transition-scoped on transition request). Rule cache at branch roots.
- **Long-horizon persistence**: Important outputs persist as external
  artifacts (files, git repos). Cross-iteration context flows through
  artifact references + selective transitions.
- **Conflict resolution**: DAG distance (closer wins). Same distance:
  later timestamp wins. One precedence dimension, no priority field.
- **Engine conventions**: Override vs accumulate per action type.
  Execution order: gates → work → handoff → post.
- **Transition design**: Decomposed into independent rules on source
  branch, not a monolithic object. Each concern independently authored.
- **Extensibility**: Conditions, actions, metrics, and triggers are all
  registries following the content type registry pattern.
- **Compile behavior**: RuleContent has compilable=False hint, never
  rendered as LLM messages.
- **Rule naming**: Not enforced unique. Same-name rules resolve by
  precedence (closer to HEAD wins). Name is stable identity for editing.
- **Compile filter modes**: selective, summarized, same_context, new_agent
  with mode-specific params.
- **Compile strategy**: full, minimal, adaptive(k). Agent controls its own
  base context via active rules. Non-destructive read-time lens, orthogonal
  to compression. compile() also exposed as read tool for cross-branch peeks.
- **Structured metadata**: MetadataContent type (compilable=False, EDIT to
  update, preserved across compression, optional path for file export).
  Bridges tract SQL storage and agents' file-based state.

## Promotion Loop

### 1. Pattern Detection
How does the system recognize "the LLM keeps making the same decision"?
- Exact match on action? (fragile — LLM wording varies)
- Semantic similarity? (needs embedding, more infrastructure)
- Explicit LLM reflection? ("I notice I keep doing X, should I make a rule?")

### 2. Draft → Active Promotion Criteria
When is a draft rule confident enough to skip the LLM?
- Fixed threshold (N successful applications)?
- Statistical (confidence interval on success rate)?
- Domain-dependent (safety rules need higher bar than formatting preferences)?

### 3. Rule Invalidation
How does a promoted rule get demoted or deleted if it starts making bad
decisions?
- The LLM overrides it and the system notices?
- Periodic re-evaluation?
- Human review?

## Architecture

### 4. Same Package or Separate?
Should the rule system live in `tract` or be a separate package?
- Same: tighter integration, single install, simpler DX
- Separate: cleaner boundaries, tract-core stays focused
- Decision deferred — design the interface first, packaging later

### 5. Migration from Current Hook System
The current system has hooks, triggers, policies, and orchestrator.
- Do these become thin wrappers over the rule system?
- Or clean break with migration path?
- How to maintain backward compatibility for existing cookbook examples?

### 6. Relationship to Existing Orchestrator
The orchestrator (Phase 7) has its own assessment → action loop. Likely
becomes an application built on top of rules (the assess → act loop is
a set of turn-scoped rules). Needs concrete mapping.

## Visualization

### 7. Graph Rendering
The macro/micro zoom model needs visualization:
- Zoom out: workflow stage graph (nodes = stages, edges = transitions)
- Zoom in: commit DAG within a stage
- Zoom into commit: content, edit history, annotations
- What visualization library/approach?
- Interactive or static?

## Querying

### 8. Query Primitives vs Compaction
Starting position: hash + tag + role queries as substrate, everything else
via search + compaction. What queries will we inevitably need?
- By content type?
- By token count range?
- By time range?
- Full-text search over content?
- Subgraph queries ("all commits between A and B")?

## Real-World Workflow Concerns

### 9. Multi-Agent Merge Strategies
Fan-out/fan-in is resolved (separate branches → merge). Remaining:
what merge strategies work best for consolidating parallel research?
LLM-synthesized merge vs structured merge vs human-curated?

### 10. Human Override Semantics
Humans interact via commits + tags (resolved). But can a human override
an active rule mid-stage? If so, is that a new rule commit with higher
precedence, or a one-time exception?

### 11. Cross-Workflow Rule Sharing
If "redact secrets" is useful in every workflow, how is it shared?
- Workflow-root rules inherited by all stages (current model)
- A "global rules" concept above workflows?
- Import/compose from a rule library?

### 12. Natural Language Rule Parsing
Concrete NL rules need to be parsed into structured RuleContent. Options:
- LLM parses at authoring time → stored as structured rule
- LLM evaluates at runtime → slower but handles ambiguity
- Hybrid: parse at authoring, flag low-confidence parses for runtime eval

## Inspiration / Prior Art

Research these further:
- **Temporal.io** — durable execution, event-history-as-log
- **LangGraph** — state machine + LLM hybrid, graph structure for stages
- **Drools/Rete** — production rule systems, efficient evaluation
- **WorkflowLLM** — fine-tuning LLMs for workflow orchestration
- Nobody currently combines: versioned state + rule engine + LLM fallback
