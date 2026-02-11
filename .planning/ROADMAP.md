# Roadmap: Trace

## Overview

Trace delivers git-like version control for LLM context windows across five phases. Phase 1 establishes the data model, storage layer, and commit/compile cycle -- the load-bearing foundation everything else depends on. Phase 2 proves the model works by building linear history operations and a CLI for debugging. Phase 3 front-loads the highest-risk work: branching, merging (including LLM-mediated semantic merge), and the LLM client infrastructure. Phase 4 delivers compression with token budget awareness, pinning, and commit reordering. Phase 5 adds multi-agent coordination (spawn, collapse, cross-repo queries) and packages the library for release.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundations** - Data model, storage, commit/compile cycle, token accounting, and SDK entry point
- [x] **Phase 1.1: Incremental Compile Cache & Token Tracking** - O(1) append-path compilation, API-reported token usage as source of truth (INSERTED)
- [ ] **Phase 2: Linear History & CLI** - Log, status, diff, reset, checkout, and CLI wrapper for inspection
- [ ] **Phase 3: Branching & Merging** - Branch, switch, merge (fast-forward + semantic), rebase, cherry-pick, and LLM client
- [ ] **Phase 4: Compression** - Token-budget-aware compression, pinned commit preservation, commit reordering, garbage collection
- [ ] **Phase 5: Multi-Agent & Release** - Spawn/collapse for subagents, session persistence, crash recovery, cross-repo queries, packaging

## Phase Details

### Phase 1: Foundations
**Goal**: Users can create a trace, commit structured context snapshots, and compile context for LLM consumption with accurate token counts
**Depends on**: Nothing (first phase)
**Requirements**: INFR-01, INFR-02, INFR-03, INFR-04, INFR-05, INFR-06, CORE-01, CORE-02, CORE-08, CORE-09, INTF-01
**Success Criteria** (what must be TRUE):
  1. User can initialize a new trace via `Repo.open()` and it persists to SQLite storage
  2. User can commit context with message, timestamp, and operation (append/edit) and retrieve it by hash; priority annotations (pin/skip/normal) control compilation inclusion
  3. User can commit structured content (plain text, conversation messages with roles, tool call results) and the structure is preserved through materialization
  4. User can compile the current context and get a coherent output suitable for LLM consumption, using either the default context compiler or a custom one
  5. Every commit and compile operation reports token counts, and users can swap in a custom tokenizer or have API-reported counts used when available
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md -- Project scaffolding, domain models (7 content types), SQLAlchemy schema, repository pattern (ABCs + SQLite)
- [x] 01-02-PLAN.md -- Deterministic hashing, token counting, commit engine, and default context compiler
- [x] 01-03-PLAN.md -- Repo class (public SDK entry point) and end-to-end integration tests

### Phase 1.1: Incremental Compile Cache & Token Tracking (INSERTED)
**Goal**: Reduce compile latency via incremental caching for append-only operations, and establish API-reported token usage as the primary source of truth over tiktoken estimates
**Depends on**: Phase 1
**Requirements**: CORE-09 (token tracking refinement), INFR-06 (performance)
**Success Criteria** (what must be TRUE):
  1. Compiling after an APPEND commit reuses cached intermediate state (O(1) incremental extend, not O(n) full chain walk)
  2. EDIT and annotate operations trigger full cache invalidation (correctness over speed)
  3. User can feed API-reported token usage back into Trace via `repo.record_usage()`, and this is preferred over tiktoken counts when available
  4. tiktoken remains the pre-call estimator for budget enforcement; API actuals are the post-call source of truth
  5. `CompiledContext.token_source` accurately reflects whether counts came from tiktoken estimate or API response
**Plans**: 2 plans

Plans:
- [x] 01.1-01-PLAN.md -- CompileSnapshot dataclass, build_message_for_commit() extraction, incremental APPEND fast path, EDIT/annotate/batch invalidation
- [x] 01.1-02-PLAN.md -- record_usage() API, OpenAI/Anthropic dict normalization, two-tier token tracking integration tests

### Phase 2: Linear History & CLI
**Goal**: Users can inspect, navigate, and manipulate linear commit history through both the SDK and a CLI
**Depends on**: Phase 1
**Requirements**: CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, INTF-02
**Success Criteria** (what must be TRUE):
  1. User can view commit history (log) with per-commit and cumulative token counts
  2. User can check current state (status) showing HEAD position, branch name, and token budget usage
  3. User can compare any two commits (diff) and see textual differences in content
  4. User can reset HEAD to a previous commit (soft keeps content accessible, hard discards forward history) and checkout a specific commit for read-only inspection
  5. User can perform all of the above via a CLI (`trace log`, `trace status`, `trace diff`, `trace reset`, `trace checkout`) with readable terminal output
**Plans**: TBD

Plans:
- [ ] 02-01: Linear history operations (log, status, diff, reset, checkout)
- [ ] 02-02: CLI wrapper with Click + Rich

### Phase 3: Branching & Merging
**Goal**: Users can create divergent context branches and merge them back together, including LLM-mediated semantic merge for conflicting content
**Depends on**: Phase 2
**Requirements**: BRNC-01, BRNC-02, BRNC-03, BRNC-04, BRNC-05, BRNC-06, INTF-03, INTF-04
**Success Criteria** (what must be TRUE):
  1. User can create a named branch from HEAD, switch between branches, and each branch maintains independent history
  2. User can merge a branch into the current branch with automatic fast-forward when possible and a merge commit when histories diverge
  3. User can trigger LLM-mediated semantic merge for conflicting or overlapping context, using either the built-in LLM client or a user-provided callable
  4. User can rebase the current branch onto a target with semantic safety checks that warn when reordering affects meaning
  5. User can cherry-pick specific commits from one branch into another
**Plans**: TBD

Plans:
- [ ] 03-01: Branch and switch operations (pointer-based branching, ref management)
- [ ] 03-02: LLM client infrastructure (built-in httpx client + user-provided callable protocol)
- [ ] 03-03: Merge strategies (fast-forward, merge commit, LLM-mediated semantic merge)
- [ ] 03-04: Rebase, cherry-pick, and semantic safety checks

### Phase 4: Compression
**Goal**: Users can compress context history to fit token budgets while preserving critical information and maintaining history integrity
**Depends on**: Phase 3 (uses LLM client from Phase 3)
**Requirements**: COMP-01, COMP-02, COMP-03, COMP-04
**Success Criteria** (what must be TRUE):
  1. User can compress a range of commits into a summary commit targeting a specific token budget, and the summary is coherent
  2. Pinned commits survive compression verbatim -- their content is unchanged and verifiable by hash
  3. User can reorder commits with semantic safety checks that warn when the reordering changes meaning
  4. User can run garbage collection to remove unreachable commits with configurable retention policies (e.g., keep last N days, keep all pinned)
**Plans**: TBD

Plans:
- [ ] 04-01: Compression engine (LLM-powered summarization, token budget targeting, hierarchical compression)
- [ ] 04-02: Pin preservation, commit reordering, and garbage collection

### Phase 5: Multi-Agent & Release
**Goal**: Users can coordinate multiple agent traces with spawn/collapse semantics, recover from crashes, and install Trace as a pip package
**Depends on**: Phase 4 (uses compression for collapse operations)
**Requirements**: MAGT-01, MAGT-02, MAGT-03, MAGT-04, MAGT-05, MAGT-06, MAGT-07, INTF-05
**Success Criteria** (what must be TRUE):
  1. User can spawn a subagent trace linked to the current commit, and each subagent gets its own full trace repository with independent history
  2. User can collapse a subagent trace back into the parent (producing a summary commit with provenance pointer) and expand it later for debugging
  3. All agent traces persist durably, and a user can resume from the last committed state after a process crash or restart
  4. User can query across repositories within a session (e.g., "what did agent-2 know at this point?")
  5. Trace is pip-installable with documentation and usage examples

**Plans**: TBD

Plans:
- [ ] 05-01: Spawn/collapse model and session management
- [ ] 05-02: Persistence, crash recovery, and cross-repo queries
- [ ] 05-03: Packaging, documentation, and release

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 1.1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundations | 3/3 | Complete | 2026-02-10 |
| 1.1 Compile Cache & Token Tracking | 2/2 | Complete | 2026-02-11 |
| 2. Linear History & CLI | 0/2 | Not started | - |
| 3. Branching & Merging | 0/4 | Not started | - |
| 4. Compression | 0/2 | Not started | - |
| 5. Multi-Agent & Release | 0/3 | Not started | - |
