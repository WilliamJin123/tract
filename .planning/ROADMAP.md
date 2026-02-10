# Roadmap: Trace

## Overview

Trace delivers git-like version control for LLM context windows across five phases. Phase 1 establishes the data model, storage layer, and commit/materialize cycle -- the load-bearing foundation everything else depends on. Phase 2 proves the model works by building linear history operations and a CLI for debugging. Phase 3 front-loads the highest-risk work: branching, merging (including LLM-mediated semantic merge), and the LLM client infrastructure. Phase 4 delivers compression with token budget awareness, pinning, and commit reordering. Phase 5 adds multi-agent coordination (spawn, collapse, cross-repo queries) and packages the library for release.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundations** - Data model, storage, commit/materialize cycle, token accounting, and SDK entry point
- [ ] **Phase 2: Linear History & CLI** - Log, status, diff, reset, checkout, and CLI wrapper for inspection
- [ ] **Phase 3: Branching & Merging** - Branch, switch, merge (fast-forward + semantic), rebase, cherry-pick, and LLM client
- [ ] **Phase 4: Compression** - Token-budget-aware compression, pinned commit preservation, commit reordering, garbage collection
- [ ] **Phase 5: Multi-Agent & Release** - Spawn/collapse for subagents, session persistence, crash recovery, cross-repo queries, packaging

## Phase Details

### Phase 1: Foundations
**Goal**: Users can create a trace, commit structured context snapshots, and materialize context for LLM consumption with accurate token counts
**Depends on**: Nothing (first phase)
**Requirements**: INFR-01, INFR-02, INFR-03, INFR-04, INFR-05, INFR-06, CORE-01, CORE-02, CORE-08, CORE-09, INTF-01
**Success Criteria** (what must be TRUE):
  1. User can initialize a new trace via `Repo.open()` and it persists to SQLite storage
  2. User can commit context with message, timestamp, and type (append/edit/pin) and retrieve it by hash
  3. User can commit structured content (plain text, conversation messages with roles, tool call results) and the structure is preserved through materialization
  4. User can materialize the current context and get a coherent output suitable for LLM consumption, using either the default concatenation materializer or a custom one
  5. Every commit and materialize operation reports token counts, and users can swap in a custom tokenizer or have API-reported counts used when available
**Plans**: TBD

Plans:
- [ ] 01-01: Storage layer and data model (SQLAlchemy models, content-addressable blobs, repository interfaces)
- [ ] 01-02: Commit engine, token accounting, and materialization
- [ ] 01-03: SDK public API surface (Repo class, commit types, pluggable protocols)

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
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundations | 0/3 | Not started | - |
| 2. Linear History & CLI | 0/2 | Not started | - |
| 3. Branching & Merging | 0/4 | Not started | - |
| 4. Compression | 0/2 | Not started | - |
| 5. Multi-Agent & Release | 0/3 | Not started | - |
