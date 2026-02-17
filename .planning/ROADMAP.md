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
- [x] **Phase 1.2: Rename Repo to Tract** - Rename entry-point class Repo→Tract, repo_id→tract_id across source, tests, and planning docs (INSERTED)
- [x] **Phase 1.3: Hyperparameter Config Storage** - Store LLM generation config (temperature, top_p, top_k, etc.) with commits for full call provenance (INSERTED)
- [x] **Phase 1.4: LRU Compile Cache & Snapshot Patching** - Replace single-snapshot cache with LRU keyed by head_hash, EDIT/annotate snapshot patching instead of invalidation (INSERTED)
- [x] **Phase 2: Linear History & CLI** - Log, status, diff, reset, checkout, and CLI wrapper for inspection
- [x] **Phase 3: Branching & Merging** - Branch, switch, merge (fast-forward + semantic), rebase, cherry-pick, and LLM client
- [x] **Phase 4: Compression** - Token-budget-aware compression, pinned commit preservation, commit reordering, garbage collection
- [ ] **Phase 5: Multi-Agent & Release** - Spawn/collapse for subagents, session persistence, crash recovery, cross-repo queries, packaging

## Phase Details

### Phase 1: Foundations
**Goal**: Users can create a trace, commit structured context snapshots, and compile context for LLM consumption with accurate token counts
**Depends on**: Nothing (first phase)
**Requirements**: INFR-01, INFR-02, INFR-03, INFR-04, INFR-05, INFR-06, CORE-01, CORE-02, CORE-08, CORE-09, INTF-01
**Success Criteria** (what must be TRUE):
  1. User can initialize a new trace via `Tract.open()` and it persists to SQLite storage
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
  3. User can feed API-reported token usage back into Trace via `tract.record_usage()`, and this is preferred over tiktoken counts when available
  4. tiktoken remains the pre-call estimator for budget enforcement; API actuals are the post-call source of truth
  5. `CompiledContext.token_source` accurately reflects whether counts came from tiktoken estimate or API response
**Plans**: 2 plans

Plans:
- [x] 01.1-01-PLAN.md -- CompileSnapshot dataclass, build_message_for_commit() extraction, incremental APPEND fast path, EDIT/annotate/batch invalidation
- [x] 01.1-02-PLAN.md -- record_usage() API, OpenAI/Anthropic dict normalization, two-tier token tracking integration tests

### Phase 1.2: Rename Repo to Tract (INSERTED)
**Goal**: Rename the public SDK entry point from `Repo` to `Tract` and `repo_id` to `tract_id` across all source, tests, and planning docs — clean vocabulary before building more on top
**Depends on**: Phase 1.1
**Requirements**: None (internal refactor)
**Success Criteria** (what must be TRUE):
  1. `Tract.open()` is the entry point; `Repo` class no longer exists
  2. `tract_id` replaces `repo_id` in all models, storage, engine, and protocols
  3. `repo.py` renamed to `tract.py`; `test_repo.py` renamed to `test_tract.py`
  4. All 220 existing tests pass with the new names
  5. Planning docs and MEMORY.md updated to reflect new terminology
**Plans**: 1 plan

Plans:
- [x] 01.2-01-PLAN.md -- Mechanical rename: Repo→Tract, repo_id→tract_id, file renames, test updates, doc updates

### Phase 1.3: Hyperparameter Config Storage (INSERTED)
**Goal**: Every commit can store the LLM generation config (temperature, top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty, model name, max_tokens, etc.) used at call time, giving full provenance for how each piece of context was generated. Enables downstream workflows like exploration/exploitation branching where hyperparams are tuned per-branch and results compared.
**Depends on**: Phase 1.2
**Requirements**: None (new capability -- extends commit model)
**Success Criteria** (what must be TRUE):
  1. User can attach a generation config dict to any commit via `tract.commit(..., generation_config={...})` and retrieve it from the stored commit
  2. Generation config is stored as a flexible schema (JSON blob or similar) that supports any provider's parameters without migration
  3. Generation config is preserved through compile -- `CompiledContext` exposes the configs associated with its commits
  4. Generation config is NOT included in commit hash -- same content with different configs produces the same content_hash, preserving content-addressable dedup
  5. Generation config is queryable: user can filter/retrieve commits by config values (e.g., "all commits with temperature > 0.8")
**Plans**: 1 plan

Plans:
- [x] 01.3-01-PLAN.md -- Add generation_config field to data model, storage, engine, compiler, Tract facade, and integration tests

### Phase 1.4: Remove Aggregation, LRU Compile Cache & Snapshot Patching (INSERTED)
**Goal**: Remove same-role message aggregation (commits are discrete events, not mergeable), replace single-snapshot compile cache with LRU cache keyed by head_hash, and upgrade EDIT/annotate handling from full invalidation to in-memory snapshot patching — so checkout, reset, and future branch switching get cache hits, and EDITs avoid expensive full recompilation
**Depends on**: Phase 1.3
**Requirements**: INFR-06 (performance refinement)
**Success Criteria** (what must be TRUE):
  1. Same-role consecutive messages are preserved as separate messages in compiled output; `_aggregate_messages()` is removed
  2. `CompiledContext.commit_hashes` lists effective commit hashes parallel to messages, populated by the compiler
  3. Multiple compile snapshots are cached simultaneously via LRU; switching HEAD to a previously-compiled position is a cache hit (O(1))
  4. Incremental APPEND still works: new commit's parent matches cached snapshot -> O(1) extend
  5. EDIT commits use snapshot patching: find message by commit hash, replace in-memory, recount tokens -- no chain re-walk
  6. Annotate (priority change to SKIP) uses snapshot patching; annotation clears stale cache entries for other HEADs
  7. batch() remains full cache clear; crash loses cache; DB is always source of truth
  8. verify_cache=True cross-checks every cache hit/patch against full recompile (oracle testing)
  9. All existing 250 tests pass with updated aggregation assertions (zero regressions)
  10. New tests cover LRU eviction, EDIT patching, annotate patching, oracle verification, commit_hashes tracking
**Plans**: 1 plan

Plans:
- [x] 01.4-01-PLAN.md -- Remove aggregation, simplify CompileSnapshot, LRU cache, EDIT/annotate snapshot patching, verify_cache oracle, tests

### Phase 2: Linear History & CLI
**Goal**: Users can inspect, navigate, and manipulate linear commit history through both the SDK and a CLI
**Depends on**: Phase 1
**Requirements**: CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, INTF-02
**Success Criteria** (what must be TRUE):
  1. User can view commit history (log) with per-commit token counts
  2. User can check current state (status) showing HEAD position, branch name, and token budget usage
  3. User can compare any two commits (diff) and see textual differences in content
  4. User can reset HEAD to a previous commit (soft keeps content accessible, hard discards forward history) and checkout a specific commit for read-only inspection
  5. User can perform all of the above via a CLI (`tract log`, `tract status`, `tract diff`, `tract reset`, `tract checkout`) with readable terminal output
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md -- Storage infrastructure (symbolic refs, prefix matching) and navigation operations (reset, checkout)
- [x] 02-02-PLAN.md -- SDK operations (enhanced log with op_filter, status, diff with structured DiffResult)
- [x] 02-03-PLAN.md -- CLI layer with Click + Rich (5 commands, formatting helpers, optional deps)

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
**Plans**: 5 plans

Plans:
- [x] 03-01-PLAN.md -- Branch infrastructure: CommitParentRow schema, DAG utilities (merge_base, ancestors), branch CRUD, compiler multi-parent support
- [x] 03-02-PLAN.md -- LLM client: httpx OpenAI-compatible client, tenacity retry, LLMClient/ResolverCallable protocols, OpenAIResolver
- [x] 03-03-PLAN.md -- Merge strategies: fast-forward, clean merge with branch-blocks, structural conflict detection, LLM-mediated resolution, MergeResult review flow
- [x] 03-04-PLAN.md -- Rebase and cherry-pick: commit replay, EDIT target remapping, semantic safety checks
- [x] 03-05-PLAN.md -- CLI commands: tract branch (list/create/delete), tract switch, tract merge

### Phase 4: Compression
**Goal**: Users can compress context history to fit token budgets while preserving critical information and maintaining history integrity
**Depends on**: Phase 3 (uses LLM client from Phase 3)
**Requirements**: COMP-01, COMP-02, COMP-03, COMP-04
**Success Criteria** (what must be TRUE):
  1. User can compress a range of commits into a summary commit targeting a specific token budget, and the summary is coherent
  2. Pinned commits survive compression verbatim -- their content is unchanged and verifiable by hash
  3. User can reorder commits with semantic safety checks that warn when the reordering changes meaning
  4. User can run garbage collection to remove unreachable commits with configurable retention policies (e.g., keep last N days, keep all pinned)
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md -- CompressionRecord schema (3 new tables), storage repo, domain models, migration v2->v3, summarization prompt
- [x] 04-02-PLAN.md -- Compression engine (compress_range operation, Tract.compress facade, 3 autonomy modes, PINNED preservation, provenance)
- [x] 04-03-PLAN.md -- Compile-time reordering (order parameter, safety checks) and garbage collection (retention policies, reachability)

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

**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md -- Storage foundation: SpawnPointerRow schema (v4 migration), SpawnPointerRepository, SessionContent model, collapse prompt
- [ ] 05-02-PLAN.md -- Session class, spawn/collapse operations (3 inheritance modes, 3 autonomy modes), cross-repo queries, Tract.parent()/children(), crash recovery
- [ ] 05-03-PLAN.md -- Packaging (tract-ai distribution), README documentation, end-to-end integration tests

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 1.1 -> 1.2 -> 1.3 -> 1.4 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundations | 3/3 | Complete | 2026-02-10 |
| 1.1 Compile Cache & Token Tracking | 2/2 | Complete | 2026-02-11 |
| 1.2 Rename Repo to Tract | 1/1 | Complete | 2026-02-11 |
| 1.3 Hyperparameter Config Storage | 1/1 | Complete | 2026-02-11 |
| 1.4 LRU Cache & Snapshot Patching | 1/1 | Complete | 2026-02-11 |
| 2. Linear History & CLI | 3/3 | Complete | 2026-02-12 |
| 3. Branching & Merging | 5/5 | Complete | 2026-02-14 |
| 4. Compression | 3/3 | Complete | 2026-02-16 |
| 5. Multi-Agent & Release | 0/3 | Not started | - |
