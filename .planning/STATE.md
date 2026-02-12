# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Agents produce better outputs when their context is clean, coherent, and relevant. Trace makes context a managed, version-controlled resource.
**Current focus:** Phase 1.4 COMPLETE. Ready for Phase 2 (Linear History & CLI).

## Current Position

Phase: 2 of 5 (Linear History & CLI)
Plan: 0 of ? in current phase
Status: Phase 1.4 complete -- ready for Phase 2
Last activity: 2026-02-11 - Completed 01.4-01-PLAN.md

Progress: [########......] 40% (8/20 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 5m
- Total execution time: 0.75 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 3/3 | 27m | 9m |
| 1.1 | 2/2 | 6m | 3m |
| 1.2 | 1/1 | 3m | 3m |
| 1.3 | 1/1 | 3m | 3m |
| 1.4 | 1/1 | 4m | 4m |

**Recent Trend:**
- Last 5 plans: 01.1-01 (3m), 01.1-02 (3m), 01.2-01 (3m), 01.3-01 (3m), 01.4-01 (4m)
- Trend: incremental plans on solid foundation execute very fast

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 5-phase structure derived from dependency analysis (Foundations -> Linear History -> Branching -> Compression -> Multi-Agent)
- [Roadmap]: LLM client (INTF-03/04) placed in Phase 3 with branching since semantic merge is the first consumer
- [Roadmap]: INTF-05 (packaging) placed in Phase 5 as final delivery step after all features complete
- [01-01]: Import package renamed from `trace` to `tract` (stdlib shadow fix on Python 3.14). All imports must use `tract`.
- [01-01]: CommitOperation and Priority enums shared between domain models and ORM (not redefined)
- [01-01]: content_type stored as String in DB (not Enum) to support custom types without migration
- [01-01]: Clean layer separation enforced: no SQLAlchemy imports in models/ or protocols.py
- [01-02]: Timezone normalization: _normalize_dt() strips tzinfo for datetime comparison (SQLite stores naive datetimes)
- [01-02]: Edit resolution: latest edit wins when multiple edits target same commit (by created_at)
- [01-02]: Token count distinction: per-commit = raw content, CompiledContext = formatted with message overhead
- [01-02]: ~~Same-role aggregation: consecutive same-role messages concatenated with double newline~~ SUPERSEDED by 01.4-01
- [01-03]: Compile cache keyed by head_hash, cleared on commit/annotate
- [01-03]: Batch implemented by temporarily replacing session.commit with noop, committing on exit
- [01-03]: Tract.open() does not create branch ref upfront; first commit sets HEAD via CommitEngine
- [01.1-01]: Compile cache replaced with CompileSnapshot-based incremental cache (APPEND = O(1) extend, EDIT/annotate/batch = full invalidation)
- [01.1-01]: build_message_for_commit() extracted as public method on DefaultContextCompiler for reuse by Tract incremental path
- [01.1-01]: ~~CompileSnapshot stores both raw and aggregated messages for correct tail aggregation~~ SUPERSEDED by 01.4-01
- [01.1-01]: Time-travel and custom compilers bypass incremental cache entirely
- [01.1-02]: record_usage() validates head_hash match before attempting compile (fail-fast)
- [01.1-02]: record_usage() auto-compiles if no snapshot exists (user doesn't need to call compile() first)
- [01.1-02]: Token source format: "tiktoken:{encoding}" for pre-call, "api:{prompt}+{completion}" for post-call
- [01.2-01]: Repo -> Tract, repo_id -> tract_id, RepoConfig -> TractConfig across all source and tests
- [01.3-01]: generation_config set at commit time only (immutable once written)
- [01.3-01]: record_usage() NOT extended with generation_config (config known before API call)
- [01.3-01]: Copy-on-output AND copy-on-input patterns prevent cache corruption from mutable dicts
- [01.3-01]: Edit-inherits-original: EDIT without generation_config preserves original commit's config
- [01.3-01]: No index on generation_config_json (acceptable at Phase 1 scale)
- [01.3-01]: generation_config NOT part of commit hash (content_hash identical for same content regardless of config)
- [01.4-01]: Same-role aggregation removed entirely; consecutive same-role messages preserved as separate messages
- [01.4-01]: CompileSnapshot simplified: messages + commit_hashes (parallel tuples, no raw/aggregated split)
- [01.4-01]: CompiledContext.commit_hashes lists effective commit hashes parallel to messages
- [01.4-01]: OrderedDict-based LRU cache (maxsize=8 default) replaces single CompileSnapshot
- [01.4-01]: EDIT commits patch cached snapshot in-memory (find by commit_hash, replace message, recount tokens); no cache clear
- [01.4-01]: Annotate SKIP patches snapshot by removing target; un-skip falls back to full recompile; entire cache cleared on annotation, then patched current HEAD re-added
- [01.4-01]: verify_cache=True on Tract.open() cross-checks cache hit/patch against full recompile (oracle testing)
- [01.4-01]: batch() clears entire LRU cache; crash loses cache; DB is always source of truth

### Pending Todos

None.

### Roadmap Evolution

- Phase 1.1 inserted after Phase 1: Incremental Compile Cache & Token Tracking (INSERTED) -- addresses two design issues: (1) full chain walk on every compile adds latency, incremental cache makes APPEND O(1); (2) tiktoken used as sole token source, but API-reported usage should be source of truth post-call
- Phase 1.2 inserted after Phase 1.1: Rename Repo to Tract (INSERTED) -- `Repo` implies a shared container, but each agent's context is self-contained. `Tract` better reflects the domain. Also renames `repo_id` -> `tract_id`. Clean vocabulary before building Phases 2-5 on top.
- Phase 1.3 inserted after Phase 1.2: Hyperparameter Config Storage (INSERTED) -- store LLM generation config (temperature, top_p, top_k, repetition_penalty, etc.) with commits for full call provenance. Enables exploration/exploitation branching pattern where hyperparams are tuned per-branch. Data model extension belongs before Phase 2 so history/diff/log can display configs from day one.
- Phase 1.4 inserted after Phase 1.3: LRU Compile Cache & Snapshot Patching (INSERTED) -- replace single-snapshot cache with LRU keyed by head_hash, EDIT/annotate snapshot patching instead of full invalidation. Checkout/reset/branch-switch get cache hits; EDITs avoid expensive full recompilation.

### Blockers/Concerns

- ~~Phase 1: Edit commit semantics (override vs in-place)~~ RESOLVED: Full commit replacement (new commit supersedes original via reply_to). No in-place mutation.
- ~~Phase 1: stdlib `trace` module shadowing~~ RESOLVED: Package renamed to `tract`.
- WATCH: External linter keeps renaming `tract` back to `trace` in working tree. The git commits have correct `tract` imports. If this affects future plan execution, may need to configure ruff to ignore this rename.
- Phase 3: Semantic merge quality is unproven for natural language context -- research flag for plan-phase
- Phase 4: Compression is inherently lossy (3-55% degradation in research) -- need validation strategy
- Phase 5: SQLite concurrent write behavior under multi-agent load is untested -- research flag for plan-phase

## Phase 1 Final Stats

| Plan | Name | Tests | Duration |
|------|------|-------|----------|
| 01-01 | Domain Models + Storage | 87 | 8m |
| 01-02 | Engine Layer | 66 | 15m |
| 01-03 | Tract Class + Public API | 47 | 4m |
| **Total** | | **200** | **27m** |

All 5 Phase 1 success criteria verified end-to-end.

## Phase 1.1 Final Stats

| Plan | Name | Tests | Duration |
|------|------|-------|----------|
| 01.1-01 | Incremental Compile Cache | 7 | 3m |
| 01.1-02 | record_usage() API | 13 | 3m |
| **Total** | | **20** | **6m** |

Total test suite: 220 tests passing.

## Phase 1.2 Final Stats

| Plan | Name | Tests | Duration |
|------|------|-------|----------|
| 01.2-01 | Repo -> Tract rename | 220 (all pass) | 3m |
| **Total** | | **0 new** | **3m** |

Total test suite: 220 tests passing.

## Phase 1.3 Final Stats

| Plan | Name | Tests | Duration |
|------|------|-------|----------|
| 01.3-01 | Hyperparameter Config Storage | 30 | 3m |
| **Total** | | **30** | **3m** |

Total test suite: 250 tests passing.

## Phase 1.4 Final Stats

| Plan | Name | Tests | Duration |
|------|------|-------|----------|
| 01.4-01 | LRU Compile Cache & Snapshot Patching | 17 | 4m |
| **Total** | | **17** | **4m** |

Total test suite: 267 tests passing.

## Session Continuity

Last session: 2026-02-11
Stopped at: Completed 01.4-01-PLAN.md (LRU Compile Cache & Snapshot Patching). Phase 1.4 complete (1/1 plans).
Resume file: None
