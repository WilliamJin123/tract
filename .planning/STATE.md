# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Agents produce better outputs when their context is clean, coherent, and relevant. Trace makes context a managed, version-controlled resource.
**Current focus:** Phase 3 COMPLETE (Branching & Merging). All 5 plans done. Ready for Phase 4.

## Current Position

Phase: 3 of 5 (Branching & Merging) -- COMPLETE
Plan: 5 of 5 in current phase
Status: Phase complete -- All plans 03-01 through 03-05 done
Last activity: 2026-02-15 - Completed 03-05-PLAN.md (CLI Commands, 12 tests, 489 total)

Progress: [################] 80% (16/20 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 5.3m
- Total execution time: 1.45 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 3/3 | 27m | 9m |
| 1.1 | 2/2 | 6m | 3m |
| 1.2 | 1/1 | 3m | 3m |
| 1.3 | 1/1 | 3m | 3m |
| 1.4 | 1/1 | 4m | 4m |
| 2 | 3/3 | 14m | 4.7m |
| 3 | 5/5 | 30m | 6m |

**Recent Trend:**
- Last 5 plans: 03-02 (6m), 03-01 (7m), 03-03 (8m), 03-04 (6m), 03-05 (3m)
- Trend: steady at ~3-8m for Phase 3 complexity

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
- [02-01]: Symbolic HEAD: first commit creates HEAD -> refs/heads/main symbolic ref (git-style)
- [02-01]: update_head backward compat: detects attached/detached/new state and updates correctly
- [02-01]: checkout('-') reads PREV_HEAD before overwriting to avoid self-reference bug
- [02-01]: reset soft == hard in Trace (no working directory); distinction for API compatibility
- [02-01]: Prefix matching minimum 4 characters (same as git)
- [02-01]: operations/ package established for higher-level composites over storage primitives
- [02-02]: StatusInfo is a frozen dataclass (not Pydantic) for lightweight status reporting
- [02-02]: compute_diff() uses SequenceMatcher on serialized message strings for alignment
- [02-02]: op_filter walks through all ancestors but only collects matching ones (limit applies to matches)
- [02-02]: EDIT auto-resolve in diff: when commit_b is EDIT, commit_a defaults to response_to target
- [02-02]: Generation config changes computed from last non-empty config in each chain
- [02-03]: CLI module never imported from tract/__init__.py; only loaded via entry point
- [02-03]: Auto-discovery queries refs table for single tract_id when --tract-id omitted
- [02-03]: Token budget not persisted to DB; CLI opens with default config
- [02-03]: --force guard on hard reset as safety mechanism
- [02-03]: CLI tests use file-backed databases (not :memory:) since CLI opens own connection
- [03-01]: CommitParentRow association table for multi-parent commits (position 0 = first parent, 1 = merged)
- [03-01]: parent_hash column on CommitRow unchanged -- backward compat for first-parent walks
- [03-01]: Schema version bumped 1 -> 2 with auto-migration for existing databases
- [03-01]: commit_hash() includes sorted parent_hashes when extra_parents provided
- [03-01]: Compiler branch-blocks ordering: first-parent chain + second-parent's unique ancestors before merge
- [03-01]: switch() is branch-only (raises BranchNotFoundError on commit hashes); use checkout() for detached HEAD
- [03-02]: Programmatic tenacity.Retrying (not decorator) for per-instance max_retries
- [03-02]: Status codes checked before raise_for_status() for domain-specific errors (LLMAuthError, LLMRateLimitError)
- [03-02]: Duck-typed resolver with getattr() for cross-plan type access (ConflictInfo defined in 03-03)
- [03-02]: Resolution.content_text as string alternative to Resolution.content (BaseModel)
- [03-03]: Merge commit created via CommitEngine.create_merge_commit() with parent_repo parameter
- [03-03]: Pre-loaded content text in ConflictInfo at detect_conflicts() time
- [03-03]: EDIT + APPEND conflict only for pre-merge-base targets (post-merge-base edits not conflicting)
- [03-03]: MergeResult._source_tip_hash/_target_tip_hash for commit_merge parent resolution
- [03-03]: configure_llm() creates default OpenAIResolver; merge() uses it as fallback
- [03-04]: Cherry-pick resolved EDIT content becomes APPEND (no valid response_to on target branch)
- [03-04]: Rebase blocks on branches containing merge commits (cannot flatten multi-parent history)
- [03-04]: Noop rebase when current branch is already ahead of target (returns empty result)
- [03-04]: Replay via CommitEngine.create_commit() -- engine reads HEAD internally for parent assignment

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

## Phase 2 Final Stats

| Plan | Name | Tests | Duration |
|------|------|-------|----------|
| 02-01 | Navigation Infrastructure | 35 | 5m |
| 02-02 | Read Operations (log/status/diff) | 27 | 4m |
| 02-03 | CLI Layer | 30 | 5m |
| **Total** | | **92** | **14m** |

Total test suite: 359 tests passing.

## Phase 3 Final Stats

| Plan | Name | Tests | Duration |
|------|------|-------|----------|
| 03-01 | Branch Infrastructure | 59 | 7m |
| 03-02 | LLM Client Infrastructure | 56 | 6m |
| 03-03 | Merge Strategies | 34 | 8m |
| 03-04 | Rebase & Cherry-Pick | 26 | 6m |
| 03-05 | CLI Commands | 12 | 3m |
| **Total** | | **187** | **30m** |

Total test suite: 489 tests passing.

## Session Continuity

Last session: 2026-02-15
Stopped at: Completed 03-05-PLAN.md (CLI Commands). Phase 3 COMPLETE (5/5 plans).
Resume file: None
