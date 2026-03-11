# Improvement Rounds 1-8 (2026-03-10)

Start: 1888 tests, ~35 cookbook examples, ~57 test files
End: 2068 tests, 55+ cookbook examples, 65+ test files

---

## Round 1 — Commit Search, Cross-Branch Diff, Merge Strategies, Tool Summarization
Commit: de5a011

### Features
- **t.find() / t.find_one()**: Search commits by content text, tags, content type, or metadata predicates. Returns matching CommitInfo objects from the DAG ancestry. Enables querying the context history programmatically (e.g. "find all commits tagged 'critical'").
- **t.compare(branch_a, branch_b)**: Cross-branch diff without switching HEAD. Returns a DiffResult with token deltas, message-level changes, and serialization via to_dict()/to_json(). Useful for A/B testing branches or reviewing what diverged.
- **MergeStrategy (ours/theirs/auto)**: When merging branches with conflicting content, `strategy="ours"` keeps current branch content, `strategy="theirs"` takes source branch content, `strategy="auto"` uses LLM resolution. Passed through t.merge(strategy=...).
- **Tool summarization in loop**: ToolSummarizationConfig was a model that existed but was never consumed. Now wired into _commit_tool_result() — when a tool result exceeds auto_threshold tokens, it's summarized via LLM before committing. Prevents context explosion from large tool outputs.
- **DiffResult.to_dict() / .to_json()**: Serialization methods on diff results for programmatic consumption.

### Cookbooks (3)
- cookbook/workflows/05_ecomm_pipeline.py — 5-stage ecomm pipeline with A/B branching and middleware gates
- cookbook/workflows/06_coding_with_tests.py — TDD workflow with branch-per-attempt and quality gates
- cookbook/agent/10_collaborative_research.py — 3-specialist agents with session spawn/collapse

### Tests
- 14 tests in test_operations.py (TestCompare class)
- Updated test_spawn.py for selective spawn error behavior

---

## Round 2 — Retry Backoff, Middleware Events, Rich Cookbooks
Commit: 689d998

### Features
- **RetryConfig**: Frozen dataclass with max_retries, initial_delay, max_delay, backoff_factor, jitter, retryable_errors. Exponential backoff with optional jitter for LLM call resilience. Set via t.open(retry=RetryConfig(...)) or per-call on chat()/generate(). Every competitor (LangChain, Agno, CrewAI) has retry — now tract does too.
- **_retry_with_backoff() / _aretry_with_backoff()**: Module-level helpers that wrap any callable with retry logic. ContentValidationError and BlockedError are never retried. Used by chat(), generate(), run_loop(), and arun_loop().
- **4 new middleware events**: pre_generate, post_generate, pre_tool_execute, post_tool_execute. These fire in the loop around LLM calls and tool executions, enabling monitoring, gating, and instrumentation at every operation boundary.
- **MiddlewareContext.pending type broadened**: Changed from BaseModel | None to BaseModel | dict | None to support passing dicts for generate/tool events.

### Cookbooks (3)
- cookbook/error_handling/01_recovery_strategies.py — 5 patterns: checkpoint/rollback, branch isolation, compression degradation, circuit breaker, multi-strategy fallback
- cookbook/config_and_middleware/06_observability.py — LLM logging, budget dashboard, audit trail, stage timing
- cookbook/persistence/01_checkpoints_and_recovery.py — Named checkpoints, cross-session persistence, branch checkpoints

### Tests
- 36 tests in test_retry_backoff.py covering RetryConfig, backoff helpers, and integration

### Bug Fix
- acompress_tool_calls() was missing turn_count parameter and tuple() wrapping — would crash at runtime. Fixed to match sync compress_tool_calls().

---

## Round 3 — Directive Templates, Workflow Profiles, Selective Spawn
Commit: 123775d

### Features
- **DirectiveTemplate system**: Frozen dataclass with name, template (string with {placeholder} syntax), description, parameters. render(**kwargs) does substitution. 9 built-in templates: review_protocol, safety_guardrails, output_format, research_protocol, citation_required, code_review, implementation_plan, brand_voice, conversion_optimization. Registry API: list_templates(), get_template(), register_template(). Applied via t.apply_template("persona", role="analyst").
- **WorkflowProfile bundles**: Frozen dataclass bundling config, directive_templates, directives, tool_profile, and stages (dict of stage_name -> stage config). 3 built-in profiles: CODING (design->implement->test->review), RESEARCH (ingest->organize->synthesize->validate), ECOMMERCE (research->creative->campaign->analysis->optimize). Applied via t.load_profile("coding"), t.apply_stage("design"). Each stage sets specific configs and directives appropriate for that workflow phase.
- **Selective spawn inheritance**: Session.spawn() now supports include_tags, exclude_tags, include_types, filter_func parameters. Instead of cloning entire parent context, you can selectively inherit only relevant commits. 3-pass implementation: determine inclusions -> filter dangling EDITs -> replay chronologically.

### Cookbooks (2)
- cookbook/optimization/01_budget_management.py — Per-stage budgets, auto-compression, strategy comparison
- cookbook/testing/01_mocking_patterns.py — MockLLMClient, fixtures, snapshot testing, 10 patterns/15 tests

### Tests
- 41 tests in test_templates.py
- 44 tests in test_profiles.py
- 18 tests in test_spawn_selective.py

---

## Round 4 — Snapshots, Health Checks, Sliding Window Compression
Commit: 0605dbd

### Features
- **t.snapshot(name) / t.list_snapshots() / t.restore_snapshot(name)**: Named snapshots of the current HEAD position. Restore resets HEAD to the snapshot point. Useful for checkpointing before risky operations. Stored as refs in the DAG (lightweight — just a pointer).
- **t.health() -> HealthReport**: DAG validation that checks blob integrity (all content_hash references resolve), parent integrity (all parent references exist), reachability (orphan detection), and branch HEAD validity. Returns a HealthReport with healthy flag, counts, and warnings. Use after compression or complex merge operations to verify DAG consistency.
- **Sliding window compression**: New strategy="sliding_window" on t.compress(). Keeps the last N commits (window_size) intact and compresses everything before the window into a summary. Alternative to partition-around-pinned for recency-biased workflows where recent context matters more.

### Cookbooks (1)
- cookbook/workflows/07_full_showcase.py — 25+ features in a single end-to-end product analysis workflow

### Tests
- 18 tests in test_snapshots.py
- 26 tests in test_health.py
- 21 tests in test_sliding_window.py
- 19 additional tests in test_compression_lifecycle.py

---

## Round 5 — Loop Budget/Validator, ConfigIndex Validation, Integration Tests
Commits: 3c40dd1, 32e1769

### Features
- **LoopConfig.step_budget**: Max total tokens across all loop steps. When accumulated usage exceeds the budget, the loop stops gracefully with result.budget_exhausted == True. Prevents runaway token consumption in production.
- **LoopConfig.tool_validator**: Callable (tool_name, args) -> (ok, error_msg) that validates tool arguments before execution. Invalid calls are committed as errors without executing the tool. Enables safety controls (block dangerous tools, validate schemas) without custom middleware.
- **LoopResult.budget_exhausted**: Property that returns True when the loop stopped due to token budget exhaustion.
- **ConfigIndex consumer validation**: ConfigIndex was listed as "Not yet validated by consumers" in CONSUMERS.md. Created comprehensive cookbook + tests proving it works for real workflow patterns: precedence, branch isolation, invalidation, middleware queries, per-stage config.

### Cookbooks (5)
- cookbook/config_and_middleware/07_config_index_patterns.py — 6 patterns for DAG-based config resolution
- cookbook/getting_started/06_agent_loop.py — Minimal 5-minute agent loop quickstart
- cookbook/agent/11_profile_staged_agent.py — WorkflowProfile stages driving agent behavior
- cookbook/error_handling/02_graceful_degradation.py — 5 failure handling patterns
- cookbook/reference/09_batch_operations.py — Comprehensive batch() guide

### Tests
- 17 tests in test_loop_config.py (step_budget, tool_validator)
- 26 tests in test_integration_combined.py (combining snapshots+health+profiles+batch+find+compare+middleware)

---

## Round 6 — Auto-Compress in Loop, StepMetrics, Production Monitoring
Commits: a82ad93, 3308b72, 85c2997

### Features
- **LoopConfig.auto_compress_threshold**: Float 0.0-1.0. When compiled context exceeds this fraction of max_tokens, the loop automatically triggers sliding_window compression before the next LLM call. Prevents context overflow errors in long-running agents. Falls back gracefully if compression fails.
- **StepMetrics**: Frozen dataclass with per-step observability: step number, duration_ms (wall clock), llm_duration_ms (LLM call time), tool_count, tool_names, context_tokens (at step start), compressed flag (whether auto-compress fired). Populated automatically in both run_loop() and arun_loop().
- **LoopResult.step_metrics**: Tuple of StepMetrics for every step. Plus properties: total_duration_ms, total_llm_duration_ms, compressions_triggered.
- **CONSUMERS.md update**: Promoted 11 APIs to Stable (ConfigIndex, find, compare, MergeStrategy, RetryConfig, templates, profiles, snapshots, health, batch). Moved chat/generate and middleware to In flux. Cleared "Not yet validated" section.

### Cookbooks (1)
- cookbook/optimization/02_production_monitoring.py — 6 patterns: token tracking, audit trail, health dashboard, budget dashboard, error rate monitoring, context growth alerting

### Tests
- 14 tests in test_loop_auto_compress.py (auto-compress trigger, StepMetrics population)

---

## Round 7 — Context Export/Import, Supervisor-Worker, Session Hardening
Commits: 5bbb703, 43d4e21

### Features
- **t.export_state(include_blobs=True) -> dict**: Serializes the current branch's full DAG as a portable JSON-serializable dict. Includes all commits (content payloads, metadata, parent references, priority annotations), branch info, and timestamps. With include_blobs=False, produces lightweight metadata-only export. This is a unique tract differentiator — no other LLM framework can serialize their entire context DAG.
- **t.load_state(state) -> int**: Replays exported commits into the current tract as new commits, preserving content types, metadata, and priority annotations via validate_content(). Returns number of commits loaded. Enables cross-agent context transfer, file-based persistence, and debug replay.

### Cookbooks (1)
- cookbook/agent/12_supervisor_worker.py — 5 multi-agent orchestration patterns:
  1. Basic supervisor-worker (spawn workers, collapse results)
  2. Parallel workers with quality gate (3 approaches, compare, select best)
  3. Pipeline (sequential A->B->C with collapse handoff)
  4. Debate (opposing positions, synthesized conclusion)
  5. Hierarchical delegation (director -> sub-supervisors -> workers)

### Tests
- 27 tests in test_export_import.py (export, load, round-trip, edge cases)
- 16 tests in test_session_advanced.py (spawn/collapse edge cases, compile_at, deep chains, many children)

---

## Round 8 — API Polish, Facade Passthrough, Discoverability
Commits: 5f6caf5, 2041e62

### Features
- **Tract.run() wired to all LoopConfig fields**: The convenience method t.run() now accepts step_budget, tool_validator, and auto_compress_threshold as direct parameters. Previously these were only accessible by constructing LoopConfig and calling run_loop() directly. Users no longer need to drop down to the low-level API for budget control or tool validation.
- **StepMetrics exported from tract package**: `from tract import StepMetrics` now works. Previously required `from tract.loop import StepMetrics`.

### Cookbooks (2)
- cookbook/persistence/02_context_portability.py — 6 patterns: basic export/import, JSON file persistence, cross-agent transfer, selective export, branch export, export after compression
- cookbook/reference/10_content_type_hints.py — 5 patterns: built-in type behaviors, non-compilable types, compression priority ordering, reasoning auto-skip, custom types with hints

### Tests
- 15 tests in test_run_facade.py (step_budget passthrough, tool_validator passthrough, auto_compress passthrough, StepMetrics in run() results, combined scenarios)
