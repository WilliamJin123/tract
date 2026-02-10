---
phase: 01-foundations
plan: 02
subsystem: engine-layer
tags: [hashing, sha256, tiktoken, token-counting, commit-engine, context-compiler, edit-resolution, time-travel]
requires:
  - 01-01 (domain models, storage layer, repository interfaces)
provides:
  - Deterministic canonical JSON hashing (content-addressable storage)
  - TiktokenCounter with o200k_base encoding and message overhead
  - CommitEngine with parent chain, blob dedup, edit validation, budget enforcement
  - DefaultContextCompiler with edit resolution, priority filtering, time-travel
  - Type-to-role mapping for all 7 content types
  - Same-role consecutive message aggregation
affects:
  - 01-03 (Repo facade will wrap CommitEngine and DefaultContextCompiler)
  - 02-XX (linear history will extend commit chain walking)
  - 03-XX (branching/merge will extend compiler for multi-branch compilation)
  - 04-XX (compression will interact with compiler pipeline)
tech-stack:
  added:
    - tiktoken (token counting, o200k_base encoding)
  patterns:
    - Content-addressable hashing (SHA-256 of canonical JSON)
    - Engine pattern (CommitEngine orchestrates repos + hashing + validation)
    - Compiler pattern (DefaultContextCompiler walks DAG, resolves edits, maps roles)
    - Timezone normalization for SQLite datetime comparisons
key-files:
  created:
    - src/trace_context/engine/__init__.py
    - src/trace_context/engine/hashing.py
    - src/trace_context/engine/tokens.py
    - src/trace_context/engine/commit.py
    - src/trace_context/engine/compiler.py
    - src/trace_context/models/compiled.py
    - tests/test_engine/__init__.py
    - tests/test_engine/test_hashing.py
    - tests/test_engine/test_tokens.py
    - tests/test_engine/test_commit.py
    - tests/test_engine/test_compiler.py
  modified: []
decisions:
  - "Timezone normalization: _normalize_dt() strips tzinfo for comparison since SQLite stores naive datetimes"
  - "Edit resolution: latest edit wins when multiple edits target same commit (by created_at)"
  - "Same-role aggregation: consecutive same-role messages concatenated with double newline"
  - "ToolIOContent formatting: 'Tool {direction}: {name}' header + indented JSON payload"
  - "FreeformContent formatting: pretty-printed JSON of payload dict"
  - "Token count distinction: per-commit token_count = raw content, CompiledContext.token_count = formatted with message overhead"
metrics:
  duration: 15m
  completed: 2026-02-10
---

# Phase 01 Plan 02: Commit Engine and Context Compiler Summary

**One-liner:** CommitEngine with SHA-256 content-addressable storage, edit validation, token budget enforcement; DefaultContextCompiler with edit resolution, priority filtering, time-travel, and 7-type role mapping.

## What Was Built

### Task 1: Deterministic Hashing, Token Counting, and Commit Engine

**Hashing (engine/hashing.py):**
- `canonical_json()`: Deterministic JSON serialization with sorted keys, compact separators, UTF-8 encoding
- `content_hash()`: SHA-256 of canonical JSON payload, returns hex digest
- `commit_hash()`: SHA-256 of structured commit data dict (content_hash, parent_hash, content_type, operation, timestamp, optional reply_to)

**Token Counting (engine/tokens.py):**
- `TiktokenCounter`: Production counter using tiktoken with o200k_base fallback, cached encoding instance, OpenAI message overhead (3 tokens/message + 1/name + 3 response primer)
- `NullTokenCounter`: Zero-returning stub for testing

**Commit Engine (engine/commit.py):**
- `CommitEngine.create_commit()`: Full commit creation pipeline -- serialize content, compute hashes, store blob (dedup), validate edits, enforce budget, create commit row, update HEAD, auto-annotate
- `CommitEngine.get_commit()`: Fetch and convert CommitRow to CommitInfo
- `CommitEngine.annotate()`: Create priority annotations with validation
- `extract_text_from_content()`: Unified text extraction from all 7 content types
- Edit validation: EDIT requires reply_to, target must exist, cannot edit an EDIT
- Token budget: WARN (log + continue), REJECT (raise BudgetExceededError), CALLBACK (invoke user function)
- Auto-PINNED annotation for instruction content type

### Task 2: Default Context Compiler

**Compiled Model (models/compiled.py):**
- `CompileOptions`: Pydantic model for as_of, up_to, include_edit_annotations, type_to_role_map, aggregate_same_role

**Context Compiler (engine/compiler.py):**
- `DefaultContextCompiler.compile()`: 8-step compilation algorithm:
  1. Walk commit chain (head to root, reverse to chronological order)
  2. Build edit resolution map (reply_to -> latest edit commit)
  3. Build priority map (annotations -> fallback to DEFAULT_TYPE_PRIORITIES)
  4. Build effective commit list (skip EDITs, skip SKIP priority)
  5. Map content types to roles (dialogue uses own role, tool_io -> "tool", instruction -> "system")
  6. Build messages (load blob, parse content, extract text, optional [edited] marker)
  7. Aggregate consecutive same-role messages (concatenate with double newline)
  8. Count tokens on compiled output (message overhead included)

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_hashing.py | 18 | Canonical JSON, content hash, commit hash, property tests |
| test_tokens.py | 13 | TiktokenCounter (text, messages, encoding fallback), NullTokenCounter |
| test_commit.py | 25 | Create, parent chain, dedup, edits, budget modes, annotate |
| test_compiler.py | 30 | Core compilation, role mapping, edits, priorities, time-travel, aggregation, tokens |
| **Total new** | **87** | |
| **Total suite** | **153** | All Plan 01 + Plan 02 tests pass together |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Linter renamed package directory from trace_context to trace**
- **Found during:** Task 2 setup (tests failed to import)
- **Issue:** An external linter renamed `src/trace_context/` to `src/trace/` and changed all imports from `trace_context` to `trace`, which shadows the stdlib `trace` module on Python 3.14
- **Fix:** Renamed directory back to `src/trace_context/`, restored all imports to `trace_context`, verified pyproject.toml already had correct config in git HEAD
- **Files affected:** All src/ and tests/ files (import paths only, no logic changes)
- **Note:** The git HEAD already had correct `trace_context` naming; the linter only affected the working tree

**2. [Rule 1 - Bug] Timezone-aware vs naive datetime comparison in compiler**
- **Found during:** Task 2 test execution
- **Issue:** `as_of` parameter is timezone-aware (datetime.now(timezone.utc)) but SQLite stores naive datetimes, causing TypeError on comparison
- **Fix:** Added `_normalize_dt()` helper that strips tzinfo for comparison, applied to all datetime comparisons in compiler (walk chain, edit map, priority map)
- **Files modified:** src/trace_context/engine/compiler.py
- **Commit:** 7d6e612

**3. [Rule 1 - Bug] Recursive fixture dependency in test_compiler.py**
- **Found during:** Task 2 test execution
- **Issue:** Local `engine` fixture conflicted with conftest.py's `engine` fixture (SQLAlchemy Engine vs CommitEngine), causing pytest recursive dependency error
- **Fix:** Renamed local fixture to `commit_engine` and updated all test method signatures
- **Files modified:** tests/test_engine/test_compiler.py
- **Commit:** 7d6e612

## Key Implementation Details

- **Content-addressable storage**: Same content produces same blob hash; blob stored once regardless of how many commits reference it
- **Commit identity**: Commit hash includes content_hash + parent_hash + content_type + operation + timestamp + optional reply_to. Same content at different positions = different commit hash
- **Edit semantics**: Edit commits are "overlay" commits -- they appear in the chain but during compilation, the original commit's position shows the edit's content. Edit commits themselves are skipped as standalone messages
- **Priority filtering**: Applied after edit resolution. A SKIP annotation on the original commit hides it even if it was edited
- **Time-travel**: `as_of` (datetime cutoff) and `up_to` (commit hash cutoff) are mutually exclusive. Both filter the commit chain before edit resolution and priority mapping

## Next Phase Readiness

Plan 01-02 is complete. Plan 01-03 (Repo facade) can proceed. It will:
- Wrap CommitEngine and DefaultContextCompiler into a user-facing Repo class
- Add batch context manager for atomic multi-commit operations
- Wire up RepoConfig -> engine construction
- Provide the top-level SDK interface

No blockers or concerns for 01-03.
