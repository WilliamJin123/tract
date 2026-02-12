---
phase: 02-linear-history-cli
plan: 02
subsystem: api
tags: [difflib, dataclass, status, diff, log, operations]

# Dependency graph
requires:
  - phase: 01-foundations
    provides: "Commit chain, compile(), Tract facade, CommitInfo, CompiledContext"
  - phase: 01.4-lru-compile-cache-snapshot-patching
    provides: "LRU compile cache for _compile_at() efficiency"
  - phase: 02-01
    provides: "Symbolic refs, prefix matching, reset/checkout, operations/ package, resolve_commit"
provides:
  - "StatusInfo dataclass for tract status inspection"
  - "DiffResult, MessageDiff, DiffStat structured diff output"
  - "compute_diff() function using difflib.SequenceMatcher"
  - "Tract.log() with op_filter and default limit 20"
  - "Tract.status() returning compiled token count, branch state, budget info"
  - "Tract.diff() with EDIT auto-resolution and prefix support"
  - "Tract._compile_at() helper for arbitrary commit compilation"
affects:
  - "02-03 (CLI layer wraps these facade methods)"
  - "03-branching (diff needed for branch comparison)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Operations modules pattern: pure data models in operations/, computation delegated to Tract facade"
    - "difflib.SequenceMatcher for message alignment in structured diff"
    - "_compile_at() for LRU-cached compilation at arbitrary commits"
    - "op_filter chain walk: filter during parent traversal, continue through non-matching"

key-files:
  created:
    - "src/tract/operations/history.py"
    - "src/tract/operations/diff.py"
    - "tests/test_operations.py"
  modified:
    - "src/tract/tract.py"
    - "src/tract/__init__.py"
    - "src/tract/storage/repositories.py"
    - "src/tract/storage/sqlite.py"

key-decisions:
  - "StatusInfo is a frozen dataclass (not Pydantic) -- lightweight, no validation overhead"
  - "compute_diff() uses SequenceMatcher on serialized message strings for alignment"
  - "op_filter implemented at chain walk level (continues through non-matching commits)"
  - "EDIT auto-resolve in diff: when commit_b is EDIT, commit_a defaults to response_to target"
  - "Generation config changes computed from last non-empty config in each chain"

patterns-established:
  - "Operations data models: frozen dataclasses in operations/ modules, exported from tract package"
  - "Facade delegation: status/diff logic lives in Tract methods, data models are decoupled"

# Metrics
duration: 4min
completed: 2026-02-12
---

# Phase 02 Plan 02: Read Operations Summary

**Structured log/status/diff operations with op_filter, StatusInfo dataclass, DiffResult with unified diff lines and EDIT auto-resolution**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-12T18:03:32Z
- **Completed:** 2026-02-12T18:07:32Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- StatusInfo, DiffResult, MessageDiff, DiffStat data models for structured read operations
- Tract.log() enhanced with op_filter parameter and default limit 20
- Tract.status() returns complete state (HEAD, branch, detached, tokens, budget, recent commits)
- Tract.diff() with EDIT auto-resolution, prefix support, and structured unified diff output
- 27 new tests covering all log, status, and diff scenarios (329 total)

## Task Commits

Each task was committed atomically:

1. **Task 1: History and diff operations modules + data models** - `d50c75a` (feat)
2. **Task 2: Tract facade methods (log, status, diff) + exports + tests** - `2024cc4` (feat)

## Files Created/Modified
- `src/tract/operations/history.py` - StatusInfo frozen dataclass
- `src/tract/operations/diff.py` - DiffResult, MessageDiff, DiffStat dataclasses + compute_diff()
- `src/tract/tract.py` - Enhanced log(), new status(), _compile_at(), diff() methods
- `src/tract/__init__.py` - Added StatusInfo, DiffResult, MessageDiff, DiffStat exports
- `src/tract/storage/repositories.py` - Added op_filter parameter to get_ancestors() ABC
- `src/tract/storage/sqlite.py` - Implemented op_filter in SQLite get_ancestors()
- `tests/test_operations.py` - 27 tests: 5 log, 8 status, 14 diff

## Decisions Made
- StatusInfo uses frozen dataclass (not Pydantic) for lightweight status reporting
- compute_diff() serializes messages to text for SequenceMatcher alignment rather than comparing by field
- op_filter walks through all ancestors but only collects matching ones (limit applies to matches only)
- EDIT auto-resolution: if commit_b is EDIT and commit_a is None, diff against the edit target
- Generation config changes use last non-empty config from each chain for comparison

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All three read operations (log, status, diff) available for CLI wrapping in Plan 03
- StatusInfo, DiffResult types exported for CLI formatting
- 329 total tests passing, clean foundation for CLI development

---
*Phase: 02-linear-history-cli*
*Completed: 2026-02-12*
