---
phase: 13-unified-operation-events-compile-records
plan: 01
subsystem: database
tags: [sqlalchemy, orm, schema-migration, sqlite, repository-pattern]

# Dependency graph
requires:
  - phase: 04-compression
    provides: "Original compression tables (CompressionRow, CompressionSourceRow, CompressionResultRow)"
  - phase: 06-policy-engine
    provides: "Schema v5 with policy tables"
provides:
  - "OperationEventRow + OperationCommitRow tables for unified operation tracking"
  - "CompileRecordRow + CompileEffectiveRow tables for compile auditing"
  - "OperationEventRepository ABC + SqliteOperationEventRepository"
  - "CompileRecordRepository ABC + SqliteCompileRecordRepository"
  - "Schema v6 with migration from v5"
affects:
  - "13-02 (operation wiring -- consumes new repositories)"
  - "13-03 (compile record wiring -- consumes new repositories)"
  - "operations/compression.py (needs TYPE_CHECKING update for new repo type)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Unified operation event pattern (compress/reorganize/import as event_type)"
    - "Compile record pattern (head_hash + effective commits for audit trail)"
    - "Raw SQL in migration chains when ORM classes are removed"

key-files:
  created: []
  modified:
    - "src/tract/storage/schema.py"
    - "src/tract/storage/engine.py"
    - "src/tract/storage/repositories.py"
    - "src/tract/storage/sqlite.py"
    - "src/tract/tract.py"
    - "src/tract/session.py"
    - "src/tract/operations/spawn.py"
    - "tests/test_compression_storage.py"

key-decisions:
  - "v2->v3 migration rewritten with raw SQL since ORM classes removed"
  - "Consumer files (tract.py, session.py, spawn.py) updated to unblock import chain"
  - "OperationCommitRow uses 3-column composite PK (event_id, commit_hash, role)"
  - "Indexed original_tokens and compressed_tokens columns on OperationEventRow"

patterns-established:
  - "Migration helper functions for data migration between schema versions"
  - "Raw SQL CREATE TABLE in migrations for removed ORM classes"

# Metrics
duration: 7min
completed: 2026-02-20
---

# Phase 13 Plan 01: Unified Storage Layer Summary

**4 new ORM tables (OperationEvent/Commit, CompileRecord/Effective), 2 repository ABCs + SQLite impls, schema v5->v6 migration with data migration, 3 old compression tables removed**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-20T22:04:28Z
- **Completed:** 2026-02-20T22:12:24Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Added 4 new schema tables: OperationEventRow, OperationCommitRow, CompileRecordRow, CompileEffectiveRow
- Implemented OperationEventRepository (9 methods) and CompileRecordRepository (5 methods) ABCs + SQLite implementations
- Schema v5->v6 migration with data migration from old compression tables and drop of old tables
- Rewrote test_compression_storage.py with 31 tests covering all new repositories, schema, migration, models, and prompts

## Task Commits

Each task was committed atomically:

1. **Task 1: New schema tables + remove old tables + migration** - `5fb45fb` (feat)
2. **Task 2: Repository interfaces + SQLite implementations + tests** - `a572638` (feat)

## Files Created/Modified
- `src/tract/storage/schema.py` - 4 new table classes, 3 old removed
- `src/tract/storage/engine.py` - v5->v6 migration, _migrate_compressions_v5_to_v6 helper, raw SQL v2->v3
- `src/tract/storage/repositories.py` - OperationEventRepository + CompileRecordRepository ABCs, CompressionRepository removed
- `src/tract/storage/sqlite.py` - SqliteOperationEventRepository + SqliteCompileRecordRepository, SqliteCompressionRepository removed
- `src/tract/tract.py` - Updated import: SqliteCompressionRepository -> SqliteOperationEventRepository
- `src/tract/session.py` - Updated import: SqliteCompressionRepository -> SqliteOperationEventRepository
- `src/tract/operations/spawn.py` - Updated import: SqliteCompressionRepository -> SqliteOperationEventRepository
- `tests/test_compression_storage.py` - Fully rewritten with 31 tests for new storage layer

## Decisions Made
- Used raw SQL for v2->v3 migration since ORM classes (CompressionRow etc.) were removed from schema.py. This ensures the full v2->v3->v4->v5->v6 migration chain still works end-to-end.
- Updated consumer files (tract.py, session.py, spawn.py) that import SqliteCompressionRepository at module level, since the import chain cascades through tract/__init__.py and would break all imports otherwise.
- OperationCommitRow uses a 3-column composite PK (event_id, commit_hash, role) to allow the same commit to appear as both source and result in different events.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated consumer imports to unblock import chain**
- **Found during:** Task 1 (schema + engine changes)
- **Issue:** Removing SqliteCompressionRepository from sqlite.py broke imports in tract.py, session.py, and spawn.py which import at module level. Since tract/__init__.py imports Tract, the entire package became unimportable.
- **Fix:** Updated import statements in tract.py, session.py, and operations/spawn.py to use SqliteOperationEventRepository instead of SqliteCompressionRepository
- **Files modified:** src/tract/tract.py, src/tract/session.py, src/tract/operations/spawn.py
- **Verification:** `python -c "from tract.storage.schema import OperationEventRow"` succeeds
- **Committed in:** 5fb45fb (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for import chain to work. No scope creep -- these files are downstream consumers that will be fully updated in subsequent plans.

## Issues Encountered
- v5->v6 migration test initially failed because Base.metadata.create_all() creates empty _trace_meta table (no rows), so scalar_one() raised NoResultFound. Fixed by using scalar_one_or_none() with conditional insert.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Storage layer foundation complete: 4 new tables, 2 repository ABCs, 2 SQLite implementations
- Ready for Plan 02 (operation wiring) and Plan 03 (compile record wiring)
- Note: operations/compression.py still uses CompressionRepository in TYPE_CHECKING imports -- will be updated in subsequent plans
- Note: tract.py stores the new repo as `_compression_repo` (old variable name) -- will be renamed in subsequent plans

---
*Phase: 13-unified-operation-events-compile-records*
*Completed: 2026-02-20*
