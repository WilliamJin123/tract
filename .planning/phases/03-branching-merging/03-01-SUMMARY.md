# Phase 3 Plan 01: Branch Infrastructure Summary

**One-liner:** Pointer-based branching with DAG traversal, multi-parent schema, and compiler merge-walk support.

## Execution Stats

- **Duration:** ~7 minutes
- **Completed:** 2026-02-15
- **Tasks:** 2/2
- **New tests:** 59
- **Total tests passing:** 417

## What Was Built

### Task 1: Schema, repositories, hashing, and compiler updates
- **CommitParentRow** association table for merge commit parents (position-ordered)
- **CommitParentRepository** ABC with add_parent/get_parents/add_parents
- **SqliteCommitParentRepository** SQLite implementation
- **init_db()** migration: v1 -> v2 schema with commit_parents table creation
- **commit_hash()** extended with `extra_parents` parameter for merge commit identity
- **DefaultContextCompiler** updated with `parent_repo` parameter and branch-blocks ordering for merge commits

### Task 2: Branch operations, DAG utilities, exceptions, and Tract facade
- **4 new exceptions:** BranchExistsError, BranchNotFoundError, InvalidBranchNameError, UnmergedBranchError
- **BranchInfo** Pydantic model (name, commit_hash, is_current, commit_count, description)
- **operations/branch.py:** validate_branch_name (git-style rules), create_branch, delete_branch (with unmerged guard), list_branches
- **operations/dag.py:** find_merge_base (BFS), get_all_ancestors, get_branch_commits, is_ancestor
- **Tract facade:** branch(), switch(), list_branches(), delete_branch()
- **Exports:** All new types added to tract.__init__.__all__

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Association table for multi-parent (not JSON column) | Referential integrity, JOIN-able queries, standard relational pattern |
| parent_hash column unchanged | Backward compat -- first-parent walk still works via parent_hash for linear history |
| switch() branch-only (not checkout-like) | Prevents silently detaching HEAD on commit hashes -- use checkout() for that |
| Branch-blocks ordering in compiler | All first-parent commits in order, then second-parent's unique commits before merge point |
| Schema version bump 1->2 with migration | Existing v1 databases auto-migrate by creating commit_parents table |

## Files Created

| File | Purpose |
|------|---------|
| src/tract/models/branch.py | BranchInfo Pydantic model |
| src/tract/operations/branch.py | Branch CRUD operations |
| src/tract/operations/dag.py | DAG utilities (merge base, ancestors) |
| tests/test_branch.py | 59 comprehensive tests |

## Files Modified

| File | Changes |
|------|---------|
| src/tract/storage/schema.py | Added CommitParentRow table |
| src/tract/storage/repositories.py | Added CommitParentRepository ABC |
| src/tract/storage/sqlite.py | Added SqliteCommitParentRepository |
| src/tract/storage/engine.py | Schema v2 migration |
| src/tract/engine/hashing.py | extra_parents in commit_hash() |
| src/tract/engine/compiler.py | parent_repo + merge-walk in _walk_chain() |
| src/tract/exceptions.py | 4 new branch exception classes |
| src/tract/tract.py | parent_repo wiring + 4 branch facade methods |
| src/tract/__init__.py | New exports (BranchInfo + 4 exceptions) |
| tests/test_storage/test_schema.py | Updated expected schema version and tables |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

1. All 417 tests pass (0 regressions)
2. All 59 new branch tests pass
3. Schema migration verified: fresh DB has commit_parents table, schema_version=2
4. Smoke test passed: branch/switch/list workflow works end-to-end

## Commits

| Hash | Message |
|------|---------|
| 7a9f4c7 | feat(03-01): multi-parent commit schema, repositories, hashing, and compiler |
| 5f29c2a | feat(03-01): branch operations, DAG utilities, and Tract facade methods |
