---
phase: 03
plan: 03
subsystem: merge-operations
tags: [merge, conflict-detection, llm-resolver, branch-blocks, fast-forward]
depends_on:
  requires:
    - "03-01 (branch infrastructure, DAG utilities, commit_parents table)"
    - "03-02 (LLM client, OpenAIResolver, ResolverCallable protocol)"
  provides:
    - "Tract.merge() with fast-forward, clean merge, and conflict detection"
    - "Tract.commit_merge() review/commit flow for conflict resolution"
    - "Tract.configure_llm() for LLM client integration"
    - "MergeResult, ConflictInfo, MergeError models"
    - "detect_conflicts() for 3 structural conflict types"
    - "create_merge_commit() with multi-parent commit support"
  affects:
    - "03-04 (rebase/cherry-pick uses same resolver pattern)"
    - "03-05 (CLI merge command wraps Tract.merge())"
    - "Phase 4 (compression may integrate with merge review flow)"
tech_stack:
  added: []
  patterns:
    - "Merge review flow: merge() returns MergeResult, user reviews, commit_merge() finalizes"
    - "Structural conflict detection from commit graph (no content diff needed)"
    - "Pre-loaded content text in ConflictInfo for resolver efficiency"
    - "create_merge_commit via CommitEngine.create_merge_commit with parent_repo"
key_files:
  created:
    - "src/tract/models/merge.py"
    - "src/tract/operations/merge.py"
    - "tests/test_merge.py"
  modified:
    - "src/tract/exceptions.py"
    - "src/tract/engine/commit.py"
    - "src/tract/tract.py"
    - "src/tract/__init__.py"
decisions:
  - id: "03-03-01"
    decision: "Merge commit created via CommitEngine.create_merge_commit() with parent_repo parameter"
    rationale: "Keeps commit creation logic centralized in the engine; parent_repo wired in via __init__"
  - id: "03-03-02"
    decision: "Pre-load content text into ConflictInfo at detect_conflicts() time"
    rationale: "Resolvers receive actual content text, not just CommitInfo metadata, without needing repo access"
  - id: "03-03-03"
    decision: "EDIT + APPEND conflict only for pre-merge-base targets"
    rationale: "Edits to a branch's own divergent history (post-merge-base) don't affect the other branch"
  - id: "03-03-04"
    decision: "MergeResult._source_tip_hash and _target_tip_hash for commit_merge parent resolution"
    rationale: "Avoid re-resolving branch refs between merge() and commit_merge() calls"
  - id: "03-03-05"
    decision: "configure_llm() creates default OpenAIResolver; merge() uses it as fallback"
    rationale: "Config-pattern: set once on Tract, override per-operation"
metrics:
  duration: "~8m"
  completed: "2026-02-15"
  tests_added: 34
  tests_total: 451
  lines_added: ~1721
---

# Phase 3 Plan 3: Merge Strategies Summary

**One-liner:** Full merge pipeline with fast-forward, clean auto-merge, structural conflict detection (3 types), LLM-mediated resolution, and MergeResult review/commit flow.

## What Was Built

### Task 1: Merge models and conflict detection

**Exceptions** (`exceptions.py`):
- `MergeError(TraceError)`: Base merge exception
- `MergeConflictError(MergeError)`: Conflicts detected, no resolver
- `NothingToMergeError(MergeError)`: Source already up-to-date

**Models** (`models/merge.py`):
- `ConflictInfo`: Rich context for a single conflict (type, both commits, pre-loaded content text, ancestor, target hash, branch histories)
- `MergeResult`: Merge outcome with review/commit flow (merge_type, conflicts, resolutions, edit_resolution(), committed flag)

**Operations** (`operations/merge.py`):
- `detect_conflicts()`: Identifies 3 structural conflict types:
  - `both_edit`: Both branches EDIT the same target commit
  - `skip_vs_edit`: One branch SKIPs, other EDITs same commit
  - `edit_plus_append`: One branch EDITs pre-merge-base commit while other APPENDs
- `merge_branches()`: Full merge pipeline:
  - Fast-forward detection (pointer move, no merge commit)
  - Clean merge for APPEND-only divergence (branch-blocks ordering)
  - Conflict detection with optional resolver invocation
  - Resolver abort handling
- `create_merge_commit()`: Creates commit with multi-parent recording

### Task 2: Tract facade methods and comprehensive tests

**CommitEngine** (`engine/commit.py`):
- `create_merge_commit()` method: similar to create_commit but with multi-parent support
- `parent_repo` parameter added to `__init__`

**Tract facade** (`tract.py`):
- `configure_llm(client)`: Stores LLM client and creates default OpenAIResolver
- `merge(source_branch, *, resolver, strategy, no_ff, auto_commit, model, delete_branch)`: Full merge interface
- `commit_merge(result)`: Finalize conflict merge after review/editing resolutions
- parent_repo wired into CommitEngine in Tract.open()

**Exports** (`__init__.py`):
- Added: MergeResult, ConflictInfo, MergeError, MergeConflictError, NothingToMergeError

**Tests** (`tests/test_merge.py`, 34 tests):
- Fast-forward (5): pointer move, no merge commit, no_ff, already up-to-date, same commit
- Clean merge (5): creates merge commit, two parents, compiled after merge, branch-blocks ordering, multiple commits
- Conflict detection (5): both_edit, skip_vs_edit, edit_plus_append, no conflict for post-merge-base edit, content preloaded
- Conflict resolution (8): without resolver, with resolver, commit_merge after review, unresolved raises, auto_commit, delete_branch, delete branch blocked on unresolved, resolver abort
- Integration (8): full workflow, generation_config preserved, cache cleared, nonexistent branch, detached head, edit_resolution, sequential merges, configure_llm
- Models (3): ConflictInfo creation, MergeResult creation, edit_resolution

## Decisions Made

1. **CommitEngine.create_merge_commit()** (03-03-01): Merge commit creation centralized in engine with parent_repo parameter, keeping the write path unified.

2. **Pre-loaded content text** (03-03-02): ConflictInfo objects have content text pre-loaded from blobs at detection time, so resolvers don't need repository access.

3. **Pre-merge-base EDIT only triggers conflict** (03-03-03): An EDIT targeting a post-merge-base commit (branch's own history) is NOT a conflict with the other branch's appends.

4. **Internal parent hashes on MergeResult** (03-03-04): _source_tip_hash and _target_tip_hash stored to avoid re-resolving refs between merge() and commit_merge().

5. **configure_llm() + default resolver** (03-03-05): Setting the LLM client on Tract creates a default OpenAIResolver; merge() falls back to it when no explicit resolver given.

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

1. All 451 tests pass (417 existing + 34 new, zero regressions)
2. All 34 merge tests pass
3. Smoke test passed:
   - branch, diverge, merge -> clean merge with 4 messages
4. All 7 success criteria verified:
   - Fast-forward merge works (pointer move, no merge commit)
   - Clean merge creates merge commit with two parents and branch-blocks ordering
   - Structural conflict detection identifies all three conflict types
   - MergeResult review flow works (get result, edit, commit_merge)
   - LLM resolver integration works for conflict resolution
   - Compiled context after merge includes all branch commits
   - All existing + new tests pass

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `f96dcef` | Merge models, conflict detection, and merge_branches |
| 2 | `cce9bdd` | Tract facade merge methods and 34 comprehensive tests |

## Next Phase Readiness

**For Plan 03-04 (Rebase & Cherry-Pick):**
- Merge infrastructure provides the template for rebase/cherry-pick conflict handling
- Same resolver pattern (ResolverCallable) applies to rebase warnings and cherry-pick issues
- create_merge_commit() available for cherry-pick commit creation

**For Plan 03-05 (CLI):**
- Tract.merge() provides the SDK interface for the CLI merge command
- MergeResult has all the display data for CLI output (conflicts, resolutions, merge_type)
