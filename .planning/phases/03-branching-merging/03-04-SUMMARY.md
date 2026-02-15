# Phase 3 Plan 4: Rebase & Cherry-Pick Summary

**One-liner:** Cherry-pick and rebase operations with commit replay, EDIT target detection, and block-until-resolved semantic safety checks.

## Execution Stats

| Metric | Value |
|--------|-------|
| Duration | ~6m |
| Tasks | 2/2 |
| Tests added | 26 |
| Tests total | 477 |
| Files created | 2 (operations/rebase.py, tests/test_rebase.py) |
| Files modified | 4 (exceptions.py, models/merge.py, tract.py, __init__.py) |

## What Was Built

### Cherry-Pick Operation
- `Tract.cherry_pick(hash)` copies a commit to the current branch with new hash and parentage
- Preserves content, message, metadata, and generation_config from original
- Detects when EDIT commits target a commit missing from the current branch
- Issues block execution: CherryPickError with no resolver, or resolver decides (resolve/abort/skip)

### Rebase Operation
- `Tract.rebase("main")` replays current branch commits onto target branch tip
- Produces new commits with new hashes (deterministic: same content + new parent + new timestamp)
- Updates branch pointer to last replayed commit, re-attaches HEAD
- Blocks on branches containing merge commits (cannot flatten multi-parent history)

### Semantic Safety Checks
- EDIT target missing: detected when rebase would move an EDIT commit away from its target
- All safety warnings block until resolver provides resolution (no warn-and-continue)
- SemanticSafetyError raised when no resolver available
- RebaseError raised when resolver aborts

### Models Added (models/merge.py)
- `RebaseWarning`: Semantic safety issue with warning_type, commit context, description
- `CherryPickIssue`: Issue with issue_type (edit_target_missing, context_dependency)
- `RebaseResult`: replayed_commits, original_commits, warnings, new_head
- `CherryPickResult`: original_commit, new_commit, issues, resolutions

### Exceptions Added (exceptions.py)
- `RebaseError(TraceError)`: Base for rebase errors
- `CherryPickError(TraceError)`: Base for cherry-pick errors
- `SemanticSafetyError(TraceError)`: Semantic safety block without resolver

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 2dfc85c | feat | Cherry-pick and rebase operations with safety checks |
| b6f296b | feat | Tract facade methods and comprehensive tests |

## Decisions Made

1. **Cherry-pick resolved content as APPEND**: When resolver provides content_text for an EDIT with missing target, the cherry-picked commit is created as APPEND (since there is no valid response_to on the target branch).
2. **Rebase blocks on merge commits**: Pre-flight check prevents rebase of branches containing merge commits. Merge commits have multiple parents that cannot be meaningfully replayed as single-parent commits.
3. **Noop rebase when already ahead**: If current branch tip's merge base with target equals the target tip (current is already ahead), rebase returns empty result rather than erroring.
4. **Replay via CommitEngine.create_commit()**: Replayed commits go through the standard commit engine, which reads HEAD internally. The rebase operation moves HEAD before each replay to ensure correct parentage.

## Deviations from Plan

None -- plan executed exactly as written.

## Test Coverage

### Cherry-pick (8 tests)
- APPEND commit: basic, content preservation, metadata preservation, by hash
- EDIT commit: target on branch (works), missing target (error), with resolver (resolved), with skip resolver

### Rebase (5 tests)
- Simple rebase, content preservation, branch pointer update, noop, detached HEAD error

### Safety checks (4 tests)
- Blocks without resolver, passes with resolver, abort on safety, edit missing target

### Integration (5 tests)
- Cherry-pick then compile, rebase then merge (fast-forward), full workflow, append-only no warnings, EDIT target on shared history

### Edge cases (4 tests)
- Result type verification (cherry-pick and rebase), detached HEAD error, nonexistent commit error

## Key Files

| File | Purpose |
|------|---------|
| `src/tract/operations/rebase.py` | Core replay_commit(), cherry_pick(), rebase() operations |
| `src/tract/models/merge.py` | RebaseWarning, CherryPickIssue, RebaseResult, CherryPickResult |
| `src/tract/exceptions.py` | RebaseError, CherryPickError, SemanticSafetyError |
| `src/tract/tract.py` | Tract.cherry_pick(), Tract.rebase() facade methods |
| `src/tract/__init__.py` | Public exports for all new types |
| `tests/test_rebase.py` | 26 tests, 650 lines |

## Next Phase Readiness

Plan 03-05 (CLI commands for branching) can proceed. All rebase and cherry-pick operations are available through the Tract facade and can be wired to CLI commands.
