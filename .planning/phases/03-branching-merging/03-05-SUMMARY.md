---
phase: 03-branching-merging
plan: 05
subsystem: cli
tags: [cli, click, rich, branch, switch, merge]
dependency-graph:
  requires: ["03-01", "03-03"]
  provides: ["CLI branch/switch/merge commands", "format_branches", "format_merge_result"]
  affects: []
tech-stack:
  added: []
  patterns: ["Click command group with invoke_without_command for branch listing", "BranchNotFoundError catch with helpful branch listing"]
key-files:
  created:
    - src/tract/cli/commands/branch.py
    - src/tract/cli/commands/switch.py
    - src/tract/cli/commands/merge.py
  modified:
    - src/tract/cli/__init__.py
    - src/tract/cli/formatting.py
    - tests/test_cli.py
decisions: []
metrics:
  duration: 3m
  completed: 2026-02-15
---

# Phase 3 Plan 5: Branch/Switch/Merge CLI Commands Summary

**One-liner:** Click CLI commands wrapping Tract.branch()/switch()/merge() with Rich formatting for branch lists and merge results.

## What Was Built

Three new CLI commands registered on the existing Click group, completing the Phase 3 terminal interface:

1. **`tract branch`** -- Command group with `invoke_without_command=True`:
   - No subcommand: lists all branches with `*` marker on current branch (git-style)
   - `tract branch create NAME [--no-switch] [--source COMMIT]`: create and optionally switch
   - `tract branch delete NAME [--force]`: delete non-current branch

2. **`tract switch TARGET`** -- Branch-only switching (unlike checkout, won't detach HEAD):
   - Catches `BranchNotFoundError` and lists available branches in the error message

3. **`tract merge SOURCE [--no-ff] [--strategy auto|semantic]`** -- Merge with formatted output:
   - Fast-forward: "Fast-forward: branch -> hash"
   - Clean merge: "Merged source into target (merge commit: hash)"
   - Conflicts: "CONFLICT: N conflicts detected" with per-conflict detail
   - Catches `NothingToMergeError`: "Already up to date."

4. **Formatting helpers** added to `cli/formatting.py`:
   - `format_branches()`: Rich-formatted branch list with green bold current branch
   - `format_merge_result()`: Type-specific merge output with colors

## Test Results

- 12 new CLI tests added (30 total CLI tests)
- 489 total tests passing (477 + 12 new)
- All tests via CliRunner with isolated filesystems and file-backed databases

### New Tests

| Test | Description |
|------|-------------|
| test_help_includes_new_commands | Help text shows branch, switch, merge |
| test_subcommand_help_new | --help works for each new command |
| test_branch_list | Lists branches with * marker |
| test_branch_create | Creates and switches to branch |
| test_branch_create_no_switch | Creates without switching |
| test_branch_delete | Deletes branch |
| test_branch_delete_current_errors | Cannot delete current branch |
| test_switch_to_branch | Switches to existing branch |
| test_switch_nonexistent | Error with helpful message |
| test_merge_fast_forward | Fast-forward merge output |
| test_merge_clean | Clean merge output |
| test_merge_already_up_to_date | "Already up to date" output |

## Decisions Made

None -- straightforward CLI wrappers following established Phase 2 patterns.

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Hash | Message |
|------|---------|
| d72e1da | feat(03-05): add branch, switch, and merge CLI commands |

## Phase 3 Completion

This is the final plan (5/5) for Phase 3: Branching & Merging. All Phase 3 deliverables are complete:
- 03-01: Branch infrastructure (59 tests)
- 03-02: LLM client infrastructure (56 tests)
- 03-03: Merge strategies (34 tests)
- 03-04: Rebase & cherry-pick (26 tests)
- 03-05: CLI commands (12 tests)

Total Phase 3: 187 new tests, 489 total suite passing.
