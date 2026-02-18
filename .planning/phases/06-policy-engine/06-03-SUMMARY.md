---
phase: "06"
plan: "03"
subsystem: policy-engine
tags: [policy, builtin, compress, pin, branch, rebase, integration, auto-load]
dependency_graph:
  requires: ["06-01", "06-02"]
  provides: ["CompressPolicy", "PinPolicy", "BranchPolicy", "RebasePolicy", "auto-load on restart", "end-to-end policy system"]
  affects: ["07"]
tech_stack:
  added: []
  patterns: ["threshold-based auto-compress", "content-type auto-pin", "tangent detection heuristic", "stale branch archiving", "auto-load from persisted config"]
key_files:
  created:
    - src/tract/policy/builtin/__init__.py
    - src/tract/policy/builtin/compress.py
    - src/tract/policy/builtin/pin.py
    - src/tract/policy/builtin/branch.py
    - src/tract/policy/builtin/rebase.py
    - tests/test_policy_builtin.py
    - tests/test_policy_integration.py
  modified:
    - src/tract/policy/__init__.py
    - src/tract/__init__.py
    - src/tract/tract.py
decisions:
  - id: "06-03-01"
    decision: "CompressPolicy fires at configurable threshold (default 90%) of token budget"
    reason: "Threshold-based triggering is simple and predictable; 90% is conservative enough to avoid premature compression"
  - id: "06-03-02"
    decision: "PinPolicy checks existing annotations before pinning (respects manual overrides)"
    reason: "User sovereignty: manual annotations always take precedence over automatic ones"
  - id: "06-03-03"
    decision: "InstructionContent already gets auto-annotated by commit engine; PinPolicy adds value for SessionContent and custom types"
    reason: "No duplicate pinning; PinPolicy extends the auto-pin behavior to types the engine doesn't handle"
  - id: "06-03-04"
    decision: "BranchPolicy ignores dialogue/tool_io transitions by default"
    reason: "Dialogue-tool switching is normal back-and-forth, not a tangent signal"
  - id: "06-03-05"
    decision: "RebasePolicy uses min_commits AND stale_days (both must be true)"
    reason: "Avoid archiving large branches that happen to be paused; both conditions must hold"
  - id: "06-03-06"
    decision: "Tract.open() auto-loads saved policy config using built-in type map"
    reason: "Enables persistent policy configuration without user having to re-configure on every open()"
metrics:
  duration: "7m"
  completed: "2026-02-18"
  tests_added: 44
  tests_total: 798
---

# Phase 6 Plan 03: Built-in Policies and Integration Tests Summary

Four built-in policies (CompressPolicy, PinPolicy, BranchPolicy, RebasePolicy) with threshold-based auto-compress, content-type auto-pin, tangent detection, stale branch archiving, auto-load from persisted config, and 44 end-to-end tests.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Built-in policies (CompressPolicy, PinPolicy, BranchPolicy, RebasePolicy) | 314ecb7 | builtin/compress.py, builtin/pin.py, builtin/branch.py, builtin/rebase.py, builtin/__init__.py |
| 2 | Unit tests and end-to-end integration tests | 1c08556 | test_policy_builtin.py, test_policy_integration.py |
| 3 | Auto-load policy config on Tract.open() | 7550647 | tract.py, test_policy_integration.py |

## What Was Built

### CompressPolicy (builtin/compress.py)
- `name = "auto-compress"`, `priority = 200`, `trigger = "compile"`
- Evaluates token_count against `max_tokens * threshold` (default 0.9)
- Returns collaborative PolicyAction with `action_type="compress"`
- Optional `summary_content` parameter for testing without LLM
- to_config/from_config serialization

### PinPolicy (builtin/pin.py)
- `name = "auto-pin"`, `priority = 100`, `trigger = "commit"`
- Auto-pins commits with `content_type in pin_types` (default: instruction, session)
- Respects manual overrides: skips commits with ANY existing annotation
- `retroactive_scan()` for one-time backfill when first enabled
- Pattern matching support for advanced content matching
- to_config/from_config serialization

### BranchPolicy (builtin/branch.py)
- `name = "auto-branch"`, `priority = 300`, `trigger = "commit"`
- Detects rapid content type transitions in recent commits
- Configurable: `content_type_window`, `switch_threshold`, `ignore_transitions`
- Default ignores dialogue/tool_io transitions (normal back-and-forth)
- Proposes `tangent/{branch}/{timestamp}` branch in collaborative mode
- to_config/from_config serialization

### RebasePolicy (builtin/rebase.py)
- `name = "auto-rebase"`, `priority = 500`, `trigger = "compile"`
- Detects stale branches: `len(commits) <= min_commits AND age >= stale_days`
- Proposes archiving to `{archive_prefix}{branch_name}`
- Skips main branch and already-archived branches
- to_config/from_config serialization

### Auto-Load on Restart (tract.py)
- `Tract.open()` reads saved policy config from `_trace_meta`
- Maps policy names to built-in classes via type map
- Reconstructs policies via `from_config()` for enabled entries
- Automatically calls `configure_policies()` if any policies found

### Exports
- All four built-in policies exported from `tract.policy.builtin`, `tract.policy`, and `tract` top-level

## Decisions Made

1. **Threshold-based compress**: 90% default threshold is conservative; avoids premature compression
2. **Manual override respect**: PinPolicy checks for ANY existing annotation, not just PINNED
3. **Instruction auto-annotation**: Commit engine already pins instructions; PinPolicy adds value for SessionContent and custom types
4. **Ignored transitions**: dialogue/tool_io back-and-forth is not a tangent signal
5. **Dual-condition archiving**: Both min_commits AND stale_days must hold to avoid archiving large paused branches
6. **Auto-load on open**: Built-in type map enables persistent config without user re-configuration

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PinPolicy test adjusted for pre-existing annotations**
- **Found during:** Task 2
- **Issue:** InstructionContent commits get auto-annotated by the commit engine (default priority for instruction type), so PinPolicy correctly returns None for them
- **Fix:** Updated tests to use SessionContent (which doesn't get auto-annotated) for testing PinPolicy's pin action. Instruction test verifies PinPolicy correctly defers to existing annotations.
- **Files modified:** tests/test_policy_builtin.py

**2. [Rule 1 - Bug] SessionContent constructor fields**
- **Found during:** Task 2
- **Issue:** SessionContent requires `session_type` and `summary` fields (not `session_id`/`event_type`/`agent_id`)
- **Fix:** Updated test to use correct SessionContent constructor
- **Files modified:** tests/test_policy_builtin.py

## Verification Results

- `from tract import CompressPolicy, PinPolicy, BranchPolicy, RebasePolicy` -- OK
- `python -m pytest tests/test_policy_builtin.py -v` -- 32/32 passed
- `python -m pytest tests/test_policy_integration.py -v` -- 12/12 passed
- `python -m pytest tests/ -x` -- 798/798 passed, zero regressions
- Priority ordering: PinPolicy(100) < CompressPolicy(200) < BranchPolicy(300) < RebasePolicy(500) -- verified

## Test Coverage

44 new tests across 14 test classes:

**Unit Tests (test_policy_builtin.py) -- 32 tests:**
- TestCompressPolicy (6): no budget, below/above threshold, custom threshold, properties, config roundtrip
- TestPinPolicy (10): pins instruction (defers), pins session, skips dialogue, respects manual, skips already-pinned, custom types, retroactive scan (2), properties, config roundtrip
- TestBranchPolicy (7): no tangent, detects tangent, too few commits, detached HEAD, custom threshold, properties, config roundtrip
- TestRebasePolicy (7): main skipped, active skipped, stale detected, archive prefix, already-archived skipped, properties, config roundtrip
- TestPriorityOrdering (2): ordering verification, exact values

**Integration Tests (test_policy_integration.py) -- 12 tests:**
- TestAutoPinOnCommit (2): auto-pin instruction, skip dialogue
- TestAutoCompressOnCompile (1): proposal created on compile
- TestPolicyPriorityOrdering (1): audit log order verification
- TestPauseResume (1): pause/resume lifecycle
- TestCollaborativeApproveReject (1): approve/reject cycle
- TestPolicyConfigPersistence (1): save/load roundtrip
- TestRecursiveEvaluationPrevention (1): no infinite recursion
- TestMultiplePoliciesCompose (1): pin + compress together
- TestCustomPolicySubclass (1): user-defined policy works
- TestOnProposalCallback (1): callback invoked on proposal
- TestPolicyConfigSurvivesRestart (1): file-backed restart persistence

## Success Criteria Verification

1. CompressPolicy fires at 90% token budget threshold (configurable), proposes compression in collaborative mode -- VERIFIED
2. PinPolicy auto-pins SessionContent on commit, respects manual overrides, supports retroactive scan -- VERIFIED
3. BranchPolicy detects content type switching patterns and proposes tangent branches -- VERIFIED
4. RebasePolicy detects stale branches and proposes archiving to archive/ prefix -- VERIFIED
5. All policies have correct priority ordering (100, 200, 300, 500) -- VERIFIED
6. All policies support to_config/from_config serialization -- VERIFIED
7. End-to-end integration: configure -> commit -> evaluate -> execute works -- VERIFIED
8. Collaborative approve/reject lifecycle works -- VERIFIED
9. Policy composition works (multiple policies interact correctly via priority ordering) -- VERIFIED
10. All tests pass, zero regressions -- VERIFIED (798/798)
11. Policies auto-load from _trace_meta on Tract.open() -- config survives restart -- VERIFIED

## Phase 6 Complete

All 3 plans of Phase 6 (Policy Engine) are now complete:
- 06-01: Policy Storage Foundation (tables, repository, domain models)
- 06-02: Policy Evaluator and Tract Integration (ABC, evaluator, hooks)
- 06-03: Built-in Policies (CompressPolicy, PinPolicy, BranchPolicy, RebasePolicy)

The full policy engine delivers the AUTO-01 through AUTO-06 requirements with:
- Configurable, priority-sorted policy evaluation
- Full autonomy spectrum (autonomous/collaborative/manual)
- Persistent configuration that survives restarts
- Comprehensive test coverage (113 policy tests total across 3 plans)
