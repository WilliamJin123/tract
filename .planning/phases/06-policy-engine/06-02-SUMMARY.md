---
phase: "06"
plan: "02"
subsystem: policy-engine
tags: [policy, evaluator, autonomy, proposals, audit-log]
dependency_graph:
  requires: ["06-01"]
  provides: ["Policy ABC", "PolicyEvaluator", "Tract policy integration", "compile/commit hooks"]
  affects: ["06-03"]
tech_stack:
  added: []
  patterns: ["ABC policy interface", "sidecar evaluator pattern", "recursion guard", "proposal lifecycle", "audit logging"]
key_files:
  created:
    - src/tract/policy/__init__.py
    - src/tract/policy/protocols.py
    - src/tract/policy/evaluator.py
    - tests/test_policy_evaluator.py
  modified:
    - src/tract/tract.py
    - src/tract/__init__.py
decisions:
  - id: "06-02-01"
    decision: "Policy ABC uses abstract evaluate(), name, priority, trigger"
    reason: "Simple interface: evaluate returns PolicyAction or None; name/priority/trigger are properties"
  - id: "06-02-02"
    decision: "PolicyEvaluator is a sidecar class, not embedded in Tract"
    reason: "Clean separation of concerns; Tract delegates to evaluator via _policy_evaluator attribute"
  - id: "06-02-03"
    decision: "Recursion guard via _evaluating flag (same pattern as _in_batch)"
    reason: "Prevents infinite loops when policies trigger compile() or commit()"
  - id: "06-02-04"
    decision: "Compile-triggered policies run BEFORE compilation; commit-triggered run AFTER commit"
    reason: "Compile policies can modify state before compilation; commit policies react to committed data"
  - id: "06-02-05"
    decision: "Cooldown is per-policy-name with configurable seconds; pending proposal dedup is implicit"
    reason: "Prevents rapid re-firing; pending proposals naturally deduplicate via DB query"
  - id: "06-02-06"
    decision: "Policy config persisted to _trace_meta as JSON under key 'policy_config'"
    reason: "Reuses existing metadata infrastructure; survives restarts"
metrics:
  duration: "6m"
  completed: "2026-02-18"
  tests_added: 40
  tests_total: 754
---

# Phase 6 Plan 02: Policy Evaluator and Tract Integration Summary

Policy ABC with priority-sorted evaluation, full autonomy spectrum (autonomous/collaborative/manual), recursion guard, cooldown, audit logging, and Tract facade integration with compile/commit hooks.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Policy ABC and PolicyEvaluator | 47300b5 | policy/protocols.py, policy/evaluator.py, policy/__init__.py |
| 2 | Tract facade integration, config persistence, and tests | c1291e4 | tract.py, __init__.py, test_policy_evaluator.py |

## What Was Built

### Policy ABC (protocols.py)
- Abstract base class with `evaluate(tract) -> PolicyAction | None`
- Abstract `name` property for unique identification
- Default `priority = 100` (lower runs first)
- Default `trigger = "compile"` (or "commit")

### PolicyEvaluator (evaluator.py)
- Sidecar class accepting `Tract`, `list[Policy]`, optional `SqlitePolicyRepository`
- Priority-sorted evaluation: policies sorted by priority on init and after register()
- Trigger filtering: only runs policies matching the given trigger
- Recursion guard: `_evaluating` flag prevents re-entrant evaluate() calls
- Cooldown tracking: `_last_fired` dict with configurable `cooldown_seconds`
- Pending proposal deduplication: skips policies with existing pending proposals in DB
- Autonomy dispatch:
  - **autonomous**: `_dispatch_action()` calls Tract methods (compress, annotate, branch, archive)
  - **collaborative**: `_create_proposal()` creates PolicyProposal with reconstructable `_execute_fn`
  - **manual/supervised**: logs and skips
- Proposal management: approve_proposal(), reject_proposal(), get_pending_proposals()
- Reconstructed proposals: `_reconstruct_proposal_fn()` rebuilds closures from DB rows
- Audit logging: `_log_evaluation()` creates PolicyLogRow entries for every triggered evaluation

### Tract Integration (tract.py)
- `_policy_evaluator` and `_policy_repo` attributes on Tract
- `Tract.open()` creates `SqlitePolicyRepository` automatically
- `configure_policies()` creates PolicyEvaluator with policies, callback, cooldown
- `register_policy()` auto-creates evaluator if needed
- `unregister_policy()`, `pause_all_policies()`, `resume_all_policies()`
- `get_pending_proposals()`, `approve_proposal()`, `reject_proposal()`
- `save_policy_config()` / `load_policy_config()` via _trace_meta
- `compile()` triggers compile-triggered policies before compilation
- `commit()` triggers commit-triggered policies after commit
- Both hooks skip during `batch()` (same guard as cache updates)

### Public Exports (__init__.py)
- Policy, PolicyEvaluator, PolicyAction, PolicyProposal, EvaluationResult, PolicyLogEntry
- PolicyExecutionError, PolicyConfigError

## Decisions Made

1. **Policy ABC interface**: evaluate() returns PolicyAction | None; name/priority/trigger are properties with sensible defaults
2. **Sidecar pattern**: PolicyEvaluator is separate from Tract, connected via _policy_evaluator attribute
3. **Recursion guard**: _evaluating flag (same pattern as Tract._in_batch) prevents infinite loops
4. **Hook placement**: compile-triggered policies run BEFORE compilation; commit-triggered run AFTER commit
5. **Cooldown + dedup**: Per-policy cooldown via _last_fired dict; pending proposals provide natural dedup via DB query
6. **Config persistence**: _trace_meta key="policy_config" stores JSON; reuses existing infrastructure

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- `from tract import Policy, PolicyEvaluator, PolicyAction, PolicyProposal, EvaluationResult` -- OK
- `from tract import PolicyExecutionError, PolicyConfigError` -- OK
- `python -m pytest tests/test_policy_evaluator.py -v` -- 40/40 passed
- `python -m pytest tests/ -x` -- 754/754 passed, zero regressions

## Test Coverage

40 new tests across 13 test classes:
- TestPolicyABC (5): subclassing, defaults, evaluate returns
- TestPolicyEvaluator (6): priority sort, trigger filter, recursion guard, pause/resume, register/unregister
- TestAutonomousMode (2): annotate action, compress action
- TestCollaborativeMode (3): proposal creation, approve, reject
- TestAuditLog (1): log entries created
- TestCooldown (1): rapid re-evaluation skipped
- TestPendingDedup (1): pending proposal deduplication
- TestErrorHandling (1): exception caught and logged
- TestTractIntegration (9): configure, register, pause/resume, compile/commit triggers, batch skip
- TestConfigPersistence (3): roundtrip, none when unset, update
- TestDispatch (2): branch action, unknown action error
- TestTractProposals (5): no evaluator, pending only, full lifecycle
- TestManualMode (1): manual mode skips execution

## Success Criteria Verification

1. Policy ABC exists with evaluate(), name, priority, trigger -- VERIFIED
2. PolicyEvaluator iterates by priority, dispatches, guards recursion -- VERIFIED
3. Tract.configure_policies() creates evaluator; register_policy() adds at runtime -- VERIFIED
4. Tract.compile() and commit() trigger at correct points -- VERIFIED
5. Collaborative mode creates proposal with approve/reject lifecycle -- VERIFIED
6. Autonomous mode executes immediately -- VERIFIED
7. Audit log entries created for every triggered evaluation -- VERIFIED
8. Policy config persists via save/load_policy_config() -- VERIFIED
9. pause/resume_all_policies() works as kill switch -- VERIFIED
10. All evaluator tests pass, zero regressions -- VERIFIED (754/754)

## Next Phase Readiness

Plan 06-03 (Built-in Policies) can proceed. It will:
- Create concrete Policy implementations (CompressPolicy, PinPolicy, etc.)
- Use the PolicyEvaluator and Tract hooks established here
- Build on the proposal and audit infrastructure
