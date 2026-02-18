---
phase: 07-agent-toolkit-orchestrator
plan: 03
subsystem: orchestrator
tags: [orchestrator, agent-loop, autonomy, tool-calling, triggers, assessment]
depends_on:
  requires: [07-01, 07-02]
  provides: [orchestrator-loop, context-assessment, tract-facade, triggers, autonomy-spectrum]
  affects: []
tech_stack:
  added: []
  patterns: [tool-calling-loop, autonomy-ceiling, proposal-callback, recursion-guard, trigger-config]
key_files:
  created:
    - src/tract/orchestrator/loop.py
    - src/tract/orchestrator/assessment.py
    - tests/test_orchestrator.py
  modified:
    - src/tract/orchestrator/__init__.py
    - src/tract/tract.py
    - src/tract/__init__.py
decisions:
  - id: "07-03-01"
    summary: "Annotation counts via batch_get_latest for efficient assessment"
  - id: "07-03-02"
    summary: "Compile trigger fires before compilation (on entry to compile())"
  - id: "07-03-03"
    summary: "Trigger errors wrapped in try/except to never break commit/compile"
metrics:
  duration: "8m"
  completed: "2026-02-18"
  tests_added: 19
  tests_total: 888
---

# Phase 7 Plan 3: Orchestrator Loop & Integration Summary

**One-liner:** Tool-calling orchestrator loop with autonomy spectrum (manual/collaborative/autonomous), context assessment, stop/pause control, trigger-based auto-invocation, and full Tract facade integration.

## What Was Built

### Orchestrator Loop (loop.py, ~350 lines)
- `Orchestrator` class with `run()`, `stop()`, `pause()`, `reset()` methods
- Agent loop: build assessment -> call LLM with tools -> extract tool calls -> execute -> repeat
- `_call_llm()` dispatches to configured llm_callable or tract's built-in OpenAI client
- `_extract_tool_calls()` parses OpenAI-format tool call responses
- `_format_tool_results()` formats results back into conversation (assistant msg + tool msgs)
- `_effective_autonomy()` computes min(ceiling, policy_autonomy) for autonomy constraint
- `_execute_tool_call()` routes through autonomy check: MANUAL=skip, COLLABORATIVE=propose, AUTONOMOUS=execute
- `_handle_collaborative()` creates OrchestratorProposal and invokes on_proposal callback
- try/finally ensures `_orchestrating` flag always cleared on exit

### Context Assessment (assessment.py, ~90 lines)
- `build_context_assessment()` gathers context health from tract.status(), tract.log(), tract.list_branches()
- Uses batch_get_latest on annotation repo for efficient pinned/skip counts
- Formats into assessment prompt via prompts/orchestrator.py build_assessment_prompt()
- Does NOT call tract.compile() directly (avoids expense and policy re-triggering)

### Tract Facade Methods (tract.py additions)
- `_orchestrating: bool` instance variable -- recursion guard
- `_set_orchestrating(flag)` -- encapsulated setter for Orchestrator to use
- `configure_orchestrator(config, llm_callable)` -- stores Orchestrator instance
- `orchestrate(config, llm_callable)` -- convenience method, creates or reuses Orchestrator
- `stop_orchestrator()` / `pause_orchestrator()` -- control methods
- `_check_orchestrator_triggers(trigger)` -- fires orchestrator from commit/compile
- Updated compile() and commit() guards: `and not self._orchestrating` on policy evaluation
- TriggerConfig support: on_commit_count (counter-based), on_token_threshold (budget %), on_compile (boolean)

### Exports (__init__.py)
- Added to tract package exports: Orchestrator, OrchestratorConfig, AutonomyLevel, OrchestratorState, TriggerConfig, OrchestratorProposal, ProposalResponse, StepResult, OrchestratorResult, auto_approve, log_and_approve, cli_prompt, reject_all, OrchestratorError

## Decisions Made

1. **Annotation counts via batch_get_latest** -- Used the existing batch_get_latest() method on SqliteAnnotationRepository rather than per-commit queries for the assessment builder.

2. **Compile trigger fires before compilation** -- The on_compile trigger check fires after policy evaluation but before the actual compile logic, to ensure it fires exactly once per compile() call.

3. **Trigger errors never break commit/compile** -- All trigger checks wrapped in try/except to ensure orchestrator trigger failures don't corrupt the commit/compile path.

## Deviations from Plan

None -- plan executed exactly as written.

## Test Summary

19 new integration tests added:
1. No action needed (LLM returns no tools)
2. Autonomous execute (tool calls execute directly)
3. Collaborative approve (auto_approve callback)
4. Collaborative reject (reject_all callback)
5. Manual skip (all tools skipped)
6. Max steps limit (loop stops at limit)
7. Stop (via on_step callback)
8. Pause (via on_step callback)
9. No LLM error (OrchestratorError raised)
10. Recursion guard (flag set during run, cleared after)
11. Error recovery (bad tool name handled gracefully)
12. Tract.orchestrate() convenience method
13. Tract.configure_orchestrator() then orchestrate()
14. Policy guard during orchestration
15. Multiple tool calls in one turn
16. as_tools integration (tools list + ToolExecutor)
17. Trigger on commit count
18. Trigger on token threshold
19. Trigger on compile

Full regression suite: 888 tests passing (869 existing + 19 new, zero regressions).

## Next Phase Readiness

Phase 7 is complete. All three plans (07-01, 07-02, 07-03) are done:
- 07-01: Agent Toolkit (definitions, profiles, executor, as_tools facade)
- 07-02: Orchestrator Data Models (config, models, callbacks, prompts)
- 07-03: Orchestrator Loop & Integration (this plan)

The v2 milestone is complete: autonomy spectrum from manual to autonomous context management is fully operational.
