# Phase 7 Plan 02: Orchestrator Data Models Summary

**One-liner:** Orchestrator config/models/callbacks/prompts -- AutonomyLevel enum, OrchestratorConfig with autonomy ceiling/triggers/callbacks, ToolCall canonical type, proposal review flow, 4 built-in callbacks, and assessment prompt builder.

## Results

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Orchestrator Config, Models, and Callbacks | 0852727 | orchestrator/__init__.py, config.py, models.py, callbacks.py, exceptions.py |
| 2 | Assessment Prompts and Tests | a9d5615 | prompts/orchestrator.py, tests/test_orchestrator_models.py |

**Duration:** ~6m
**Tests:** 34 new, 832 total passing (798 + 34 new)

## What Was Built

### Configuration Types (config.py)
- **AutonomyLevel(str, Enum)**: MANUAL, COLLABORATIVE, AUTONOMOUS -- controls orchestrator independence
- **OrchestratorState(str, Enum)**: IDLE, RUNNING, PAUSING, STOPPED -- lifecycle states
- **TriggerConfig(frozen=True)**: on_commit_count, on_token_threshold, on_compile -- activation triggers
- **OrchestratorConfig**: autonomy_ceiling, max_steps, profile, system_prompt, task_context, triggers, model, temperature, on_proposal callback, on_step callback. Mutable dataclass (matches TractConfig pattern).

### Model Types (models.py)
- **ToolCall(frozen=True)**: id, name, arguments. CANONICAL location -- toolkit will re-export from here.
- **ProposalDecision(str, Enum)**: APPROVED, REJECTED, MODIFIED
- **OrchestratorProposal**: Mutable proposal with recommended_action, reasoning, alternatives, context_summary, decision field
- **ProposalResponse(frozen=True)**: Callback return type with decision, optional modified_action, reason
- **StepResult(frozen=True)**: step number, tool_call, result_output/error, success flag, optional proposal
- **OrchestratorResult(frozen=True)**: steps list, state, assessment, total_tool_calls, `succeeded` property

### Built-in Callbacks (callbacks.py)
- **auto_approve**: Returns APPROVED unconditionally. For autonomous mode.
- **log_and_approve**: Logs proposal details via logger.info, then returns APPROVED. For audit trail.
- **cli_prompt**: Interactive Rich Panel prompt with approve/reject/modify. Falls back to plain input() if Rich unavailable. JSON-validated modified arguments.
- **reject_all**: Returns REJECTED with "Auto-rejected" reason. For testing/safety.

### Assessment Prompts (prompts/orchestrator.py)
- **ORCHESTRATOR_SYSTEM_PROMPT**: Decision framework (compress > pin > branch), behavioral rules (no compress pinned, small targeted actions), relevance/coherence guidance for LLM.
- **build_assessment_prompt()**: Formats token usage, commit count, branch, annotations (pinned/skipped), recent commits (max 10), optional task_context. Returns structured prompt for context assessment.

### Exception (exceptions.py)
- **OrchestratorError(TraceError)**: Base exception for orchestrator failures

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| OrchestratorResult.state default via __post_init__ | Avoids circular import between models.py and config.py at module level; lazy imports OrchestratorState |
| ToolCall as canonical in orchestrator.models | Single source of truth; toolkit will re-export, avoiding duplication |
| cli_prompt lazy-imports Rich | Matches existing [cli] optional extra pattern; graceful fallback to plain input() |
| TriggerConfig frozen, OrchestratorConfig mutable | Triggers are set once; config may be adjusted between runs (like TractConfig) |

## Success Criteria Verification

1. OrchestratorConfig supports all configuration options -- VERIFIED (autonomy ceiling, max steps, profile, triggers, callbacks)
2. OrchestratorProposal contains recommended action, reasoning, alternatives, and mutable decision -- VERIFIED
3. Four built-in callbacks return correct ProposalResponse -- VERIFIED (auto_approve, log_and_approve, reject_all tested; cli_prompt tested manually)
4. Assessment prompt builder formats context state with structural health indicators -- VERIFIED (pinned_count, skip_count, branch_count)
5. All tests pass with zero regressions -- VERIFIED (832 passed)

## Key Files

### Created
- `src/tract/orchestrator/__init__.py` -- Public exports for orchestrator module
- `src/tract/orchestrator/config.py` -- AutonomyLevel, OrchestratorState, TriggerConfig, OrchestratorConfig
- `src/tract/orchestrator/models.py` -- ToolCall, ProposalDecision, OrchestratorProposal, ProposalResponse, StepResult, OrchestratorResult
- `src/tract/orchestrator/callbacks.py` -- auto_approve, log_and_approve, cli_prompt, reject_all
- `src/tract/prompts/orchestrator.py` -- ORCHESTRATOR_SYSTEM_PROMPT, build_assessment_prompt()
- `tests/test_orchestrator_models.py` -- 34 tests for all orchestrator types

### Modified
- `src/tract/exceptions.py` -- Added OrchestratorError

## Next Phase Readiness

Plan 03 (Orchestrator Loop) can now import all types from `tract.orchestrator`:
- OrchestratorConfig for configuration
- OrchestratorProposal/ProposalResponse for the proposal flow
- StepResult/OrchestratorResult for tracking execution
- Built-in callbacks for default behaviors
- ToolCall for tool invocations
- Assessment prompts for LLM context evaluation
