---
phase: 07-agent-toolkit-orchestrator
plan: 01
subsystem: toolkit
tags: [tools, profiles, executor, llm-integration, openai, anthropic]
depends_on:
  requires: [phase-6]
  provides: [toolkit-models, tool-definitions, profiles, executor, as-tools-facade]
  affects: [07-03]
tech_stack:
  added: []
  patterns: [frozen-dataclasses, handler-lambdas, explicit-parameter-whitelisting, lazy-imports]
key_files:
  created:
    - src/tract/toolkit/__init__.py
    - src/tract/toolkit/models.py
    - src/tract/toolkit/definitions.py
    - src/tract/toolkit/profiles.py
    - src/tract/toolkit/executor.py
    - tests/test_toolkit.py
  modified:
    - src/tract/tract.py
    - src/tract/__init__.py
decisions:
  - "Handler lambdas use explicit parameter whitelisting (not **kwargs) to prevent hallucinated arguments"
  - "ToolCall canonical location is orchestrator.models; toolkit re-exports with try/except ImportError"
  - "Tract.as_tools() uses lazy imports to avoid circular dependencies"
  - "Handlers convert complex return values to human-readable strings (not raw repr)"
  - "GC tool converts min_age_hours to days for the gc() API"
metrics:
  duration: 9m
  completed: 2026-02-18
---

# Phase 7 Plan 01: Agent Toolkit Summary

Toolkit layer exposing Tract operations as LLM-consumable tool schemas with profiles, executor, and Tract.as_tools() facade.

## One-Liner

15 hand-crafted tool definitions with 3 built-in profiles, ToolExecutor dispatch, and Tract.as_tools() facade producing OpenAI/Anthropic format dicts.

## What Was Built

### Task 1: Toolkit Models, Tool Definitions, and Profiles
- **models.py**: Frozen dataclasses -- ToolDefinition (with to_openai/to_anthropic), ToolConfig, ToolProfile (with filter_tools), ToolResult
- **definitions.py**: `get_all_tools(tract)` returns 15 ToolDefinitions bound to a Tract instance via handler lambdas with explicit parameter whitelisting
- **profiles.py**: Three built-in profiles:
  - SELF_PROFILE (9 tools): commit, compile, annotate, status, log, compress, branch, switch, reset -- with self-referential descriptions ("your context")
  - SUPERVISOR_PROFILE (15 tools): all tools with managerial descriptions ("the managed agent's context")
  - FULL_PROFILE (15 tools): all tools with default descriptions
- **__init__.py**: Re-exports all public types; ToolCall re-export from orchestrator.models with try/except ImportError; lazy ToolExecutor import via __getattr__

### Task 2: ToolExecutor, Tract.as_tools() Facade, and Tests
- **executor.py**: ToolExecutor class with execute(tool_name, arguments) -> ToolResult dispatch, available_tools() listing
- **tract.py**: Added Tract.as_tools(profile, overrides, format) method that combines get_all_tools + profile filtering + description overrides + format conversion
- **__init__.py**: Added ToolDefinition, ToolProfile, ToolConfig, ToolResult, ToolExecutor to tract package exports
- **tests/test_toolkit.py**: 37 comprehensive tests covering formats, profiles, executor, as_tools, and end-to-end integration

## Tool Definitions

| # | Tool | Description | Required Args |
|---|------|-------------|---------------|
| 1 | commit | Record new context | content |
| 2 | compile | Compile context to messages | (none) |
| 3 | annotate | Set priority on commit | target_hash, priority |
| 4 | status | Get current tract status | (none) |
| 5 | log | View commit history | (none) |
| 6 | diff | Compare two commits | (none) |
| 7 | compress | Compress commit range | (none) |
| 8 | branch | Create new branch | name |
| 9 | switch | Switch branch | target |
| 10 | merge | Merge branch | source |
| 11 | reset | Reset HEAD | target |
| 12 | checkout | Read-only inspection | target |
| 13 | gc | Garbage collection | (none) |
| 14 | list_branches | List all branches | (none) |
| 15 | get_commit | Get commit details | commit_hash |

## Decisions Made

1. **Explicit parameter whitelisting**: Handler lambdas name only the parameters defined in their JSON Schema, preventing LLM-hallucinated arguments from flowing through to Tract methods.
2. **ToolCall re-export**: ToolCall is canonically defined in orchestrator.models (Plan 02); toolkit re-exports with try/except ImportError for parallel plan execution.
3. **Lazy imports**: Tract.as_tools() and toolkit __init__.py use lazy imports to avoid circular dependencies.
4. **Human-readable handler output**: Handlers convert complex Tract return types (StatusInfo, CompressResult, etc.) to concise human-readable strings.
5. **GC parameter mapping**: Tool exposes min_age_hours for simpler LLM interaction; internally converts to days for gc() API.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed CompressResult attribute references in compress handler**
- **Found during:** Task 2 (pre-test verification)
- **Issue:** Handler referenced `result.summaries_created` and `result.commits_removed` which don't exist on CompressResult
- **Fix:** Changed to `len(result.summary_commits)` and `len(result.source_commits)`
- **Commit:** 01ddc71

**2. [Rule 1 - Bug] Fixed DiffStat attribute name in diff handler**
- **Found during:** Task 2 (pre-test verification)
- **Issue:** Handler referenced `result.stat.token_delta` but DiffStat uses `total_token_delta`
- **Fix:** Changed to `result.stat.total_token_delta`
- **Commit:** 01ddc71

## Test Results

- **37 new tests**, all passing
- **869 total tests** (798 existing + 34 orchestrator + 37 toolkit), zero regressions
- Test duration: ~5s for toolkit tests, ~82s for full suite

## Verification

1. `Tract.as_tools()` returns tool definition dicts ready for LLM API consumption -- VERIFIED
2. Three built-in profiles curate appropriate tool subsets with scenario-appropriate descriptions -- VERIFIED
3. User can override descriptions on top of any profile -- VERIFIED
4. ToolExecutor dispatches tool call dicts to Tract methods and returns structured results -- VERIFIED
5. All tests pass with zero regressions -- VERIFIED (869/869)

## Next Phase Readiness

Plan 07-01 provides the foundation for Plan 07-03 (Orchestrator Integration). The toolkit is independently useful: external agents can use `Tract.as_tools()` to get tool schemas and `ToolExecutor` to dispatch calls without the built-in orchestrator.
