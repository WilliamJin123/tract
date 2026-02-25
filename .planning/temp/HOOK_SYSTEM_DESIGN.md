# Unified Hook System Design

## Problem

Tract has three separate interception patterns that solve the same problem differently:
- **PendingCompression**: mutable object with `_commit_fn` closure, `approve()` only
- **PolicyProposal**: dual callbacks (`_execute_fn`/`_reject_fn`), DB persistence (being removed)
- **OrchestratorProposal**: external callback returns `ProposalResponse(APPROVED/MODIFIED/REJECTED)`

These should be one system. Operations that destroy history (gc), transform history (compression, rebase), or involve judgment calls (merge resolution, policy actions) all need the same thing: build a plan, optionally intercept it, then execute or abort.

## Principles

1. **Commits record, hooks gate.** Append/edit commits are atomic provenance events — don't intercept recording. Only intercept operations that destroy, transform, or involve judgment.
2. **The granularity itself is automatable.** The interception layer is a single interface where either a human or an agent can sit. Same Pending, same methods, different occupant.
3. **Deterministic or fuzzy at every decision point.** Every slot in the pipeline (trigger, review, retry, adaptation) can be a simple `if` statement, an LLM call, or a combination. The infrastructure doesn't care — it provides structured data and clean action methods.
4. **The handler owns control.** The hook handler decides how many retries, with what guidance, when to escalate, when to bail. The operation provides tools (`retry()`, `validate()`), the handler drives.
5. **Judgment and execution are separable.** Any LLM operation can be decomposed into a guidance stage (what should this cover?) and an execution stage (produce the output). The guidance slot can be filled by a human, an LLM, or both. The infrastructure treats guidance as a first-class editable field, not an internal detail.

## Design

### 1. Pending Base

Every hookable operation produces a `Pending` — a mutable container with methods to approve, reject, modify, or retry. Fields: `operation`, `pending_id`, `created_at`, `tract` (full access), `status`, `triggered_by`, `rejection_reason`.

**Core methods:** `approve()`, `reject()` — subclass-specific. **Agent interface:** `to_dict()`, `to_tools()`, `describe_api()` — auto-generated from subclass methods. **Dispatch:** `apply_decision(dict)`, `execute_tool(name, args)` — guarded by `_public_actions` whitelist. **Interactive:** `pprint()`, `review()`.

### 2. Pending Subclasses

| Subclass | Key Fields | Actions |
|---|---|---|
| `PendingCompress` | summaries, source_commits, preserved_commits, original_tokens, estimated_tokens, guidance, guidance_source | approve, reject, edit_summary, edit_guidance, regenerate_guidance, retry, validate, edit_interactive |
| `PendingGC` | commits_to_remove, tokens_to_free | approve, reject, exclude |
| `PendingRebase` | replay_plan, target_base, warnings | approve, reject, exclude |
| `PendingMerge` | resolutions, source_branch, target_branch, conflicts, guidance, guidance_source | approve, reject, edit_resolution, edit_guidance, regenerate_guidance, retry, edit_interactive |
| `PendingPolicy` | policy_name, action_type, action_params, reason | approve, reject, modify_params |

All subclasses store internal `_execute_fn` closures. Guidance fields appear on operations with LLM output (compress, merge).

### 3. Hook Registration & Routing

**API:** `t.on("compress", handler)`, `t.off("compress")`, `t.hooks` (dict). `t.on("*", handler)` for catch-all fallback. `ValueError` on non-hookable operations. One hook per operation.

**Handler signature:** `def handler(pending: Pending) -> None` — handler calls methods on the Pending, no return value.

**Three-tier routing** (inside every hookable operation): `review=True` (return Pending to caller) > registered hook (`_fire_hook`) > auto-approve. `review=True` replaces old `auto_commit=False`.

**Recursion guard:** Global `_in_hook: bool` flag. When inside a hook handler, all hookable operations auto-approve — prevents direct recursion and indirect cycles (compress->gc->compress). Handler can still call operations, they just skip hook interception.

**Unresolved handler guard:** If handler returns without calling approve/reject, emit `warnings.warn()`.

**Policy feedback routing:** After hook completes, if `triggered_by` is set, route outcome to `policy.on_rejection()` or `policy.on_success()`.

### 4. Retry Integration

Handler owns the retry loop. `retry(index, *, guidance="", **llm_overrides)` is a single-step method — re-runs LLM for one summary/resolution with guidance injected into the prompt (previous output + feedback). Handler decides escalation: retry -> stronger model -> manual edit -> give up.

**`auto_retry(pending, max_retries=3)`** — convenience wrapper in `tract.hooks` for standard validate->retry loops.

**`ValidationResult`**: `passed: bool`, `diagnosis: str | None`, `index: int | None`.

**`retry_with_steering()`** in `retry.py` stays for `chat()`/`generate()`. Hookable operations use `retry()` on the Pending instead.

### 5. Policy + Hook Composition

**Lifecycle:** Event -> PolicyEvaluator -> Policy fires -> `t.compress(triggered_by="policy:auto_compress")` -> PendingCompress -> hook fires -> handler approves/rejects -> feedback to policy via `on_rejection()`/`on_success()`.

**Policy ABC additions:** `default_handler(pending)` — override for policy-specific review logic, overridden by user hooks. `on_rejection(rejection: HookRejection)` — adapt behavior on rejection (cooldown, adjust params). `on_success(result)` — learn from success.

**HookRejection:** `reason`, `pending` (full), `rejection_source` ("hook"/"handler"/"validation"), `metadata`.

**Three-tier handler precedence:** User hook (`t.on()`) > Policy `default_handler()` > Auto-approve.

**Replaces:** PolicyProposal closures -> Pending.approve()/reject(). PolicyEvaluator `on_proposal` -> hook system. OrchestratorProposal -> hook system. Orchestrator becomes a template, not core infra.

### 6. Two-Stage Guidance Pattern

LLM operations decompose into guidance (what should the output cover?) and execution (produce the output). Guidance is the harder cognitive task — judgment about what matters.

**`guidance_source`:** `None` (one-shot) | `"user"` (instructions param) | `"llm"` (two_stage=True) | `"user+llm"` (instructions + improve=True). Same slot, different occupant.

**Three-level opt-in:** `configure_operations(compress={"two_stage": True})` (default) | per-call `t.compress(two_stage=True/False)` | policy decides dynamically via `PolicyAction` params.

**Handler interaction:** Guidance is a visible, editable field on the Pending. Handler can `edit_guidance()` -> `retry()`, or `regenerate_guidance()` -> `retry()`, or provide fully manual guidance. Flat model — no nested hooks.

### 7. Event Taxonomy

**Hooked:** `compress` (PendingCompress), `gc` (PendingGC), `rebase` (PendingRebase), `merge` conflicts (PendingMerge), policy actions (PendingPolicy). All destructive, transformative, or judgment operations.

**Not hooked:** commits (`commit/user/assistant/system/import_commit`), LLM calls (`chat/generate` — validator pattern), reads (`log/status/diff/compile`), lightweight edits (`annotate/edit/edit_history/restore`), pointer ops (`branch/switch/checkout/reset`), fast-forward merge. `ValueError` on `t.on()` for any of these.

### 8. Serialization & Agent Interface

Four interaction patterns, handler chooses:

| Pattern | Method | Use case |
|---|---|---|
| Direct Python | Call methods directly | Deterministic handlers, human REPL |
| Structured dicts | `to_dict()` -> LLM -> `apply_decision(dict)` | One-shot LLM decisions |
| Tool use protocol | `to_tools()` -> LLM tool loop -> `execute_tool(name, args)` | Multi-turn LLM with feedback |
| Code generation | `describe_api()` -> LLM writes Python -> `exec()` | Maximum flexibility, requires trust |

`apply_decision()` and `execute_tool()` validate against `_public_actions` whitelist — blocks access to internal methods (`_execute_fn`, etc.).

### 9. Content Improvement (`improve=True`)

Pairs human intent with LLM articulation. Not a hook — uses existing patterns.

**On content** (user messages, summaries, resolutions): EDIT commit pattern. Original committed first, LLM improvement is an EDIT. `restore(version=0)` recovers original. LLM failure -> original stands, warning emitted. Works on: `t.user()`, `t.compress(content=...)`, `pending.edit_resolution()`, commit messages.

**On instructions** (operational metadata): No commit to EDIT. Original and improved stored on `OperationEventRow` (`original_instructions` / `effective_instructions`). Works on: `t.compress(instructions=...)`, policy config.

**Improvement style inferred from context:** polish (messages), expand (summaries), translate (natural language -> config). No explicit mode selection.

### 10. CLI Integration

Every hookable CLI command gets `--review`: passes `review=True`, gets Pending, runs interactive flow (pprint -> edit_interactive -> confirm -> approve/reject). Replaces `--edit` on compress.

## Deletions & Migrations

**Deleted:** `PendingCompression`, `PolicyProposal`, `PolicyProposalRow` persistence (audit-only via OperationEventRow), `PolicyEvaluator.get_pending_proposals()`/`approve_proposal()`/`_reconstruct_proposal_fn()`, `OrchestratorProposal`, `ProposalResponse`, `PolicyEvaluator.on_proposal` callback, `orchestrator/callbacks.py`, orchestrator as core infra.

**Stays:** `retry_with_steering()` for chat/generate, `PolicyEvaluator` (without proposal system), Policy ABC (gains default_handler/on_rejection/on_success), all non-hookable operations. `auto_commit` renamed to `review` (inverted).

## Concurrency Model

One Tract instance per agent/coroutine. Multiple agents share the **database** (SQLite WAL), not the Python object. This matches the git model: multiple processes work on the same repo without sharing in-memory state. Cross-tract merge is the structural composition primitive for bringing results back together.

## Resolved Decisions

1. **Async handlers**: Sync only. Entire Tract API is synchronous. Async is an additive future enhancement (detect via `inspect.iscoroutinefunction`), not a blocker.
2. **Multiple hooks per operation**: One hook per operation. Composition happens inside the handler. No chaining, no ordering questions.
3. **Pending persistence**: Removed entirely. Pending is ephemeral — lives in memory for the duration of the handler or caller scope. Hook system's inline execution model eliminates the temporal gap that required PolicyProposal DB persistence. Savepoint rollback handles crash safety. `OperationEventRow` provides the audit trail.
4. **Catch-all ordering**: Fallback only. `get(operation) or get("*")` — specific hook wins, `*` covers unconfigured operations.
5. **Test migration**: Clean break. Pre-1.0, no external consumers. Replace old classes directly, migrate tests in each phase.
6. **Two-stage prompt design**: Design during implementation. Interface is defined (guidance field, regenerate_guidance). Actual prompts are empirical — start with compress, iterate.
7. **Improvement quality**: LLM-as-judge + EDIT safety net. Original always recoverable via `restore()`. Low risk.
8. **getattr whitelist**: `_public_actions` set on each Pending subclass. `apply_decision()` and `execute_tool()` validate against it. Defense-in-depth for agent interaction patterns.

## Implementation Phases

### Phase 1: Core Infrastructure + Compression (proof of concept)

End-to-end validation of the hook system on the most mature hookable operation.

**New files:**
- `src/tract/hooks/__init__.py` — public API (`auto_retry`, `ValidationResult`)
- `src/tract/hooks/pending.py` — `Pending` base class with `_public_actions` whitelist, status management, `apply_decision()`/`execute_tool()` guarded dispatch
- `src/tract/hooks/compress.py` — `PendingCompress` subclass (summaries, source_commits, preserved_commits, tokens, guidance fields, approve/reject/retry/edit_summary/validate/edit_interactive)
- `src/tract/hooks/gc.py` — `PendingGC` stub (approve/reject/exclude — full wiring in Phase 2)
- `src/tract/hooks/rebase.py` — `PendingRebase` stub
- `src/tract/hooks/merge.py` — `PendingMerge` stub
- `src/tract/hooks/policy.py` — `PendingPolicy` stub
- `src/tract/hooks/validation.py` — `ValidationResult`, `HookRejection`
- `src/tract/hooks/retry.py` — `auto_retry()` convenience

**Modified files:**
- `src/tract/tract.py` — Add `_hooks: dict`, `_in_hook: bool`, `on()`, `off()`, `hooks` property, `_fire_hook()`, `_HOOKABLE_OPS` set. Refactor `compress()`: replace `auto_commit` with `review`, build `PendingCompress`, three-tier routing.
- `src/tract/operations/compression.py` — `compress_range()` always returns `PendingCompress` (no more dual return type). The caller (Tract.compress) handles routing.
- `src/tract/cli/commands/compress.py` — `--review` replaces `--edit`, uses `pending.review()` flow
- `src/tract/cli/__init__.py` — update compress command registration if needed

**Deleted:**
- `PendingCompression` class from `src/tract/models/compression.py`

**Tests:**
- New: `tests/test_hooks.py` — Pending base, hook registration, _fire_hook guard, three-tier routing, PendingCompress full lifecycle, auto_retry, ValidationResult, _public_actions whitelist
- Migrate: all compression tests referencing `PendingCompression` → `PendingCompress`
- Migrate: CLI compress tests (`--edit` → `--review`)

**Success criteria:**
- `t.compress()` auto-approves (no hook registered)
- `t.compress(review=True)` returns PendingCompress
- `t.on("compress", handler)` → handler fires on compress
- `_in_hook` prevents recursion
- `auto_retry()` works end-to-end
- `_public_actions` blocks `getattr` on internal methods
- `ValueError` on `t.on("commit", handler)`
- All existing compression tests pass with new API

### Phase 2: Policy + Remaining Operations

Wire all remaining hookable operations. Integrate policy feedback loop.

**Modified files:**
- `src/tract/hooks/gc.py` — Full implementation (wire to `gc()` operation)
- `src/tract/hooks/rebase.py` — Full implementation (wire to `rebase()`)
- `src/tract/hooks/merge.py` — Full implementation (wire to `merge()` conflict path)
- `src/tract/hooks/policy.py` — Full implementation (wire to PolicyEvaluator)
- `src/tract/policy/protocols.py` — Add `default_handler()`, `on_rejection()`, `on_success()` to Policy ABC
- `src/tract/policy/evaluator.py` — Remove `on_proposal` callback, `get_pending_proposals()`, `approve_proposal()`, `_reconstruct_proposal_fn()`. `_create_proposal()` → creates `PendingPolicy`, routes through `_fire_hook`. Three-tier: user hook > policy `default_handler()` > auto-approve.
- `src/tract/tract.py` — Wire `gc()`, `rebase()`, `merge()` to produce Pending + three-tier routing. Add `triggered_by` threading for policy→hook→feedback.
- `src/tract/policy/builtin/*.py` — Update 4 built-in policies for new Policy ABC methods

**Deleted:**
- `PolicyProposal` class from `src/tract/models/policy.py`
- `PolicyEvaluator.get_pending_proposals()`, `approve_proposal()`, `_reconstruct_proposal_fn()`

**Tests:**
- New: `tests/test_hooks_policy.py` — policy→hook→feedback lifecycle, default_handler precedence, on_rejection/on_success, PendingPolicy
- New: `tests/test_hooks_gc.py`, `tests/test_hooks_rebase.py`, `tests/test_hooks_merge.py`
- Migrate: all PolicyProposal tests → PendingPolicy
- Migrate: policy integration tests for new evaluator flow

**Success criteria:**
- Policy triggers compress → hook fires → handler rejects → policy.on_rejection() called
- Three-tier precedence: user hook > default_handler > auto-approve
- gc/rebase/merge all support `review=True` and hook registration
- PendingGC.exclude() removes commits from plan
- PendingRebase.exclude() removes commits from replay
- PendingMerge with conflict resolutions editable
- All existing policy/gc/rebase/merge tests pass

### Phase 3: Serialization + Agent Interface

Make Pending objects LLM-accessible. Auto-generate tool schemas and structured data from methods.

**Modified files:**
- `src/tract/hooks/pending.py` — Implement `to_dict()`, `to_tools()`, `describe_api()` with introspection-based auto-generation. `pprint()` with Rich. `review()` interactive flow.
- All Pending subclasses — Ensure `_public_actions` is comprehensive, method docstrings are LLM-readable

**New files:**
- `src/tract/hooks/introspection.py` — Method-to-tool-schema conversion, method-to-dict serialization, API description generation. Reads type hints, docstrings, parameter defaults.

**Tests:**
- New: `tests/test_hooks_serialization.py` — to_dict round-trip, to_tools schema validity, describe_api completeness, apply_decision routing, execute_tool whitelist enforcement, pprint smoke test

**Success criteria:**
- `to_tools()` produces valid JSON Schema tool definitions
- `apply_decision({"action": "approve"})` calls approve()
- `apply_decision({"action": "_execute_fn"})` raises ValueError (whitelist)
- `execute_tool("reject", {"reason": "bad"})` works
- `execute_tool("_commit_fn", {})` raises ValueError
- Adding a method to a subclass auto-appears in to_dict/to_tools/describe_api

### Phase 4: Two-Stage Guidance + improve=True

Add the judgment/execution split and content improvement capabilities.

**New files:**
- `src/tract/hooks/guidance.py` — `GuidanceMixin` with `guidance`, `guidance_source`, `edit_guidance()`, `regenerate_guidance()` fields/methods. Mixed into PendingCompress and PendingMerge.
- `src/tract/prompts/guidance.py` — Stage 1 prompts for compress guidance, merge guidance
- `src/tract/prompts/improve.py` — Content improvement prompts (polish/expand/translate based on context)
- `src/tract/hooks/improve.py` — `_improve_content()` helper: commit original → EDIT with improved. `_improve_instructions()`: store both on OperationEventRow.

**Modified files:**
- `src/tract/hooks/compress.py` — Mix in GuidanceMixin, wire `two_stage=True` to guidance generation before execution
- `src/tract/hooks/merge.py` — Mix in GuidanceMixin for conflict resolutions
- `src/tract/tract.py` — `improve=True` on `user()`, `assistant()`, `system()`, `compress()`. Two-stage config via `configure_operations()`.
- `src/tract/storage/schema.py` — Add `original_instructions` / `effective_instructions` columns to OperationEventRow (if not already present)

**Tests:**
- New: `tests/test_hooks_guidance.py` — two-stage compress, guidance editing, regenerate_guidance, guidance_source tracking
- New: `tests/test_improve.py` — content improvement EDIT chain, instruction improvement on OperationEventRow, restore() recovers original, LLM failure leaves original intact

**Success criteria:**
- `t.compress(two_stage=True)` produces guidance before summaries
- Handler can edit guidance and retry → new summaries reflect updated guidance
- `t.user("rough text", improve=True)` commits original then EDITs with improved version
- `restore(version=0)` recovers pre-improvement content
- LLM failure on improve → original stands, warning emitted
- `t.compress(instructions="...", improve=True)` stores both original and effective instructions

### Phase 5: Orchestrator Migration

Refactor the orchestrator from core infrastructure to a template built on hooks + policies.

**Modified files:**
- `src/tract/orchestrator/loop.py` — Refactor to use hook system. Tool proposals become `PendingPolicy` or direct hook calls. Assessment → policy evaluation → hook gating.
- `src/tract/orchestrator/config.py` — Simplify. Autonomy levels map to hook configurations.
- `src/tract/orchestrator/models.py` — Remove `OrchestratorProposal`, `ProposalResponse`, `ProposalDecision`. Keep `ToolCall`, `OrchestratorState`, assessment models.

**Deleted:**
- `src/tract/orchestrator/callbacks.py` — entirely replaced by hook handlers
- `OrchestratorProposal`, `ProposalResponse`, `ProposalDecision` from models.py

**New files:**
- `src/tract/hooks/templates/orchestrator.py` — Pre-built hook handler that implements the orchestrator review pattern (assess → propose → review → execute). Shows how to build orchestrator-like behavior from hooks + policies.

**Tests:**
- Migrate: all orchestrator tests to new hook-based flow
- New: `tests/test_hooks_orchestrator.py` — orchestrator template tests

**Success criteria:**
- Orchestrator loop works with hook system (no OrchestratorProposal)
- `auto_approve`, `log_and_approve`, `cli_prompt`, `reject_all` patterns reimplemented as hook handlers
- All existing orchestrator tests pass or are migrated
- Orchestrator is importable as a template, not required infrastructure
