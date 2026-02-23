# Tract Usage Scenarios

11 workflow stories covering every fundamental operation, then compositions that combine them. Each story has 2-3 standalone sub-scenarios that build incrementally within the story. Stories are independent — jump to any one without reading the others.

## File Tree

```
cookbook/
├── SCENARIOS.md
├── 01_conversation/
│   ├── 01_manual_commit_compile.py
│   ├── 02_shorthand_and_format.py
│   └── 03_chat_and_persist.py
├── 02_token_awareness/
│   ├── 01_status_and_budget.py
│   └── 02_budget_guardrail_loop.py
├── 03_llm_setup/
│   ├── 01_per_call_config.py
│   ├── 02_defaults_and_operations.py
│   ├── 03_per_operation_clients.py
│   └── 04_resolution_chain.py
├── 04_curation/
│   ├── 01_edit_in_place.py
│   ├── 02_pin_skip_annotate.py
│   ├── 03_atomic_batch.py
│   └── 04_important_and_retention.py
├── 05_tracing/
│   ├── 01_log_and_diff.py
│   ├── 02_time_travel.py
│   ├── 03_config_provenance.py
│   └── 04_tool_provenance.py
├── 06_branching/
│   ├── 01_branch_switch_delete.py
│   ├── 02_merge_strategies.py
│   └── 03_cherry_pick_rebase.py
├── 07_compression/
│   ├── 01_manual_compression.py
│   ├── 02_llm_compression.py
│   ├── 03_gc.py
│   └── 04_retention_aware_compression.py
├── 08_multi_agent/
│   ├── 01_parent_child.py
│   └── 02_delegation.py
├── 09_policies/
│   ├── 01_builtin_policies.py
│   ├── 02_custom_policy.py
│   └── 03_autonomy_spectrum.py
├── 10_orchestrator/
│   ├── 01_toolkit.py
│   ├── 02_trigger_orchestration.py
│   └── 03_hitl_orchestration.py
├── 11_retry/
│   ├── 01_validate_and_retry.py
│   ├── 02_chat_with_validation.py
│   └── 03_custom_retry_pipeline.py
├── 12_integrations/
│   └── 01_agno_sidecar.py
└── compositions/
    ├── ab_testing.py
    ├── context_forensics.py
    ├── drift_steering.py
    ├── full_autonomous.py
    ├── human_interjection.py
    ├── long_running_session.py
    ├── streaming_integration.py
    ├── undo_redo.py
    └── self_correcting_agent.py
```

---

# Part 1: Workflow Stories

## 01 — Having a Conversation

The core loop: record messages, compile context, talk to an LLM, persist.

### 01/01 — Manual Commit and Compile

**Use case:** You want to understand what tract is actually doing under the hood.

Open a tract, `commit()` a system prompt, a user message, and an assistant response manually (specifying role and content_type). Call `compile()` to see the resulting message list. Inspect `CompiledContext.messages` — each has role, content_type, and content.

> `Tract.open()`, `commit(role=, content_type=, content=)`, `compile()`, `CompiledContext.messages`

### 01/02 — Shorthand and Format Methods

**Use case:** You know how commit/compile works and want the convenience layer.

Replace manual commits with `system()`, `user()`, `assistant()` — same result, less boilerplate. Then format the compiled context for your LLM provider: `to_openai()`, `to_anthropic()`, or `to_dicts()` for raw output.

> `system()`, `user()`, `assistant()`, `to_openai()`, `to_anthropic()`, `to_dicts()`

### 01/03 — Chat and Persist

**Use case:** A coding assistant that chats, persists to disk, and resumes the next day.

`chat()` does everything in one call: commits the user message, compiles context, calls the LLM, commits the response, records token usage. Close the tract, reopen from the same path — the full conversation is restored. Walk the log to confirm.

> `chat()`, `ChatResponse`, persistence, `log()`

**Existing example:** `first_conversation.py` (moves here)

---

## 02 — Token Awareness

Knowing what you're spending and staying within limits.

### 02/01 — Status and Budget

**Use case:** You want to know how many tokens are in the context and how close you are to the limit.

Set a token budget via `TractConfig(token_budget=...)`. Call `status()` to see commit count, estimated token count, budget max, and budget percentage. The budget is a tracking tool — it doesn't block commits.

> `TractConfig(token_budget=)`, `status()`, `budget_pct`

### 02/02 — Budget Guardrail Loop

**Use case:** A chatbot that checks budget before every LLM call and stops when it's running hot.

In a multi-turn loop, check `status()` before each `chat()` call. After the call, usage is auto-recorded from the API response. When budget exceeds a threshold, stop or take action (compress, branch, etc.).

> `status()` in a loop, `chat()` auto-records usage, `record_usage()` for manual calls

**Existing example:** `token_budget_guardrail.py` (moves here)

---

## 03 — LLM Setup

Controlling which model, which settings, and which client — at every granularity.

### 03/01 — Per-Call Config

**Use case:** You want this specific call to use temperature 0.9, just this once.

Pass an `LLMConfig` or use sugar params (`temperature=`, `model=`) directly on `chat()` or `generate()`. Use `generate()` for two-step control where you inspect the response before committing.

> `LLMConfig(temperature=0.9)`, `chat(temperature=0.9)`, `generate(llm_config=...)`

### 03/02 — Defaults and Per-Operation Config

**Use case:** Chat should be creative, compression should be deterministic, and everything else uses a sane default.

Set tract-level defaults with `default_config=LLMConfig(...)` on open. Override per-operation with `configure_operations(chat=LLMConfig(...), compress=LLMConfig(...))`. Each operation inherits the default and applies its own overrides.

> `default_config=`, `configure_operations()`, `OperationConfigs`

### 03/03 — Per-Operation Clients

**Use case:** Chat goes to OpenAI, compression goes to local Ollama, merge conflict resolution goes to Anthropic.

`configure_clients()` assigns a different LLM *client* to each operation. LLMConfig controls *what settings* to use; LLMClient controls *where to send the request*. They're fully decoupled. Per-operation clients are user-managed — `close()` only closes the tract's own internally-created client.

> `configure_clients(chat=openai, compress=ollama)`, `OperationClients`, client lifecycle

### 03/04 — Resolution Chain and Cross-Framework Config

**Use case:** You need to understand which setting wins, and your config comes from an OpenAI-style JSON dict.

The 4-level chain resolves each field independently: sugar > llm_config > operation > default. `LLMConfig.from_dict()` handles cross-framework aliases (`stop` -> `stop_sequences`, `max_completion_tokens` -> `max_tokens`). Every resolved config is auto-captured on assistant commits for provenance.

> 4-level resolution, `LLMConfig.from_dict()`, auto-captured `generation_config`

**Existing example:** `config_hierarchy.py` (moves here)

---

## 04 — Curating Context

Shaping what the LLM sees without losing history.

### 04/01 — Edit in Place

**Use case:** The assistant said "60 day return policy" but it's 30 days. Fix it without cluttering the conversation.

Commit with `operation=EDIT` and `edit_target=original_hash`. The next `compile()` serves the corrected content as if the original never existed. The original commit is still in history for audit.

> `commit(operation=EDIT, edit_target=original_hash)`

### 04/02 — Pin, Skip, and Reset Annotations

**Use case:** Pin your system prompt so it survives compression. Skip noisy tool outputs so they don't bloat context. Un-skip something when you realize you need it.

`annotate(hash, PINNED)` protects a message from compression and guarantees inclusion. `annotate(hash, SKIP)` hides it from compiled context while keeping it in history. `annotate(hash, NORMAL)` removes any annotation.

> `annotate(hash, PINNED)`, `annotate(hash, SKIP)`, `annotate(hash, NORMAL)`

### 04/03 — Atomic Batch

**Use case:** A RAG retrieval + user question + assistant response should land as one unit or not at all.

Wrap multiple commits in `batch()`. If any commit fails or an exception is raised, all commits in the batch roll back. Useful for multi-step pipelines where partial state is worse than no state.

> `with t.batch(): ...`

**Existing example:** batch portion of `atomic_batch.py` (moves here, config/provenance parts split to 03 and 05)

### 04/04 — IMPORTANT Priority and Retention Criteria

**Use case:** Some context is too important to lose in compression, but not important enough to pin forever. You want the LLM to preserve specific details when summarizing.

`annotate(hash, IMPORTANT)` tells compression to be conservative. Add fuzzy guidance with `retain="preserve all dollar amounts and deadlines"` — injected into the summarization prompt. Add deterministic checks with `retain_match=["$50k", "2026-03-01"]` — validated against the output. Both can combine. Shorthand works too: `t.user(..., priority=IMPORTANT, retain="keep the budget figures")`.

> `annotate(hash, IMPORTANT)`, `retain=`, `retain_match=`, `RetentionCriteria`, shorthand with `priority=IMPORTANT`

---

## 05 — Tracing and Audit

Understanding what happened, when, why, and with what settings.

### 05/01 — Log and Diff

**Use case:** You want to see the conversation history and understand what changed between two points.

`log()` returns every commit with hash, role, content type, message, and timestamp. `diff()` compares two commits and shows added, removed, and modified messages.

> `log()`, `diff(earlier_hash, later_hash)`

### 05/02 — Time Travel

**Use case:** An agent gave a bad answer 15 turns ago. Reconstruct exactly what it was seeing.

`compile(at_commit=hash)` rebuilds the message list as of any past commit. `compile(at_time=datetime)` does the same by timestamp. `checkout(hash)` moves HEAD there for interactive inspection. `reset(hash)` moves HEAD backward permanently (orphaned commits survive until GC).

> `compile(at_commit=)`, `compile(at_time=)`, `checkout()`, `reset()`

### 05/03 — Config Provenance

**Use case:** "Which model produced this output? What temperature was used?"

Every assistant commit stores the fully-resolved `generation_config`. Query it with `query_by_config()` — single-field, multi-field AND, or whole-config matching. Also supports the IN operator for multi-value queries.

> `commit_info.generation_config`, `query_by_config(model="gpt-4o")`, `query_by_config(temperature=0.7, model="gpt-4o")`

### 05/04 — Tool Provenance

**Use case:** "What tools were available when this response was generated? Did the tool set change mid-session?"

`set_tools([...])` registers tool schemas that auto-link to subsequent commits. Each unique schema is content-hashed and stored once — the same tool across 100 commits costs one row. `get_commit_tools(hash)` reconstructs exactly what tools a commit had. `compile()` populates `CompiledContext.tools` with the latest active set. `to_openai_params()` and `to_anthropic_params()` return full API-ready dicts with both messages and tools, while `to_openai()` stays backward-compatible (messages only).

> `set_tools()`, `get_commit_tools()`, `to_openai_params()`, `to_anthropic_params()`, `hash_tool_schema()`

---

## 06 — Branching and Exploration

Parallel timelines for experimentation, merging results back.

### 06/01 — Branch, Switch, Delete

**Use case:** Try an experimental approach without affecting main. Clean up when done.

`branch("experiment")` creates a new timeline from current HEAD. `switch("experiment")` moves to it. Work there, then `switch("main")` to come back. `delete_branch("experiment")` removes the pointer when you're done.

> `branch()`, `switch()`, `delete_branch()`

### 06/02 — Merge Strategies

**Use case:** The experiment worked. Bring it back to main.

Three merge modes: **fast-forward** (branch is ahead of main, just advance the pointer), **clean** (branches diverged but no conflicts, auto-merge), **conflict** (branches have overlapping edits, LLM resolves). All via `merge()`.

> `merge("experiment")` — FF / clean / conflict

### 06/03 — Cherry-Pick and Rebase

**Use case:** Grab one useful commit from an experiment, and update a stale branch to include the latest main.

`cherry_pick(hash)` copies a single commit onto the current branch. `rebase("main")` replays the current branch's commits on top of main's tip — with safety checks warning if edit_target chains would break.

> `cherry_pick(hash)`, `rebase("main")`

---

## 07 — Compression and Memory

Keeping conversations alive past the context window limit.

### 07/01 — Manual Compression

**Use case:** You know exactly what the summary should say. No LLM needed.

Pass your own text to `compress(content="...")`. The original commits in the range are archived and replaced with a single summary commit. Deterministic — same input always produces same output.

> `compress(from_commit=a, to_commit=b, content="...")`

### 07/02 — LLM and Collaborative Compression

**Use case:** Let the LLM summarize, with optional human review.

`compress(target_tokens=2000)` auto-summarizes using the configured LLM. PINNED commits pass through untouched; SKIP commits are excluded. For human review, use `auto_commit=False` — the LLM drafts a summary, you inspect and `approve_compression()`.

> `compress(target_tokens=)`, `compress(auto_commit=False)`, `approve_compression()`, pinned survives

### 07/03 — Garbage Collection

**Use case:** Archived pre-compression commits are piling up. Reclaim storage.

`gc()` removes orphaned commits older than N days. Archived commits can have a separate retention window. Non-destructive to any reachable commit chain.

> `gc(orphan_retention_days=7, archive_retention_days=30)`

### 07/04 — Retention-Aware Compression

**Use case:** Compressing a long conversation that contains critical financial figures. You need to guarantee those figures survive summarization.

IMPORTANT commits with `retain_match=["$50k", "2026-03-01"]` get deterministic validation after summarization. If the summary is missing a required substring or regex, `retry_with_steering()` amends the prompt with the diagnosis and retries. Fuzzy `retain=` instructions are injected into the summarization prompt to guide the LLM. Combines IMPORTANT priority (story 04/04) with the retry protocol (story 11).

> `compress()` with IMPORTANT commits, `retain_match=` validation, `retry_with_steering()`, enriched summarization prompt

---

## 08 — Multi-Agent

Coordinating context across parent and child agents.

### 08/01 — Parent-Child Relationship

**Use case:** A main agent needs to spawn a sub-agent with its own isolated context.

Create a child tract linked to the parent. The child has independent commit history but the relationship is tracked for provenance — you can always trace which parent spawned which child.

> `parent()`, `children()`

### 08/02 — Sub-Agent Delegation

**Use case:** Spawn a research sub-agent, let it work for 40 turns, ingest just the summary.

The sub-agent works in its child tract. When finished, compress its history into a summary. The parent commits that summary as a single message — 40 turns collapsed into one commit on the parent's timeline.

> Child work -> `compress()` -> parent `commit()` with summary

---

## 09 — Policies

Automated rules that fire on events without manual intervention.

### 09/01 — Built-In Policies

**Use case:** Auto-compress at 80% budget, auto-pin system prompts, detect topic drift, archive stale branches.

Four built-in policies cover common needs: `CompressPolicy(threshold=0.8)`, `PinPolicy()`, `BranchPolicy()`, `ArchivePolicy(inactive_days=7)`. Configure one or all at once.

> `configure_policies([CompressPolicy(0.8), PinPolicy(), BranchPolicy(), ArchivePolicy()])`

### 09/02 — Custom Policy

**Use case:** Auto-skip any commit containing PII, or auto-skip tool outputs older than 10 turns.

Implement the `Policy` protocol: define `name`, `trigger`, and `evaluate(tract) -> PolicyAction | None`. Your policy inspects commits, proposes annotations or operations, and the evaluator executes them.

> `Policy` protocol: `name`, `priority`, `trigger`, `evaluate()`

### 09/03 — Autonomy Spectrum

**Use case:** In dev, approve everything. In staging, auto-approve safe ops. In prod, full auto.

Same policies, three modes. **Manual:** every action is a proposal. **Collaborative:** low-risk (pin, skip) auto-executes, high-risk (compress, branch) needs approval. **Autonomous:** everything fires immediately.

> `autonomy="manual"`, `"collaborative"`, `"autonomous"`

---

## 10 — Toolkit and Orchestrator

The agent manages its own context window end-to-end.

### 10/01 — Toolkit: Expose Operations as LLM Tools

**Use case:** Let the LLM decide when to compress, branch, or annotate via function calling.

`as_tools()` returns tract operations formatted for OpenAI or Anthropic tool schemas. Three profiles control scope: **self** (full CRUD), **supervisor** (read + high-level), **observer** (read-only).

> `as_tools(format="openai", profile="self")`, `as_tools(format="anthropic", profile="supervisor")`

### 10/02 — Trigger-Based Orchestration

**Use case:** Auto-assess context health every 20 commits or at 70% token usage.

Configure triggers on the orchestrator. When a trigger fires, it assesses context (fragmentation, budget pressure, stale branches) and executes tools (compress, GC, cleanup) in an autonomous loop.

> `OrchestratorConfig(triggers=TriggerConfig(on_commit_count=20, on_token_threshold=0.7))`

### 10/03 — Human-in-the-Loop Orchestration

**Use case:** Compression and merges need human sign-off for compliance.

Configure a callback. The orchestrator proposes actions with reasoning. The human approves, rejects, or modifies. The orchestrator executes the approved action and continues.

> `OrchestratorConfig(callbacks=callback_fn)`

---

## 11 — Retry and Validation

Validate LLM output and retry with steering when it fails. The retry protocol is operation-agnostic — it works for chat, compression, merge resolution, or any callable.

### 11/01 — Validate and Retry (Core Primitive)

**Use case:** You have a function that produces unreliable output. You want to validate, steer, and retry — without coupling to any specific tract operation.

`retry_with_steering()` takes four callables: `attempt` (produces a result), `validate` (checks it), `steer` (injects correction), and `head_fn`/`reset_fn` (for optional history management). It loops until validation passes or retries are exhausted. Returns `RetryResult` with the value, attempt count, and failure history. Raises `RetryExhaustedError` if all retries fail. No Tract dependency — pure logic + callbacks.

> `retry_with_steering()`, `RetryResult`, `RetryExhaustedError`, `validate`, `steer`

### 11/02 — Chat with Validation

**Use case:** An agent must return valid JSON. If the LLM returns malformed output, steer it and retry automatically.

`chat(validator=my_validator, max_retries=3)` wraps the LLM call in the retry loop. On failure, a steering message with the diagnosis is committed as a user message — the LLM sees its mistake in context. `purify=True` resets HEAD to before the retry chain and re-commits only the clean result. `provenance_note=True` adds a meta-commit recording how many retries it took.

> `chat(validator=, max_retries=, purify=, provenance_note=, retry_prompt=)`

### 11/03 — Custom Retry Pipeline

**Use case:** Wrap any operation — compression, merge resolution, a custom pipeline step — with validate-steer-retry.

Build your own retry pipeline by providing closures to `retry_with_steering()`. Validate compression summaries for key terms, steer by amending the prompt, retry merge resolutions with different strategies. The same primitive handles all cases — only the callables change.

> `retry_with_steering()` with custom `attempt`/`validate`/`steer`, wrapping arbitrary operations

---

## 12 — Framework Integrations

Using Tract alongside agent frameworks. Tract handles versioning and provenance; the framework handles the execution loop.

### 12/01 — Agno Sidecar

**Use case:** You're building an agent with Agno (tools, multi-turn, session management) but want version-controlled context history — branching, time-travel, compression, and config provenance that Agno doesn't provide.

Run Tract as a sidecar alongside Agno. Agno owns the tool call loop and live session. After each Agno run, commit messages into Tract. Now you can branch conversations, compress old context, time-travel to debug bad responses, and audit tool/config provenance — all on top of Agno's agent execution.

> Agno `Agent`, Agno `Toolkit`, Tract sidecar pattern, `t.user()`, `t.assistant()`, `t.branch()`, `t.compile(at_commit=)`

---

# Part 2: Compositions

Real-world scenarios that combine features across multiple stories. Each references the stories it builds on.

### X.1 — First Full Conversation

**Combines:** 01 (conversation) + 02 (token awareness)

A coding assistant chats across multiple turns, checks budget before each call, persists overnight, and resumes the next day.

### X.2 — Correcting and Protecting

**Combines:** 04 (curation) + 01 (conversation)

A support agent's hallucination is edited in-place, the correction is pinned, and the next compile serves clean context.

### X.3 — Debugging a Bad Response

**Combines:** 05 (tracing) + 04 (curation)

Walk the log, time-travel to the bad response, reconstruct what the LLM saw, diff against current state.

### X.4 — A/B Testing Model Configs

**Combines:** 06 (branching) + 03 (config) + 05 (tracing)

Branch from the same state, run identical prompts with different configs, diff results, query by config to compare.

### X.5 — RAG Pipeline

**Combines:** 04 (curation — batch) + 03 (config) + 05 (tracing — provenance)

Batch retrieval + question atomically, call LLM with specific config, verify provenance on the response.

### X.6 — Undo / Redo

**Combines:** 05 (tracing — reset) + 06 (branching — cherry-pick)

Soft reset to undo, try a new approach, cherry-pick the original back if the new one is worse.

### X.7 — Drift Steering

**Combines:** 09 (policies) + 10 (orchestrator) + 07 (compression)

BranchPolicy detects drift, orchestrator compresses the tangent, merges summary back to main.

### X.8 — Human Interjection Mid-Conversation

**Combines:** 10 (orchestrator) + 04 (curation) + 09 (policies)

Pause orchestrator, edit the wrong commit, pin the correction, resume.

### X.9 — Context Forensics

**Combines:** 05 (tracing) + 06 (branching)

Walk log to find bad data entry point, branch from before contamination, cherry-pick good work, rebase clean.

### X.10 — Long-Running Autonomous Session

**Combines:** 07 (compression) + 09 (policies) + 10 (orchestrator)

CompressPolicy fires at 80%, pinned alerts survive, GC reclaims archives. Runs for 8 hours unattended.

### X.11 — Streaming Integration

**Combines:** 01 (conversation) + 04 (curation — edit)

Compile and stream from LLM, commit full response on completion, EDIT if interrupted and retried.

### X.12 — Sub-Agent Research Team

**Combines:** 08 (multi-agent) + 06 (branching) + 07 (compression)

Three sub-agents research in parallel, compress their findings, supervisor merges summaries on main.

### X.13 — Fully Autonomous Agent

**Combines:** 09 (policies) + 10 (orchestrator) + 07 (compression) + 06 (branching)

All policies active, orchestrator on triggers, agent self-manages via toolkit, GC runs periodically. Full autopilot.

### X.14 — Self-Correcting Agent

**Combines:** 11 (retry) + 04 (curation — IMPORTANT priority) + 07 (compression) + 05 (tracing — tool provenance)

An agent validates its own JSON output via `chat(validator=json_validator)`, retries on failure with steering. Critical decisions are marked IMPORTANT with `retain_match=` for key terms. Tools are tracked per-commit for provenance. When the session compresses, retention criteria guarantee key details survive. Audit the full session later: `get_commit_tools()` and `query_by_config()` reconstruct exactly what the agent had available at each step.
