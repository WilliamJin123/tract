# Tract Cookbook — Scenarios

The cookbook is organized by **what you're doing**, not by difficulty level. **Basics** covers the core mental model. **Operations** groups history-modifying actions (compress, merge, branch, rebase, gc, rollback). **Metadata** covers data you attach to commits (tags, priorities, tool results, reasoning). **Config** handles LLM routing and budgets. **Queries** and **Validation** cover inspection and retry patterns. **Hooks** provide the approval layer. **Orchestrator** and **Multi-Agent** handle autonomous and multi-agent workflows. **E2E** combines everything into real-world scenarios.

Every file is standalone — jump to any individual file if you know what you need.

## File Tree

```
cookbook/
├── SCENARIOS.md
├── 00_basics/                           # Core mental model + basic operations
│   ├── 01_commit_and_compile.py         # Tract.open, commit(), compile(), CompiledContext
│   ├── 02_shorthand_and_format.py       # system/user/assistant, to_openai/anthropic/dicts
│   ├── 03_status_and_budget.py          # status(), TractConfig, token budget tracking
│   ├── 04_log_and_diff.py              # log, show, diff, compile(at_commit=), compile(at_time=)
│   ├── 05_batch_and_rollback.py         # batch() context manager, atomic operations
│   └── 06_chat_and_persist.py           # chat(), ChatResponse, persistence, session resume
│
├── operations/                          # History-modifying operations
│   ├── compress/
│   │   ├── 01_manual.py                # compress(content=), PINNED preservation, preserve=
│   │   ├── 02_llm_auto.py             # compress(target_tokens=), instructions=, system_prompt=
│   │   ├── 03_guided.py               # IMPORTANT priority + guided compression
│   │   ├── 04_collaborative_review.py  # auto_commit=False, PendingCompress, edit_summary
│   │   ├── 05_important_and_retain.py  # retain=, retain_match=, retain_match_mode=
│   │   └── sample_contract.md          # Sample data for compression demos
│   ├── merge/
│   │   ├── 01_merge_strategies.py      # FF merge, clean merge, no_ff, delete_branch
│   │   └── 02_merge_conflicts.py       # ConflictInfo, edit_resolution, commit_merge
│   ├── branch/
│   │   └── 01_branch_lifecycle.py      # branch, switch, list, delete, create without switch
│   ├── rebase/
│   │   ├── 01_import_commit.py         # import_commit (cherry-pick), ImportResult
│   │   └── 02_rebase.py               # rebase, RebaseResult, replayed_commits
│   ├── gc/
│   │   ├── 01_gc_after_compression.py  # gc(), GCResult, archive retention
│   │   ├── 02_retention_policies.py    # archive_retention_days, conservative vs aggressive
│   │   └── 03_message_reordering.py    # compile(order=), ReorderWarning
│   └── rollback/
│       └── 01_rollback.py              # checkout, reset, compile(at_commit=)
│
├── metadata/                            # Data attached to commits
│   ├── tags/
│   │   ├── 01_classify_and_query.py    # auto-classify, explicit tags, mutable tags, registry, queries
│   │   └── 02_llm_auto_tagger.py      # orchestrator-driven tagging via tools
│   ├── priority/
│   │   ├── 01_pin_skip_reset.py        # annotate(), Priority.PINNED/SKIP/NORMAL
│   │   └── 02_edit_in_place.py         # system(edit=hash), edit-in-place workflow
│   ├── tool_results/
│   │   ├── 01_agentic_loop.py          # set_tools, tool_result, compress_tool_calls
│   │   ├── 02_auto_summarization.py    # configure_tool_summarization, auto-summarize
│   │   ├── 03_offline_tool_management.py # tool management without LLM calls
│   │   └── dummy_file.py              # Search target for agentic loop demo
│   └── reasoning/
│       ├── 01_manual_reasoning.py      # t.reasoning(), ReasoningContent, SKIP default
│       ├── 02_compile_control.py       # compile(include_reasoning=True), annotate overrides
│       ├── 03_formatting.py            # pprint reasoning in dim cyan
│       └── 04_llm_integration.py       # generate() auto-extract, reasoning=False, commit_reasoning=
│
├── config/                              # LLM routing, budgets, generation config
│   ├── 01_per_call_config.py           # LLMConfig on chat/generate, sugar params
│   ├── 02_operation_config.py          # configure_operations, per-op LLMConfig
│   ├── 03_operation_clients.py         # configure_clients, separate LLM clients per op
│   ├── 04_resolution_chain.py          # 4-level chain: sugar > llm_config > operation > default
│   ├── 05_summarize_config.py          # compression-specific LLM config
│   └── 06_budget_guardrail.py          # status() loop, budget check before chat, auto-stop
│
├── queries/                             # Inspecting and auditing history
│   ├── 01_query_api.py                 # find_tool_results, find_tool_calls, find_tool_turns
│   ├── 02_surgical_edits.py            # tool_result(edit=hash), trimming verbose results
│   ├── 03_selective_compression.py     # compress_tool_calls(name=), targeted compression
│   ├── 04_config_provenance.py         # query_by_config, generation_config tracking
│   ├── 05_tool_provenance.py           # set_tools, get_commit_tools, to_openai_params
│   └── 06_edit_history.py              # log(include_edits=True), edit chain tracking
│
├── validation/                          # Retry and validation patterns
│   ├── 01_core_primitive.py            # retry_with_steering, RetryResult, RetryExhaustedError
│   ├── 02_chat_validation.py           # chat(validator=), purify=, provenance_note=, retry_prompt=
│   └── 03_compress_validation.py       # compress(validator=), retain_match= combo
│
├── hooks/                               # Approval layer for operations
│   ├── 01_registration/
│   │   ├── 01_routing.py              # t.on/off, three-tier routing, review=True
│   │   ├── 02_catch_all.py            # Catch-all handlers
│   │   └── 03_recursion_guard.py      # Preventing hook recursion
│   ├── 02_compression/
│   │   ├── 01_lifecycle.py            # PendingCompress lifecycle
│   │   ├── 02_handler_patterns.py     # Common handler patterns
│   │   ├── 03_guidance.py             # GuidanceMixin, two-stage reasoning
│   │   ├── 04_retry_validate.py       # Retry and validate in hooks
│   │   └── 05_two_stage.py            # Two-stage judgment + execution
│   ├── 03_operations/
│   │   ├── 01_gc.py                   # PendingGC, exclude()
│   │   ├── 02_rebase.py              # PendingRebase, replay inspection
│   │   ├── 03_merge_conflicts.py      # PendingMerge, conflict review
│   │   └── 04_merge_retry.py         # Merge retry patterns
│   ├── 04_tool_results/
│   │   ├── 01_basics.py              # PendingToolResult, approve/reject/edit
│   │   ├── 02_edit_and_summarize.py   # edit_result, summarize(instructions=)
│   │   ├── 03_declarative_config.py   # configure_tool_summarization()
│   │   └── 04_custom_routing.py       # Per-tool routing strategies
│   ├── 05_agent_interface/
│   │   ├── 01_to_dict.py             # Pending.to_dict() for serialization
│   │   ├── 02_to_tools.py            # Pending.to_tools() for function calling
│   │   ├── 03_describe_api.py        # Pending.describe_api() human-readable
│   │   └── 04_dispatch.py            # apply_decision() routing
│   ├── 06_token_tolerance/
│   │   ├── 01_basic_gate.py          # Token budget gate
│   │   ├── 02_auto_truncate.py       # Auto-truncate on threshold
│   │   ├── 03_middleware_enforcer.py  # Middleware-style enforcement
│   │   └── 04_dynamic_budget.py      # Dynamic budget adjustment
│   ├── 07_ordering_middleware/
│   │   ├── 01_ordering_basics.py     # Message ordering hooks
│   │   ├── 02_pass_through.py        # Pass-through patterns
│   │   ├── 03_conditional.py         # Conditional ordering
│   │   ├── 04_dynamic_insertion.py   # Dynamic message insertion
│   │   └── 05_full_pipeline.py       # Full ordering pipeline
│   └── 08_dynamic_operations/
│       ├── 01_register_and_fire.py   # Dynamic hook registration
│       ├── 02_introspection.py       # Hook introspection
│       └── 03_review_and_execute.py  # Review-then-execute pattern
│
├── orchestrator/                        # Autonomous agent operations
│   ├── 01_toolkit.py                   # as_tools, profiles, ToolExecutor
│   ├── 02_orchestrator_loop.py         # OrchestratorConfig, assessment loop, HITL
│   └── 03_triggers.py                 # Built-in triggers, autonomy spectrum, hook interception
│
├── multi_agent/                         # Multi-agent coordination
│   ├── 01_parent_child.py             # Child tracts, provenance, parent()
│   ├── 02_delegation.py               # Branch-delegate-merge, compress-and-ingest
│   └── 03_curated_deploy.py           # session.deploy(), curation, merge-back
│
└── e2e/                                 # End-to-end scenarios combining features
    ├── self_correcting_agent.py         # retry + edit + validation + provenance
    ├── long_running_session.py          # triggers + compression + gc + 50+ turns
    ├── ab_testing.py                    # branch + config + diff + provenance query
    ├── context_forensics.py             # log + time-travel + branch + rebase
    ├── research_delegation.py           # multi-agent + compress + merge
    └── autonomous_steering.py           # orchestrator + triggers + hooks + drift
```

---

# Basics

The primitives. No LLM key required for most files. Read these to understand what tract is and how it works internally before adding LLM calls on top.

## 01 — Commit and Compile

**File:** `00_basics/01_commit_and_compile.py`

**Use case:** You want to understand what tract is actually doing under the hood — no shortcuts, no magic.

Open an in-memory tract with `Tract.open()`. Commit a system prompt, a user message, and an assistant response using the content type models `InstructionContent` and `DialogueContent`. Call `compile()` to turn the commit chain into a message list and inspect `CompiledContext.messages`. Use `ctx.pprint()` to display all messages in a rich table with token totals.

> `Tract.open()`, `commit()`, `InstructionContent`, `DialogueContent`, `compile()`, `CompiledContext.messages`, `ctx.pprint()`

## 02 — Shorthand and Format Methods

**File:** `00_basics/02_shorthand_and_format.py`

**Use case:** You know how commit/compile works and want the convenience layer. You also need to format output for a specific LLM provider.

Replace manual content model commits with `system()`, `user()`, and `assistant()` — same result, three fewer imports. Then format the compiled context with `to_dicts()` for a generic list of dicts, `to_openai()` for OpenAI-ready messages, or `to_anthropic()` for Anthropic's format where the system prompt is extracted separately. `str(ctx)` gives a compact one-liner for logging; `ctx.pprint()` gives the full table.

> `system()`, `user()`, `assistant()`, `to_dicts()`, `to_openai()`, `to_anthropic()`, `str(ctx)`, `ctx.pprint()`

## 03 — Status and Token Budget

**File:** `00_basics/03_status_and_budget.py`

**Use case:** You want to know how many tokens are in the context window and how close you are to a limit — without an LLM call.

Part 1 opens a tract without a budget and calls `status()` to see the raw token count and source. Part 2 sets a token budget via `TractConfig(token_budget=TokenBudgetConfig(max_tokens=...))`, adds messages, and watches the percentage fill up. `str(status)` gives a compact one-liner for loop output; `status.pprint()` gives the full panel. The budget is a tracking and guardrail tool — it does not block commits.

> `TractConfig(token_budget=TokenBudgetConfig(max_tokens=))`, `status()`, `token_count`, `token_budget_max`, `str(status)`, `status.pprint()`

## 04 — Log and Diff

**File:** `00_basics/04_log_and_diff.py`

**Use case:** Walk history, compare two states, and reconstruct exactly what the LLM was seeing at any past point.

`log()` returns every commit with hash, role, content type, message, and timestamp. `diff(earlier, later)` shows what changed between two commits. `compile(at_commit=hash)` rebuilds the message list as of any past commit — useful for debugging bad responses. `compile(at_time=datetime)` does the same by timestamp.

> `log()`, `diff(hash_a, hash_b)`, `compile(at_commit=)`, `compile(at_time=)`

## 05 — Batch and Rollback

**File:** `00_basics/05_batch_and_rollback.py`

**Use case:** A RAG retrieval plus user question must land as one atomic unit — partial state is worse than nothing.

Wrap multiple commits in `with t.batch(): ...`. If any commit fails or an exception is raised inside the block, all commits in the batch roll back and the tract state is unchanged. The example simulates a flaky data source that fails on the first call: the batch rolls back cleanly, the retry succeeds, and compile shows the expected message count.

> `with t.batch(): ...`, rollback on exception, clean retry after rollback

## 06 — Chat and Persist

**File:** `00_basics/06_chat_and_persist.py`

**Use case:** A coding assistant that chats, persists to disk, and resumes the next session.

`chat()` does everything in one call: commits the user message, compiles context, calls the LLM, commits the response, and records token usage from the API. Open a tract with a file path and `tract_id`. Close it, reopen from the same path — the full conversation is restored. Walk `log()` to confirm. `response.pprint()` shows the response text, token usage, and config in one panel. `str(status)` gives a compact one-liner for session summary.

> `chat()`, `ChatResponse`, `response.pprint()`, persistence with file path + `tract_id`, `log()`

---

# Operations

History-modifying operations. Each folder groups related operations with progression from manual to automated.

## Compress

### 01 — Manual Compression

**File:** `operations/compress/01_manual.py`

**Use case:** Replace verbose history with your own summary, no LLM needed.

Manual compression with `compress(content="...")` — your text replaces archived commits. PINNED commits survive verbatim. Use `preserve=[hash1, hash2]` for one-shot protection without permanent annotation.

> `compress(content=)`, `compress(preserve=)`, PINNED preservation

### 02 — LLM Compression

**File:** `operations/compress/02_llm_auto.py`

**Use case:** Let the LLM summarize old context to free up token budget.

LLM compression with `compress(target_tokens=200)`. PINNED passes through untouched, SKIP commits are excluded. Guide the summary with `instructions=` or replace the entire prompt with `system_prompt=`.

> `compress(target_tokens=)`, `instructions=`, `system_prompt=`

### 03 — Guided Compression

**File:** `operations/compress/03_guided.py`

**Use case:** Direct the LLM's summarization with IMPORTANT priority and specific guidance.

`annotate(hash, Priority.IMPORTANT)` tells the compressor to be conservative with specific commits. Combine with `instructions=` for domain-specific summarization guidance.

> `Priority.IMPORTANT`, `instructions=`, guided summarization

### 04 — Collaborative Review

**File:** `operations/compress/04_collaborative_review.py`

**Use case:** A human or secondary LLM reviews and edits compression summaries before they commit.

Collaborative review with `auto_commit=False` — returns a `PendingCompress` with the LLM's draft. Inspect with `.summaries`, edit with `.edit_summary(i, text)`, then `.approve()` to finalize.

> `auto_commit=False`, `PendingCompress`, `edit_summary()`, `approve()`

### 05 — IMPORTANT Priority and Retention

**File:** `operations/compress/05_important_and_retain.py`

**Use case:** Some context is too important to lose in compression. Guarantee retention of specific values.

`annotate(hash, Priority.IMPORTANT)` tells the compressor to be conservative. Add fuzzy guidance with `retain="preserve all dollar amounts"`. Add deterministic checks with `retain_match=[r"\$2,847,000"]` in `"regex"` mode — validated against the summary before committing.

> `Priority.IMPORTANT`, `retain=`, `retain_match=`, `retain_match_mode=`, `compress(max_retries=)`

## Merge

### 01 — Merge Strategies

**File:** `operations/merge/01_merge_strategies.py`

**Use case:** Merge branches back together using fast-forward or clean merge.

The two non-conflicting merge modes: **fast-forward** (branch is ahead of main, just advance the pointer) and **clean** (diverged branches with no overlapping edits, auto-merge). Use `no_ff=True` to force a merge commit even on fast-forward. Use `delete_branch=True` to clean up after merge.

> `merge()`, `MergeResult`, `merge_type`, `no_ff`, `delete_branch=True`

### 02 — Merge Conflicts

**File:** `operations/merge/02_merge_conflicts.py`

**Use case:** Two branches both edit the same message. Detect the conflict, resolve it manually, then finalize.

`merge()` returns a `MergeResult`. For conflicts, inspect `result.conflicts` — each `ConflictInfo` shows the target commit and the competing edits. Call `result.edit_resolution(target_hash, "merged content")` to write the resolved text, then `t.commit_merge(result)` to finalize.

> `ConflictInfo`, `edit_resolution()`, `commit_merge()`

## Branch

### 01 — Branch Lifecycle

**File:** `operations/branch/01_branch_lifecycle.py`

**Use case:** Try an experimental approach without affecting main.

`branch("name")` creates a new timeline from current HEAD and switches to it. `switch("main")` moves back. `list_branches()` shows all branches. `branch("name", switch=False)` creates without switching. `delete_branch("name", force=True)` removes unmerged branches.

> `branch()`, `switch()`, `list_branches()`, `current_branch`, `delete_branch(force=True)`

## Rebase

### 01 — Import Commit

**File:** `operations/rebase/01_import_commit.py`

**Use case:** Grab one useful commit from an experiment (cherry-pick).

`import_commit(hash)` copies a single commit onto the current branch with a new hash but the same content. `content_hash` matches but `commit_hash` differs.

> `import_commit(hash)`, `ImportResult`

### 02 — Rebase

**File:** `operations/rebase/02_rebase.py`

**Use case:** Update a stale branch to include the latest from main.

`rebase("main")` replays the current branch's commits on top of main's tip. `RebaseResult` exposes `replayed_commits`, `original_commits`, and `new_head`.

> `rebase("main")`, `RebaseResult`, `replayed_commits`, `new_head`

## GC

### 01 — GC After Compression

**File:** `operations/gc/01_gc_after_compression.py`

**Use case:** Reclaim storage after compression removes history.

Compress, then run `gc(archive_retention_days=0)` to reclaim storage. `GCResult` shows commits removed, blobs removed, and tokens freed. Compiled context is unchanged — GC only touches unreachable data.

> `gc(archive_retention_days=)`, `GCResult`

### 02 — Retention Policies

**File:** `operations/gc/02_retention_policies.py`

**Use case:** Control how long archived data is preserved before GC.

Compare conservative (`archive_retention_days=None`) vs aggressive (`=0`) retention.

> `archive_retention_days` parameter

### 03 — Message Reordering

**File:** `operations/gc/03_message_reordering.py`

**Use case:** Reorder messages for better LLM context flow.

`compile(order=[hash_list])` reorders the compiled context and returns `(CompiledContext, list[ReorderWarning])` with safety checks for structural issues.

> `compile(order=)`, `ReorderWarning`

## Rollback

### 01 — Rollback

**File:** `operations/rollback/01_rollback.py`

**Use case:** Undo recent changes and go back to a known good state.

`checkout(hash)` moves HEAD to a past commit for interactive inspection. `reset(hash)` moves HEAD backward permanently; orphaned commits survive until GC.

> `checkout()`, `reset()`

---

# Metadata

Data attached to commits — tags, priority annotations, tool results, and reasoning traces.

## Tags

### 01 — Classify and Query

**File:** `metadata/tags/01_classify_and_query.py`

**Use case:** Classify commits by what the content *is* and query by tag.

Part 1 (auto-classification): `system()` auto-tags with `"instruction"`, `assistant()` with `"reasoning"`, tool-calling messages with `"tool_call"`. Part 2 (explicit tags): `t.user("...", tags=["hypothesis"])` adds tags at commit time. Part 3 (mutable annotations): `t.tag(hash, "dead_end")` and `t.untag()` for retrospective tagging. Part 4 (tag registry): `register_tag(name, description)` and `list_tags()`. Part 5 (queries): `query_by_tags(match="any"|"all")`, `log(tags=)`.

> `tags=["..."]`, `t.tag()`, `t.untag()`, `get_tags()`, `register_tag()`, `list_tags()`, `query_by_tags()`, `log(tags=)`

### 02 — LLM Auto-Tagger

**File:** `metadata/tags/02_llm_auto_tagger.py`

**Use case:** Let an LLM agent autonomously tag a conversation using the orchestrator.

An orchestrator agent reviews a completed conversation and retrospectively tags each message using tract's built-in tools (register_tag, get_tags, tag). The LLM decides what tags to apply by calling tools autonomously, rather than the developer hardcoding tags at commit time.

> `Orchestrator`, `OrchestratorConfig`, `TAGGER_SYSTEM_PROMPT`, `build_tagger_task_prompt()`

## Priority

### 01 — Pin, Skip, Reset

**File:** `metadata/priority/01_pin_skip_reset.py`

**Use case:** Control what the LLM sees without deleting history.

`system()` commits are `PINNED` by default — they survive compression verbatim. `annotate(hash, Priority.SKIP)` hides a commit from `compile()`. `annotate(hash, Priority.NORMAL)` removes any annotation.

> `annotate(hash, Priority.PINNED/SKIP/NORMAL)`, `Priority` enum

### 02 — Edit in Place

**File:** `metadata/priority/02_edit_in_place.py`

**Use case:** Fix mistakes after the fact without losing the audit trail.

Commit a system prompt with a mistake, then fix with `system(edit=original_hash)`, skip the stale Q&A pair. The LLM sees the corrected version; both versions remain in `log()`.

> `system(edit=hash)`, edit-in-place workflow

## Tool Results

### 01 — Agentic Loop

**File:** `metadata/tool_results/01_agentic_loop.py`

**Use case:** Build an agentic tool-calling loop where the LLM decides which tools to call, every step is committed for provenance, and verbose tool output is compressed afterward.

Define tools in OpenAI function-calling format, register with `set_tools()`, run a compile-call-execute loop. Each tool call is committed as an assistant message with `metadata.tool_calls`, each result via `tool_result()`. After the agent answers, `compress_tool_calls()` collapses verbose intermediate messages into a concise summary.

> `set_tools()`, `tool_result()`, `ToolCall`, `compress_tool_calls()`

### 02 — Auto-Summarization

**File:** `metadata/tool_results/02_auto_summarization.py`

**Use case:** Automatically summarize verbose tool results based on per-tool instructions.

`configure_tool_summarization()` sets up a hook that auto-summarizes tool results based on per-tool instructions and token thresholds.

> `configure_tool_summarization()`, auto-summarize hooks

### 03 — Offline Tool Management

**File:** `metadata/tool_results/03_offline_tool_management.py`

**Use case:** Query and manage tool history, surgical edits to verbose results.

`find_tool_results()`, `find_tool_calls()`, and `find_tool_turns()` inspect tool history. `tool_result(edit=hash)` surgically replaces a verbose result with a trimmed version.

> `find_tool_results()`, `find_tool_calls()`, `find_tool_turns()`, `tool_result(edit=)`

## Reasoning

### 01 — Manual Reasoning

**File:** `metadata/reasoning/01_manual_reasoning.py`

**Use case:** Capture LLM chain-of-thought as first-class commits.

`t.reasoning("Let me think...")` commits a `ReasoningContent` with SKIP priority by default. The reasoning is in `log()` but excluded from `compile()`. Use `format=` to track the extraction source.

> `t.reasoning()`, `ReasoningContent`, `format=`

### 02 — Compile Control

**File:** `metadata/reasoning/02_compile_control.py`

**Use case:** Include or exclude reasoning from compiled context.

`compile(include_reasoning=True)` promotes reasoning from SKIP to NORMAL. Explicit `annotate()` calls always take precedence — PINNED reasoning always appears, explicit SKIP always hides.

> `compile(include_reasoning=True)`, `annotate()` overrides

### 03 — Formatting

**File:** `metadata/reasoning/03_formatting.py`

**Use case:** Display reasoning traces in the terminal.

`pprint()` renders reasoning in dim cyan across all three styles, visually distinct from dialogue.

> `pprint()` rendering for reasoning

### 04 — LLM Integration

**File:** `metadata/reasoning/04_llm_integration.py`

**Use case:** Auto-extract reasoning from provider responses.

`generate()` auto-extracts reasoning from provider responses (Cerebras parsed, OpenAI o1/o3, Anthropic thinking, `<think>` tags) and auto-commits before the assistant response. `generate(reasoning=False)` skips the commit. `Tract.open(commit_reasoning=False)` disables globally.

> `generate()` auto-extract, `reasoning=False`, `commit_reasoning=False`, `ChatResponse.reasoning`

---

# Config

LLM routing, budgets, and generation configuration.

## 01 — Per-Call Config

**File:** `config/01_per_call_config.py`

**Use case:** Override model settings for a single call.

Pass `LLMConfig(temperature=0.9)` or sugar params directly to `chat()` or `generate()` for a one-off override.

> `LLMConfig`, `chat(temperature=)`, `generate(llm_config=)`

## 02 — Operation Config

**File:** `config/02_operation_config.py`

**Use case:** Set different defaults for chat vs compression vs other operations.

Set tract-level defaults with `default_config=LLMConfig(...)` and per-operation overrides with `configure_operations(chat=LLMConfig(...), compress=LLMConfig(...))`.

> `default_config=`, `configure_operations()`, `OperationConfigs`

## 03 — Operation Clients

**File:** `config/03_operation_clients.py`

**Use case:** Route different operations to different LLM providers.

Assign a different LLM client to each operation with `configure_clients(chat=openai_client, compress=ollama_client)`. LLMConfig controls settings, the client controls where requests go.

> `configure_clients()`, per-operation routing

## 04 — Resolution Chain

**File:** `config/04_resolution_chain.py`

**Use case:** Understand which config wins when multiple are set.

Trace the 4-level resolution chain: sugar > llm_config > operation > default. Use `LLMConfig.from_dict()` for cross-framework alias handling (`stop` -> `stop_sequences`, `max_completion_tokens` -> `max_tokens`).

> 4-level resolution chain, `LLMConfig.from_dict()`, alias handling

## 05 — Summarize Config

**File:** `config/05_summarize_config.py`

**Use case:** Configure the LLM used specifically for compression summaries.

Compression-specific LLM configuration and prompt customization.

> Compression LLM config

## 06 — Budget Guardrail

**File:** `config/06_budget_guardrail.py`

**Use case:** A chatbot that checks its token budget before every LLM call and stops when it's running hot.

Check `status()` before each `chat()` call. `chat()` auto-records the API's actual token count. When usage exceeds a threshold, stop and indicate that compression or branching is the next step.

> `status()` in a loop, budget threshold check, `record_usage()`

---

# Queries

Inspecting and auditing history — provenance, tool history, and edit chains.

## 01 — Query API

**File:** `queries/01_query_api.py`

**Use case:** Inspect which tools were called and how many tokens each consumed.

`find_tool_results(name="grep")` returns all grep result commits. `find_tool_calls()` returns assistant commits that requested tool calls. `find_tool_turns()` returns `ToolTurn` objects pairing each tool-call commit with its results.

> `find_tool_results(name=, after=)`, `find_tool_calls(name=)`, `find_tool_turns(name=)`, `ToolTurn`

## 02 — Surgical Edits

**File:** `queries/02_surgical_edits.py`

**Use case:** Replace verbose tool results with trimmed versions.

Walk verbose tool results with `find_tool_results()`, replace each with a trimmed version via `tool_result(edit=hash)`. The original is preserved in history.

> `tool_result(edit=)`, surgical replacement

## 03 — Selective Compression

**File:** `queries/03_selective_compression.py`

**Use case:** Compress only specific tool turns, leave others untouched.

`compress_tool_calls(name="grep")` compresses only grep tool turns while leaving read_file and bash results untouched.

> `compress_tool_calls(name=)`, targeted compression

## 04 — Config Provenance

**File:** `queries/04_config_provenance.py`

**Use case:** "Which model produced this output? What temperature was used?"

Every assistant commit stores the fully-resolved `generation_config`. Query with `query_by_config()` — single-field, multi-field AND, or whole-config matching. The IN operator handles multi-value queries.

> `query_by_config(model=, temperature=)`, `generation_config`

## 05 — Tool Provenance

**File:** `queries/05_tool_provenance.py`

**Use case:** "What tools were available when this response was generated?"

`set_tools([...])` registers tool schemas that auto-link to subsequent commits. `get_commit_tools(hash)` reconstructs exactly what tools a commit had. `to_openai_params()` and `to_anthropic_params()` return full API-ready dicts including tools.

> `set_tools()`, `get_commit_tools()`, `to_openai_params()`, `to_anthropic_params()`

## 06 — Edit History

**File:** `queries/06_edit_history.py`

**Use case:** "How did this message get to its current state?"

`log(include_edits=True)` reveals the full chain of edits, showing original and replacement commits side by side.

> `log(include_edits=True)`, edit chain

---

# Validation

Retry and validation patterns for LLM output.

## 01 — Core Primitive

**File:** `validation/01_core_primitive.py`

**Use case:** Validate LLM output and retry with steering when it fails.

`retry_with_steering()` takes `attempt` (produces a result), `validate` (checks it), `steer` (injects a correction), and `head_fn`/`reset_fn` for history management. `RetryExhaustedError` carries `last_result` for fallback recovery.

> `retry_with_steering()`, `RetryResult`, `RetryExhaustedError`

## 02 — Chat Validation

**File:** `validation/02_chat_validation.py`

**Use case:** Validate chat responses inline with retry.

`chat(validator=my_validator, max_retries=3)` wraps the LLM call. On failure, a steering message is committed. `purify=True` resets HEAD and re-commits only the clean result. `provenance_note=True` records retry count.

> `chat(validator=, max_retries=, purify=, provenance_note=, retry_prompt=)`

## 03 — Compress Validation

**File:** `validation/03_compress_validation.py`

**Use case:** Validate compression summaries before committing.

`compress(validator=, max_retries=)` validates summaries after LLM generation. Combine with `retain_match=` for a two-layer safety net: deterministic regex patterns + semantic validator.

> `compress(validator=, max_retries=)`, `retain_match=` combo

---

# Hooks

The approval layer for operations. Every `Pending` subclass follows the same `approve()`/`reject()` protocol. The hook system has three routing tiers: auto-execute for low-risk actions, call the handler for review, fall through to the default for unhandled cases.

## 01 — Registration

**Files:** `hooks/01_registration/`

Hook registration, routing tiers, catch-all handlers, and recursion guards.

> `t.on()`, `t.off()`, `Pending` base class, three-tier routing, `review=True`, recursion guard

## 02 — Compression Hooks

**Files:** `hooks/02_compression/`

`PendingCompress` lifecycle, handler patterns, `GuidanceMixin` for two-stage reasoning, retry/validate in hooks, and two-stage judgment + execution.

> `PendingCompress`, `edit_summary()`, `approve()`, `reject()`, `GuidanceMixin`, `judge()`, `Judgment`

## 03 — Operation Hooks

**Files:** `hooks/03_operations/`

`PendingGC` (with `exclude()` to protect commits), `PendingRebase` (replay inspection), `PendingMerge` (conflict review), and merge retry patterns.

> `PendingGC`, `PendingRebase`, `PendingMerge`, `exclude()`, `approve()`, `reject()`

## 04 — Tool Result Hooks

**Files:** `hooks/04_tool_results/`

`PendingToolResult` basics (approve/reject/edit), `edit_result()` and `summarize(instructions=)`, declarative config via `configure_tool_summarization()`, and per-tool custom routing.

> `PendingToolResult`, `edit_result()`, `summarize()`, `configure_tool_summarization()`

## 05 — Agent Interface

**Files:** `hooks/05_agent_interface/`

Every `Pending` subclass auto-generates an agent-facing interface: `to_dict()` for serialization, `to_tools()` for function calling, `describe_api()` for human-readable docs, and `apply_decision()` for routing agent decisions.

> `to_dict()`, `to_tools()`, `describe_api()`, `apply_decision()`

## 06 — Token Tolerance

**Files:** `hooks/06_token_tolerance/`

Token budget enforcement via hooks: basic gate, auto-truncate on threshold, middleware-style enforcement, and dynamic budget adjustment.

> Token budget hooks, auto-truncate, dynamic budget

## 07 — Ordering Middleware

**Files:** `hooks/07_ordering_middleware/`

Message ordering hooks: basics, pass-through patterns, conditional ordering, dynamic message insertion, and full pipeline composition.

> `compile(order=)` hooks, middleware pipeline

## 08 — Dynamic Operations

**Files:** `hooks/08_dynamic_operations/`

Dynamic hook registration and firing, introspection of registered hooks, and review-then-execute patterns.

> Dynamic `t.on()`, hook introspection, review-and-execute

---

# Orchestrator

Autonomous agent operations — the LLM manages its own context window using tract's toolkit.

## 01 — Toolkit and Profiles

**File:** `orchestrator/01_toolkit.py`

**Use case:** Let the LLM decide when to compress, branch, annotate, or query status via function calling.

`t.as_tools(format="openai", profile="self")` returns tract operations as LLM tool schemas. Three profiles control scope: **self** (full CRUD), **supervisor** (read + high-level ops), **observer** (read-only). `ToolExecutor` dispatches tool call results.

> `as_tools(format=, profile=)`, `ToolExecutor`, profiles: `"self"`, `"supervisor"`, `"observer"`

## 02 — Orchestrator Loop

**File:** `orchestrator/02_orchestrator_loop.py`

**Use case:** Auto-assess context health and execute maintenance operations autonomously.

Configure triggers on `OrchestratorConfig`: `on_commit_count=20`, `on_token_threshold=0.7`. The orchestrator runs assessment, selects tools, and executes in a loop. Attach hooks for human-in-the-loop.

> `OrchestratorConfig`, `TriggerConfig`, assessment loop, HITL via hooks

## 03 — Triggers

**File:** `orchestrator/03_triggers.py`

**Use case:** Make an agent self-managing with automatic operations and hooks for override.

7 built-in triggers: `CompressTrigger`, `PinTrigger`, `RebaseTrigger`, `GCTrigger`, `MergeTrigger`, `BranchTrigger`, `ArchiveTrigger`. Hook interception via `t.on("trigger", handler)`. Autonomy spectrum from fully autonomous to collaborative.

> `CompressTrigger`, `PinTrigger`, `configure_triggers()`, `PendingTrigger`, autonomy spectrum

---

# Multi-Agent

Multi-agent coordination — parent-child tracts, delegation, and curated deployment.

## 01 — Parent-Child Relationship

**File:** `multi_agent/01_parent_child.py`

**Use case:** Spawn a sub-agent with its own isolated context, preserving lineage for provenance.

Create a child tract linked to the parent. The child has independent history and branch structure. `t.children()` lists all children; `t.parent()` accesses the parent. Commits carry parent `tract_id` in provenance metadata.

> `parent()`, `children()`, parent-child provenance

## 02 — Sub-Agent Delegation

**File:** `multi_agent/02_delegation.py`

**Use case:** Spawn a research sub-agent, let it work for 40 turns, then ingest only the summary into the parent.

The sub-agent works in a child tract. When finished, compress into a summary. The parent calls `import_commit()` to pull the summary in. Forty turns collapse into one commit on the parent.

> `compress()` summary, `import_commit()` across tracts, compress-and-ingest pattern

## 03 — Curated Deploy

**File:** `multi_agent/03_curated_deploy.py`

**Use case:** Deploy a sub-agent on a purpose-built branch with filtered context, then merge back.

`session.deploy()` creates a branch from parent HEAD. `curate={"keep_tags": [...]}` filters context. `curate={"drop": [hash], "compact_before": hash}` for targeted cleanup. Merge-back via `parent.merge()` or `session.collapse()`.

> `session.deploy()`, `curate=`, merge-back, collapse

---

# E2E

End-to-end scenarios combining features from across the cookbook. Each scenario is a realistic application.

## self_correcting_agent.py

**Combines:** validation (retry) + metadata/priority (edit + annotations) + operations/compress + queries/provenance

An agent that validates its own JSON output via `chat(validator=json_validator, purify=True)`, retries on failure with a steering message in context, and uses `provenance_note=True` to record retry counts. Critical decisions are annotated `IMPORTANT` with `retain_match=` patterns so they survive compression verbatim.

## long_running_session.py

**Combines:** operations/compress + orchestrator/triggers + operations/gc + 00_basics/chat

A session that runs for 50+ turns unattended. `CompressTrigger(threshold=0.8)` fires automatically when the budget fills up. PINNED alerts survive every compression cycle. `gc(archive_retention_days=30)` reclaims storage while preserving a month of audit history.

## ab_testing.py

**Combines:** operations/branch + config + 00_basics/log_and_diff + queries/provenance

Branch from the same conversation state, run identical prompts on each branch with different model configs, then `diff()` the results and `query_by_config(model=)` to compare.

## context_forensics.py

**Combines:** 00_basics/log_and_diff + operations/branch + operations/rebase

Walk `log()` to find the commit where bad data entered. `compile(at_commit=hash)` reconstructs what the LLM saw. Branch from before contamination, cherry-pick good work, rebase onto main.

## research_delegation.py

**Combines:** multi_agent + operations/compress + operations/merge

Three sub-agents research in parallel via `session.deploy()` with tag-filtered context. Compress each sub-agent's history into a summary. Merge summaries onto main, resolving conflicts.

## autonomous_steering.py

**Combines:** orchestrator + orchestrator/triggers + hooks/compression + operations/compress

All built-in triggers active at `autonomy="autonomous"`. The orchestrator fires on commit count and token threshold, running assessment and maintenance in a loop. One hook on `"compress"` for human sign-off on large compressions.
