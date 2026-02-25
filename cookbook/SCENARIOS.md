# Tract Cookbook — Scenarios

The cookbook is organized as a three-tier progression. **Fundamentals** build the mental model: no LLM required, no assumptions about your use case. **Patterns** show real workflows with live LLM calls — each one is a self-contained scenario you can run and adapt. **Advanced** covers power features: hooks, policies, the orchestrator, and multi-agent coordination. **Compositions** combine features from all three tiers into end-to-end real-world applications.

Work through the tiers in order if you're new. Jump to any individual file if you know what you need — every file is standalone.

## File Tree

```
cookbook/
├── SCENARIOS.md
├── fundamentals/                     # Tier 1: Core mental model + basic operations
│   ├── 01_commit_and_compile.py      # Tract.open, commit(), compile(), CompiledContext
│   ├── 02_shorthand_and_format.py    # system/user/assistant, to_openai/anthropic/dicts
│   ├── 03_status_and_budget.py       # status(), TractConfig, token budget tracking
│   ├── 04_annotations.py            # pin, skip, important, edit-in-place, annotate()
│   ├── 05_log_diff_time_travel.py   # log, show, diff, checkout, reset, compile(at_commit=)
│   ├── 06_batch_and_rollback.py     # batch() context manager, atomic operations
│   └── 07_branching.py              # branch, switch, delete, list, FF merge, clean merge
│
├── patterns/                         # Tier 2: Real workflows with LLM
│   ├── 01_chat_and_persist.py       # chat(), ChatResponse, persistence, session resume
│   ├── 02_budget_guardrail.py       # status loop, budget check before chat, auto-stop
│   ├── 03_llm_config.py            # per-call config, defaults, per-op configs/clients, resolution chain
│   ├── 04_compression.py           # manual, LLM, target_tokens, guided, collaborative review
│   ├── 05_important_and_retain.py   # IMPORTANT priority + retain_match through compression
│   ├── 06_merge_conflicts.py        # conflict detection, manual resolution, commit_merge
│   ├── 07_rebase_and_import.py      # import_commit (cherry-pick), rebase
│   ├── 08_gc_and_reorder.py         # gc, archive retention, compile(order=), hook review
│   ├── 09_provenance.py            # config provenance, tool provenance, edit history
│   └── 10_retry_and_validation.py   # chat(validator=), retry_with_steering, purify
│
├── advanced/                         # Tier 3: Power features
│   ├── hooks/
│   │   ├── 01_concept.py            # t.on/off, Pending, three-tier routing, review=True
│   │   ├── 02_compress_review.py    # PendingCompress, edit_summary, approve/reject
│   │   ├── 03_gc_rebase_merge.py    # PendingGC, PendingRebase, PendingMerge, exclude()
│   │   ├── 04_guidance.py           # GuidanceMixin, two-stage reasoning, edit_guidance
│   │   └── 05_agent_interface.py    # to_dict, to_tools, describe_api, apply_decision
│   ├── policies/
│   │   ├── 01_builtin.py            # CompressPolicy, PinPolicy, BranchPolicy, ArchivePolicy
│   │   ├── 02_custom.py             # Policy protocol, evaluate(), custom triggers
│   │   └── 03_autonomy.py           # manual/collaborative/autonomous, PendingPolicy
│   ├── orchestrator/
│   │   ├── 01_toolkit.py            # as_tools, profiles, ToolExecutor
│   │   └── 02_orchestrator_loop.py  # triggers, assessment, HITL via hooks
│   └── multi_agent/
│       ├── 01_parent_child.py       # child tracts, provenance
│       └── 02_delegation.py         # branch-delegate-merge, compress-and-ingest
│
└── compositions/                     # Real-world scenarios combining features
    ├── self_correcting_agent.py      # retry + edit + validation + provenance
    ├── long_running_session.py       # policies + compression + gc + 50+ turns
    ├── ab_testing.py                 # branch + config + diff + provenance query
    ├── context_forensics.py          # log + time-travel + branch + rebase
    ├── research_delegation.py        # multi-agent + compress + merge
    └── autonomous_steering.py        # orchestrator + policies + hooks + drift
```

---

# Tier 1: Fundamentals

The primitives. No LLM key required for most files. Read these to understand what tract is and how it works internally before adding LLM calls on top.

## 01 — Commit and Compile

**Use case:** You want to understand what tract is actually doing under the hood — no shortcuts, no magic.

Open an in-memory tract with `Tract.open()`. Commit a system prompt, a user message, and an assistant response using the content type models `InstructionContent` and `DialogueContent`. Call `compile()` to turn the commit chain into a message list and inspect `CompiledContext.messages`. Use `ctx.pprint()` to display all messages in a rich table with token totals.

> `Tract.open()`, `commit()`, `InstructionContent`, `DialogueContent`, `compile()`, `CompiledContext.messages`, `ctx.pprint()`

## 02 — Shorthand and Format Methods

**Use case:** You know how commit/compile works and want the convenience layer. You also need to format output for a specific LLM provider.

Replace manual content model commits with `system()`, `user()`, and `assistant()` — same result, three fewer imports. Then format the compiled context with `to_dicts()` for a generic list of dicts, `to_openai()` for OpenAI-ready messages, or `to_anthropic()` for Anthropic's format where the system prompt is extracted separately. `str(ctx)` gives a compact one-liner for logging; `ctx.pprint()` gives the full table.

> `system()`, `user()`, `assistant()`, `to_dicts()`, `to_openai()`, `to_anthropic()`, `str(ctx)`, `ctx.pprint()`

## 03 — Status and Token Budget

**Use case:** You want to know how many tokens are in the context window and how close you are to a limit — without an LLM call.

Part 1 opens a tract without a budget and calls `status()` to see the raw token count and source. Part 2 sets a token budget via `TractConfig(token_budget=TokenBudgetConfig(max_tokens=...))`, adds messages, and watches the percentage fill up. `str(status)` gives a compact one-liner for loop output; `status.pprint()` gives the full panel. The budget is a tracking and guardrail tool — it does not block commits.

> `TractConfig(token_budget=TokenBudgetConfig(max_tokens=))`, `status()`, `token_count`, `token_budget_max`, `str(status)`, `status.pprint()`

## 04 — Annotations and Edit in Place

**Use case:** Control what the LLM sees without deleting history. Fix mistakes after the fact without losing the audit trail.

Part 1 covers priority annotations without an LLM. `system()` commits are `PINNED` by default — they survive compression verbatim. `annotate(hash, Priority.SKIP)` hides a commit from `compile()` while keeping it in history. `annotate(hash, Priority.NORMAL)` removes any annotation. Part 2 uses an LLM to show edit-in-place: commit a system prompt with a mistake, chat with the LLM (it parrots the error), then fix the system prompt with `system(edit=original_hash)`, skip the stale Q&A pair, and chat again — the LLM sees the corrected version. Both versions remain in `log()`.

> `annotate(hash, Priority.PINNED/SKIP/NORMAL)`, `system(edit=hash)`, `Priority` enum, `log()`, compile reflects annotations

## 05 — Log, Diff, and Time Travel

**Use case:** Walk history, compare two states, and reconstruct exactly what the LLM was seeing at any past point.

`log()` returns every commit with hash, role, content type, message, and timestamp. `diff(earlier, later)` shows what changed between two commits. `compile(at_commit=hash)` rebuilds the message list as of any past commit — useful for debugging bad responses. `compile(at_time=datetime)` does the same by timestamp. `checkout(hash)` moves HEAD to a past commit for interactive inspection. `reset(hash)` moves HEAD backward permanently; orphaned commits survive until GC.

> `log()`, `diff(hash_a, hash_b)`, `compile(at_commit=)`, `compile(at_time=)`, `checkout()`, `reset()`

## 06 — Batch and Rollback

**Use case:** A RAG retrieval plus user question must land as one atomic unit — partial state is worse than nothing.

Wrap multiple commits in `with t.batch(): ...`. If any commit fails or an exception is raised inside the block, all commits in the batch roll back and the tract state is unchanged. The example simulates a flaky data source that fails on the first call: the batch rolls back cleanly, the retry succeeds, and compile shows the expected message count.

> `with t.batch(): ...`, rollback on exception, clean retry after rollback

## 07 — Branching

**Use case:** Try an experimental approach without affecting main. Merge back when done.

`branch("name")` creates a new timeline from current HEAD and switches to it. `switch("main")` moves back. `list_branches()` shows all branches with a `*` on the current one. `branch("name", switch=False)` creates without switching. `delete_branch("name", force=True)` removes unmerged branches. The file also covers the two non-conflicting merge modes: **fast-forward** (branch is ahead of main, just advance the pointer) and **clean** (diverged branches with no overlapping edits, auto-merge). Conflict resolution is in `patterns/06_merge_conflicts.py`.

> `branch()`, `switch()`, `list_branches()`, `current_branch`, `branch(switch=False)`, `delete_branch(force=True)`, `merge()`, fast-forward, clean merge

---

# Tier 2: Patterns

Real workflows with live LLM calls. Each file is a self-contained scenario. Files marked as implemented exist on disk; files not yet written are described by what they will teach.

## 01 — Chat and Persist

**Use case:** A coding assistant that chats, persists to disk, and resumes the next session.

`chat()` does everything in one call: commits the user message, compiles context, calls the LLM, commits the response, and records token usage from the API. Open a tract with a file path and `tract_id`. Close it, reopen from the same path — the full conversation is restored. Walk `log()` to confirm. `response.pprint()` shows the response text, token usage, and config in one panel. `str(status)` gives a compact one-liner for session summary.

> `chat()`, `ChatResponse`, `response.pprint()`, persistence with file path + `tract_id`, `log()`

## 02 — Budget Guardrail Loop

**Use case:** A chatbot that checks its token budget before every LLM call and stops when it's running hot.

In a multi-turn loop, check `status()` before each `chat()` call. `chat()` automatically records the API's actual token count, so the budget tracks real usage — not just tiktoken estimates. When usage exceeds a threshold (e.g., 90%), stop and indicate that compression or branching is the next step. `str(status)` is used in the loop for compact per-turn output; `status.pprint()` gives the final summary panel.

> `status()` in a loop, `chat()` auto-records usage, `record_usage()` for manual calls, `str(status)`, `status.pprint()`

## 03 — LLM Config

**Use case:** Control which model, which settings, and which client — at every granularity from per-call overrides down to cross-framework alias handling.

Part 1: pass `LLMConfig(temperature=0.9)` or sugar params directly to `chat()` or `generate()` for a one-off override. Part 2: set tract-level defaults with `default_config=LLMConfig(...)` and per-operation overrides with `configure_operations(chat=LLMConfig(...), compress=LLMConfig(...))`. Part 3: assign a different LLM client to each operation with `configure_clients(chat=openai_client, compress=ollama_client)` — LLMConfig controls settings, the client controls where requests go. Part 4: trace the 4-level resolution chain (sugar > llm_config > operation > default) and use `LLMConfig.from_dict()` for cross-framework alias handling (`stop` -> `stop_sequences`, `max_completion_tokens` -> `max_tokens`).

> `LLMConfig`, `chat(temperature=)`, `generate(llm_config=)`, `default_config=`, `configure_operations()`, `configure_clients()`, `OperationConfigs`, `LLMConfig.from_dict()`

## 04 — Compression

**Use case:** Keep conversations alive past the context window limit using manual, LLM-driven, and collaborative compression modes.

Part 1: manual compression with `compress(content="...")` — your text replaces archived commits, no LLM needed. PINNED commits survive verbatim. Use `preserve=[hash1, hash2]` for one-shot protection without permanent annotation. Part 2: LLM compression with `compress(target_tokens=200)`. PINNED passes through untouched, SKIP commits are excluded. Guide the summary with `instructions=` or replace the entire prompt with `system_prompt=`. Part 3: collaborative review with `auto_commit=False` — returns a `PendingCompress` with the LLM's draft. Inspect with `.summaries`, edit with `.edit_summary(i, text)`, then `.approve()` to finalize.

> `compress(content=)`, `compress(target_tokens=)`, `compress(preserve=)`, `instructions=`, `auto_commit=False`, `PendingCompress`, `edit_summary()`, `approve()`, `CompressResult`

## 05 — IMPORTANT Priority and Retention Criteria

**Use case:** Some context is too important to lose in compression. Load a real contract, discuss it over several turns, then compress with guaranteed retention of specific dollar amounts, dates, and penalty clauses.

`annotate(hash, Priority.IMPORTANT)` tells the compressor to be conservative. Add fuzzy guidance with `retain="preserve all dollar amounts and deadlines"` — injected into the summarization prompt. Add deterministic pattern checks with `retain_match=[r"\$2,847,000", r"Net[\s\-]*45"]` in `"regex"` mode — validated against the summary output before committing. After compression, verify the LLM still knows the key terms by calling `chat()` again.

> `Priority.IMPORTANT`, `annotate(hash, IMPORTANT, retain=, retain_match=, retain_match_mode=)`, `Priority.SKIP`, `compress(max_retries=)`, post-compression verification

## 06 — Merge Conflicts

**Use case:** Two branches both edit the same message. Detect the conflict, resolve it manually, then finalize.

`merge()` returns a `MergeResult` with `merge_type` indicating what happened. For conflicts, inspect `result.conflicts` — each `ConflictInfo` shows the target commit and the competing edits from each branch. Call `result.edit_resolution(target_hash, "merged content")` to write the resolved text, then `t.commit_merge(result)` to finalize. Use `no_ff=True` to force a merge commit even on fast-forward cases. Use `delete_branch=True` to clean up after merge.

> `merge()`, `MergeResult`, `merge_type`, `ConflictInfo`, `edit_resolution()`, `commit_merge()`, `no_ff`, `delete_branch=True`

## 07 — Rebase and Import Commit

**Use case:** Grab one useful commit from an experiment (cherry-pick), and update a stale branch to include the latest from main (rebase).

Part 1: `import_commit(hash)` copies a single commit onto the current branch with a new hash but the same content — Tract's cherry-pick. Shows that `content_hash` matches but `commit_hash` differs. Part 2: `rebase("main")` replays the current branch's commits on top of main's tip. The result has new hashes (new parentage) but the same content. `RebaseResult` exposes `replayed_commits`, `original_commits`, and `new_head`. `ctx.pprint(style="chat")` renders the conversation view.

> `import_commit(hash)`, `ImportResult`, `rebase("main")`, `RebaseResult`, `replayed_commits`, `original_commits`, `new_head`, `pprint(style=)`

## 08 — GC and Reorder

**Use case:** Reclaim storage after compression, control archive retention policy, and reorder messages for better LLM context flow.

Part 1: compress a conversation using collaborative review (`review=True` returns a `PendingCompress`), approve the result, then run `gc(archive_retention_days=0)` to reclaim storage. `GCResult` shows commits removed, blobs removed, and tokens freed. Compiled context is unchanged — GC only touches unreachable data. Part 2: compare conservative (`archive_retention_days=None`) vs aggressive (`=0`) retention. Part 3: `compile(order=[hash_list])` reorders the compiled context by commit hash and returns `(CompiledContext, list[ReorderWarning])` with safety checks for structural issues like edits appearing before their targets.

> `compress(review=True)`, `PendingCompress`, `approve()`, `gc(archive_retention_days=)`, `GCResult`, `compile(order=)`, `ReorderWarning`

## 09 — Provenance

**Use case:** "Which model produced this output? What temperature was used? What tools were available? How did this message get to its current state?"

Part 1 (config provenance): every assistant commit stores the fully-resolved `generation_config`. Query with `query_by_config()` — single-field, multi-field AND, or whole-config matching. The IN operator handles multi-value queries. Part 2 (tool provenance): `set_tools([...])` registers tool schemas that auto-link to subsequent commits. Each unique schema is content-hashed and stored once. `get_commit_tools(hash)` reconstructs exactly what tools a commit had. `to_openai_params()` and `to_anthropic_params()` return full API-ready dicts including tools. Part 3 (edit history): `log(include_edits=True)` reveals the full chain of edits, showing original and replacement commits side by side.

> `query_by_config(model=, temperature=)`, `set_tools()`, `get_commit_tools()`, `to_openai_params()`, `to_anthropic_params()`, `log(include_edits=True)`

## 10 — Retry and Validation

**Use case:** Validate LLM output and retry with steering when it fails. Works for chat, compression summaries, merge resolutions, or any callable.

Part 1 (core primitive): `retry_with_steering()` takes `attempt` (produces a result), `validate` (checks it), `steer` (injects a correction), and optional `head_fn`/`reset_fn` for history management. Loops until validation passes or retries are exhausted. Returns `RetryResult` with the value, attempt count, and failure history. Raises `RetryExhaustedError` on exhaustion. No Tract dependency — pure logic and callbacks. Part 2: `chat(validator=my_validator, max_retries=3)` wraps the LLM call. On failure, a steering message is committed as a user message — the LLM sees its mistake in context. `purify=True` resets HEAD and re-commits only the clean result. Part 3: wrap arbitrary operations (compression, merge resolution) with the same `retry_with_steering()` primitive by providing different closures.

> `retry_with_steering()`, `RetryResult`, `RetryExhaustedError`, `chat(validator=, max_retries=, purify=, provenance_note=)`

---

# Tier 3: Advanced

Power features for production systems. These files are stubs describing what each file will teach.

## Hooks

### hooks/01 — Hook Concepts

**Use case:** Understand how tract's hook system works before using it in a real scenario.

`t.on("compress", handler)` registers a handler that fires when a compress operation completes. `t.off("compress")` removes it. The hook system has three routing tiers: auto-execute for low-risk actions, call the handler for review, fall through to the default for unhandled cases. `review=True` on any operation forces it into the review tier. The `Pending` base class carries the operation's proposal; each subclass adds operation-specific fields and methods.

> `t.on()`, `t.off()`, `Pending` base class, three-tier routing, `review=True`

### hooks/02 — Compress Review

**Use case:** A human or secondary LLM reviews and edits every compression summary before it commits.

Register a hook on `"compress"`. The hook receives a `PendingCompress` with draft summaries, the source commit list, and token estimates. Inspect the summaries, call `edit_summary(index, new_text)` to refine them, then call `approve()` to finalize. Call `reject(reason=)` to abort. The hook can also modify `target_tokens` before deciding.

> `PendingCompress`, `summaries`, `source_commits`, `edit_summary()`, `approve()`, `reject()`

### hooks/03 — GC, Rebase, and Merge Hooks

**Use case:** Require explicit approval before GC removes data, before a rebase rewrites history, or before a merge commits.

`PendingGC` carries the list of commits scheduled for removal — call `exclude(hash)` to protect specific commits before approving. `PendingRebase` carries the replay plan and allows inspecting `original_commits` vs `replayed_commits` before approving. `PendingMerge` carries the merge result with conflict resolutions baked in. All three follow the same `approve()`/`reject()` protocol.

> `PendingGC`, `PendingRebase`, `PendingMerge`, `exclude()`, `approve()`, `reject()`

### hooks/04 — Guidance Mixin

**Use case:** A hook that reasons about a proposal before deciding — two-stage: judgment first, then execution.

`GuidanceMixin` adds a structured judgment phase to any `Pending` subclass. Call `pending.judge(llm_call)` to produce a `Judgment` with a recommendation and rationale. Call `edit_guidance(new_rationale)` to refine the reasoning. Then execute based on the judgment. The mixin is designed for hooks where the decision itself needs to be audited or logged.

> `GuidanceMixin`, `judge()`, `Judgment`, `edit_guidance()`, two-stage reasoning pattern

### hooks/05 — Agent Interface

**Use case:** Expose a hook's proposal to an LLM agent so the agent can decide whether to approve, reject, or modify it.

Every `Pending` subclass auto-generates an agent-facing interface: `to_dict()` returns a JSON-serializable description of the proposal, `to_tools()` returns tool schemas for function calling, and `describe_api()` returns a human-readable API description. `apply_decision(tool_result)` routes the agent's tool call back to the right method. Use this to build LLM-driven review loops without writing custom parsing logic.

> `to_dict()`, `to_tools()`, `describe_api()`, `apply_decision()`

---

## Policies

### policies/01 — Built-In Policies

**Use case:** Auto-compress at 80% budget, auto-pin system prompts, detect topic drift, and archive stale branches — with no custom code.

Four built-in policies cover common needs. `CompressPolicy(threshold=0.8)` triggers compression when token usage crosses the threshold. `PinPolicy()` automatically pins system prompts and other instruction-role commits. `BranchPolicy()` detects topic drift and suggests branching. `ArchivePolicy(inactive_days=7)` archives branches that have not been active for the configured period. Register one or all with `configure_policies([...])`.

> `CompressPolicy`, `PinPolicy`, `BranchPolicy`, `ArchivePolicy`, `configure_policies()`

### policies/02 — Custom Policy

**Use case:** Auto-skip any commit containing PII, or auto-skip tool outputs older than N turns.

Implement the `Policy` protocol: define a `name`, `trigger` (event type that fires evaluation), and `evaluate(tract) -> PolicyAction | None`. Your policy inspects the current commit log and proposes annotations or operations. The evaluator executes the proposal immediately or routes it through the hook system depending on autonomy mode. Policies are composable — multiple policies can fire on the same event.

> `Policy` protocol, `name`, `trigger`, `evaluate()`, `PolicyAction`

### policies/03 — Autonomy Spectrum

**Use case:** In development, approve everything manually. In staging, auto-approve safe operations. In production, full autonomy.

Same policies, three modes configured at `configure_policies(..., autonomy=)`. **Manual**: every policy action produces a `PendingPolicy` that waits for explicit approval. **Collaborative**: low-risk actions (pin, skip, annotate) execute immediately; high-risk actions (compress, branch, GC) route through hooks for approval. **Autonomous**: all actions execute immediately without review. The autonomy level does not change the policy logic — only the execution path.

> `autonomy="manual"`, `autonomy="collaborative"`, `autonomy="autonomous"`, `PendingPolicy`

---

## Orchestrator

### orchestrator/01 — Toolkit and Profiles

**Use case:** Let the LLM decide when to compress, branch, annotate, or query status via function calling.

`t.as_tools(format="openai", profile="self")` returns tract operations as LLM tool schemas. Three profiles control scope: **self** gives the agent full CRUD over its own context (compress, GC, annotate, branch, rebase); **supervisor** gives read access plus high-level operations (compress, branch, archive); **observer** is read-only (status, log, diff, compile). `ToolExecutor` dispatches tool call results to the right operation. Use `as_tools(format="anthropic")` for Anthropic's tool format.

> `as_tools(format=, profile=)`, `ToolExecutor`, profiles: `"self"`, `"supervisor"`, `"observer"`

### orchestrator/02 — Orchestrator Loop

**Use case:** Auto-assess context health on a schedule and execute maintenance operations autonomously, with optional human sign-off via hooks.

Configure triggers on `OrchestratorConfig`: `on_commit_count=20` fires every 20 commits, `on_token_threshold=0.7` fires at 70% token usage. When a trigger fires, the orchestrator runs an assessment (fragmentation, budget pressure, stale branches), selects tools from the active profile, and executes in a loop. Attach hooks for human-in-the-loop: the orchestrator proposes actions, the hook approves or rejects, and the orchestrator executes the approved set.

> `OrchestratorConfig`, `TriggerConfig(on_commit_count=, on_token_threshold=)`, assessment loop, HITL via hooks

---

## Multi-Agent

### multi_agent/01 — Parent-Child Relationship

**Use case:** A main agent needs to spawn a sub-agent with its own isolated context, while preserving the lineage for provenance.

Create a child tract linked to the parent tract. The child has independent commit history, its own branch structure, and a separate SQLite file. The parent-child relationship is tracked — `t.children()` lists all child tracts; child tracts can access `t.parent()`. Commits in the child carry the parent `tract_id` in their provenance metadata.

> `parent()`, `children()`, parent-child provenance, isolated child history

### multi_agent/02 — Sub-Agent Delegation

**Use case:** Spawn a research sub-agent, let it work for 40 turns, then ingest only the summary into the parent.

The sub-agent works in its own child tract. When finished, compress its history into a summary commit. The parent calls `import_commit()` on the summary commit, pulling it from the child tract into the parent's timeline as a single message. Forty turns collapse into one commit on the parent. The full research history remains available in the child tract for audit.

> child tract workflow, `compress()` summary, `import_commit()` across tracts, compress-and-ingest pattern

---

# Compositions

Real-world scenarios that combine features across multiple tiers. Each composition references the features it builds on.

## self_correcting_agent.py

**Combines:** patterns/10 (retry + validation) + fundamentals/04 (edit + annotations) + patterns/04 (compression) + patterns/09 (provenance)

An agent that validates its own JSON output via `chat(validator=json_validator, purify=True)`, retries on failure with a steering message in context, and uses `provenance_note=True` to record retry counts. Critical decisions are annotated `IMPORTANT` with `retain_match=` patterns so they survive compression verbatim. `get_commit_tools()` and `query_by_config()` reconstruct exactly what the agent had available at each step.

## long_running_session.py

**Combines:** patterns/04 (compression) + advanced/policies (auto-compress) + patterns/08 (gc) + patterns/01 (chat loop)

A session that runs for 50+ turns unattended. `CompressPolicy(threshold=0.8)` fires automatically when the budget fills up. PINNED alerts and critical context survive every compression cycle. `gc(archive_retention_days=30)` reclaims storage while preserving a month of audit history. The session ends with `status.pprint()` showing the full budget history.

## ab_testing.py

**Combines:** fundamentals/07 (branching) + patterns/03 (LLM config) + fundamentals/05 (diff) + patterns/09 (provenance query)

Branch from the same conversation state, run identical prompts on each branch with different model configs (e.g., `gpt-4o` vs `claude-3-5-sonnet`), then `diff()` the results and `query_by_config(model=)` to retrieve each branch's outputs by config. Compare response quality, token usage, and latency across branches.

## context_forensics.py

**Combines:** fundamentals/05 (log + time travel) + fundamentals/07 (branching) + patterns/07 (rebase + import)

Walk `log()` to find the commit where bad data entered the conversation. `compile(at_commit=hash)` reconstructs exactly what the LLM was seeing at that point. Branch from just before the contamination, cherry-pick the good work with `import_commit()`, and rebase the clean branch onto main.

## research_delegation.py

**Combines:** advanced/multi_agent (parent-child + delegation) + patterns/04 (compression) + fundamentals/07 (branching + merge)

Three sub-agents research in parallel, each in its own child tract on a dedicated branch. When each sub-agent finishes, compress its history into a summary. The supervisor merges the summary commits onto main, resolving any conflicts. The full research history is preserved in child tracts; main carries only the synthesized findings.

## autonomous_steering.py

**Combines:** advanced/orchestrator (loop + triggers) + advanced/policies (builtin + autonomy) + advanced/hooks (compress review) + patterns/04 (compression)

All built-in policies are active at `autonomy="autonomous"`. The orchestrator fires on commit count and token threshold triggers, running context assessment and maintenance in a loop. A single hook is attached to `"compress"` for human sign-off on large compressions. The agent manages its own context window end-to-end, with one override point for high-stakes decisions.
