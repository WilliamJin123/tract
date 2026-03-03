# Tract Cookbook — Scenarios

The cookbook is organized around **who drives the context**. **Getting Started** has two on-ramps: one for developers writing code, one for agents using tools. **Developer** covers everything you do when calling tract methods directly. **Agentic** covers the spectrum from self-managing agents (tools in the agent's own loop) to sidecar agents (companion handles context management). **Integrations** shows how to use tract with external agent frameworks (Agno, LangChain, CrewAI) via the framework-agnostic `as_callable_tools()` and pluggable `AgentLoop` protocol. **Hooks** provide the cross-cutting approval and middleware layer used by both patterns. **E2E** combines everything into real-world scenarios.

### Two Mental Models

| Pattern | Who decides | Where to start |
|---------|------------|----------------|
| **Developer** | You call methods, you control the flow | `getting_started/01_chat.py` |
| **Agent** | The agent uses tools, manages its own context | `getting_started/02_agent.py` |

The agent pattern has a spectrum:

| Approach | Complexity | When to use |
|----------|-----------|-------------|
| **Self-managing** | Simple meta-decisions inline | Good tool descriptions are enough (temperature, pin, tag) |
| **Sidecar** | Complex meta-reasoning separated | Compress, GC, branch decisions shouldn't burden the main model |

### 2-Tier Convention

Most cookbook files follow a two-tier pattern showing the same feature at different autonomy levels:

| Tier | Label | Description |
|------|-------|-------------|
| **PART 1** | Manual | Direct API calls, no LLM, fully deterministic |
| **PART 2** | Agent / Automated | Orchestrator, triggers, hooks auto-manage |

For interactive (HITL) patterns, see `hooks/` — the hook system (`review=True`, `t.on()`) is the built-in interactivity mechanism. Every file is standalone.

## File Tree

```
cookbook/
├── SCENARIOS.md
│
├── getting_started/                        # Two doors in
│   ├── 01_chat.py                            # Developer: persistent chat in 25 lines
│   └── 02_agent.py                           # Agent: self-managing with tools in 40 lines
│
├── developer/                               # You call the methods, you control the flow
│   │
│   ├── 00_internals.py                        # Under the hood: commit(), compile(), content types
│   │
│   ├── conversations/                         # Building and managing conversations
│   │   ├── 01_shorthand_and_format.py           # system/user/assistant, to_openai/anthropic/dicts
│   │   ├── 02_batch.py                          # batch() context manager, atomic operations
│   │   ├── 03_status_and_budget.py              # status(), TractConfig, token budget tracking
│   │   └── 04_chat_and_persist.py               # chat(), ChatResponse, persistence, session resume
│   │
│   ├── history/                               # Inspecting and navigating the past
│   │   ├── 01_log_and_diff.py                   # log, show, diff, compile(at_commit=), compile(at_time=)
│   │   ├── 02_rollback.py                       # checkout, reset, compile(at_commit=)
│   │   └── 03_edit_history.py                   # log(include_edits=True), edit chain tracking
│   │
│   ├── operations/                            # Context-shaping operations
│   │   ├── 01_compress.py                       # Manual, interactive, and LLM compression
│   │   ├── 02_guided_compression.py             # Priorities, retention guarantees, retain_match
│   │   ├── 03_autonomous_compression.py         # ToolExecutor, hooks, CompressTrigger automation
│   │   ├── 04_branch.py                         # branch, switch, list, delete, BranchTrigger tangent detection
│   │   ├── 05_merge_strategies.py               # FF merge, clean merge, no_ff, MergeTrigger completion
│   │   ├── 06_merge_conflicts.py                # ConflictInfo, edit_resolution, commit_merge
│   │   ├── 07_import_commit.py                  # import_commit (cherry-pick), ImportResult
│   │   ├── 08_rebase.py                         # rebase, RebaseResult, replayed_commits
│   │   ├── 09_gc.py                             # gc(), GCResult, archive retention
│   │   ├── 10_retention_policies.py             # archive_retention_days, conservative vs aggressive
│   │   ├── 11_reorder.py                        # compile(order=), ReorderWarning
│   │   └── sample_contract.md                   # Sample data for compression demos
│   │
│   ├── metadata/                              # Data attached to commits
│   │   ├── 01_tags.py                           # auto-classify, explicit tags, mutable tags, registry, queries
│   │   ├── 02_priority.py                       # annotate(), Priority.PINNED/SKIP/NORMAL
│   │   ├── 03_edit_in_place.py                  # system(edit=hash), edit-in-place workflow
│   │   ├── 04_tool_results.py                   # set_tools, tool_result, compress_tool_calls
│   │   ├── 05_tool_summarization.py             # configure_tool_summarization, auto-summarize
│   │   ├── 06_offline_tool_management.py        # tool management without LLM calls
│   │   ├── 07_reasoning.py                      # t.reasoning(), ReasoningContent, SKIP default
│   │   ├── 08_reasoning_compile.py              # compile(include_reasoning=True), annotate overrides
│   │   ├── 09_reasoning_formatting.py           # pprint reasoning in dim cyan
│   │   ├── 10_reasoning_llm.py                  # generate() auto-extract, reasoning=False
│   │   └── _helpers.py                          # Shared utilities for tool result examples
│   │
│   ├── config/                                # LLM routing, budgets, generation config
│   │   ├── 01_per_call.py                       # LLMConfig on chat/generate, sugar params
│   │   ├── 02_operation_config.py               # configure_operations, per-op LLMConfig
│   │   ├── 03_operation_clients.py              # configure_clients, separate LLM clients per op
│   │   ├── 04_resolution_chain.py               # 4-level chain: sugar > llm_config > operation > default
│   │   ├── 05_message_config.py                  # auto-message commit message LLM config
│   │   └── 06_budget_guardrail.py               # status() loop, budget check before chat, auto-stop
│   │
│   ├── queries/                               # Inspecting and auditing history
│   │   ├── 01_tool_queries.py                   # find_tool_results, find_tool_calls, find_tool_turns
│   │   ├── 02_surgical_edits.py                 # tool_result(edit=hash), trimming verbose results
│   │   ├── 03_selective_compression.py          # compress_tool_calls(name=), targeted compression
│   │   ├── 04_config_provenance.py              # query_by_config, generation_config tracking
│   │   ├── 05_tool_provenance.py                # set_tools, get_commit_tools, to_openai_params
│   │   └── _helpers.py                          # Shared utilities for query examples
│   │
│   └── validation/                            # Retry and validation patterns
│       ├── 01_core_retry.py                     # retry_with_steering, RetryResult, RetryExhaustedError
│       ├── 02_chat_validation.py                # chat(validator=), purify=, provenance_note=
│       └── 03_compress_validation.py            # compress(validator=), retain_match= combo
│
├── agentic/                                 # Your agent drives the context
│   │
│   ├── self_managing/                         # Agent has tract tools in its own loop
│   │   ├── 01_tool_hints.py                     # Description-driven behavior, no system prompt crutches
│   │   ├── 02_lightweight_ops.py                # Pin, tag, configure_model — inline decisions
│   │   ├── 03_budget_awareness.py               # Agent reads status(), toggle_triggers for bulk ops
│   │   └── 04_profiles.py                       # self/supervisor/observer tool scoping
│   │
│   ├── sidecar/                               # Companion agent handles context management
│   │   ├── 01_triggers.py                       # Built-in triggers, autonomy spectrum, hook interception
│   │   ├── 02_assessment_loop.py                # OrchestratorConfig, assessment loop, adaptive triggers
│   │   ├── 03_toolkit.py                        # as_tools, profiles, agent registers own triggers
│   │   └── 04_auto_tagger.py                    # Orchestrator-driven retrospective tagging
│   │
│   └── multi_agent/                           # Coordination across agents
│       ├── 01_parent_child.py                   # Child tracts, provenance, parent()
│       ├── 02_delegation.py                     # Branch-delegate-merge, compress-and-ingest
│       └── 03_curated_deploy.py                 # session.deploy(), curation, merge-back
│
├── integrations/                              # External framework integration (requires extra deps)
│   ├── 01_callable_tools.py                     # as_callable_tools() -- framework-agnostic export
│   ├── 02_agent_loop.py                         # AgentLoop protocol -- pluggable orchestrator
│   ├── 03_agno.py                               # Agno: TractToolkit, message sync, adapter
│   ├── 04_langchain.py                          # LangChain/LangGraph: tools, graph nodes, adapter
│   └── 05_crewai.py                             # CrewAI: tools, multi-agent tracts, delegation
│
├── hooks/                                   # Approval + middleware layer (cross-cutting)
│   ├── 01_routing/                            # Core registration and dispatch
│   │   ├── 01_three_tier.py                     # t.on/off, three-tier routing, review=True
│   │   ├── 02_catch_all.py                      # Catch-all handlers
│   │   └── 03_recursion_guard.py                # Preventing hook recursion
│   ├── 02_pending/                            # Per-operation Pending objects
│   │   ├── 01_compress_lifecycle.py             # PendingCompress lifecycle
│   │   ├── 02_compress_handlers.py              # Common handler patterns
│   │   ├── 03_compress_retry.py                 # Retry and validate in hooks
│   │   ├── 04_gc.py                             # PendingGC, exclude()
│   │   ├── 05_rebase.py                         # PendingRebase, replay inspection
│   │   ├── 06_merge_conflicts.py                # PendingMerge, conflict review
│   │   ├── 07_merge_retry.py                    # Merge retry patterns
│   │   ├── 08_tool_result_basics.py             # PendingToolResult, approve/reject/edit
│   │   ├── 09_tool_result_edit.py               # edit_result, summarize(instructions=)
│   │   ├── 10_tool_result_config.py             # configure_tool_summarization()
│   │   └── 11_tool_result_routing.py            # Per-tool routing strategies
│   ├── 03_agent_interface/                    # Making Pending objects agent-consumable
│   │   ├── 01_serialization.py                  # Pending.to_dict() for serialization
│   │   ├── 02_tool_schemas.py                   # Pending.to_tools() for function calling
│   │   ├── 03_docs.py                           # Pending.describe_api() human-readable
│   │   └── 04_dispatch.py                       # apply_decision() routing
│   ├── 04_middleware/                         # Developer and agent middleware patterns
│   │   ├── 01_basic_gate.py                     # Token budget gate
│   │   ├── 02_auto_truncate.py                  # Auto-truncate on threshold
│   │   ├── 03_middleware_enforcer.py            # Middleware-style enforcement
│   │   ├── 04_dynamic_budget.py                 # Dynamic budget adjustment
│   │   ├── 05_ordering_basics.py                # Message ordering hooks
│   │   ├── 06_pass_through.py                   # Pass-through patterns
│   │   ├── 07_conditional.py                    # Conditional ordering
│   │   ├── 08_dynamic_insertion.py              # Dynamic message insertion
│   │   ├── 09_full_pipeline.py                  # Full ordering pipeline
│   │   ├── 10_register_and_fire.py              # Dynamic hook registration
│   │   ├── 11_introspection.py                  # Hook introspection
│   │   └── 12_review_and_execute.py             # Review-then-execute pattern
│   └── 05_guidance/                           # Two-stage judgment patterns
│       ├── 01_guidance.py                       # GuidanceMixin, two-stage reasoning
│       └── 02_two_stage.py                      # Two-stage judgment + execution
│
└── e2e/                                     # End-to-end scenarios combining features
    ├── self_correcting_agent.py               # [self-managing] retry + edit + validation + provenance
    ├── long_running_session.py                # [sidecar] triggers + agent self-configures triggers + 50+ turns
    ├── ab_testing.py                          # [developer] branch + config + diff + provenance query
    ├── context_forensics.py                   # [developer] log + time-travel + branch + rebase
    ├── research_delegation.py                 # [sidecar + multi-agent] compress + merge
    └── autonomous_steering.py                 # [sidecar] orchestrator + triggers + hooks + drift
```

---

# Getting Started

Two on-ramps — pick the one that matches how you'll use tract.

## 01 — Hello Chat (Developer On-Ramp)

**File:** `getting_started/01_chat.py`

**Use case:** You're building an app and want persistent, managed conversation history.

Open a tract with a file path, set a system prompt, chat with `t.chat()`. Close, reopen — the conversation continues. The simplest possible tract usage in ~25 lines.

> `Tract.open()`, `system()`, `chat()`, persistence, `status()`

## 02 — Hello Agent (Agent On-Ramp)

**File:** `getting_started/02_agent.py`

**Use case:** You're building an agent that should manage its own context window.

Open a tract, get tools with `as_tools()`, use `ToolExecutor` to dispatch agent operations: check status, pin important messages, compress when budget is high. The building blocks for self-managing agents.

> `as_tools()`, `ToolExecutor`, agent-driven status/annotate/compress

---

# Developer

You call the methods, you control the flow. This section covers every tract feature from the developer's perspective.

## Internals

**File:** `developer/00_internals.py`

**Use case:** You want to understand what tract is actually doing under the hood — no shortcuts, no magic.

Open an in-memory tract. Commit messages using `InstructionContent` and `DialogueContent` models directly. Call `compile()` to turn the commit chain into a message list. Use `ctx.pprint()` to inspect.

> `commit()`, `InstructionContent`, `DialogueContent`, `compile()`, `CompiledContext.messages`, `ctx.pprint()`

## Conversations

### 01 — Shorthand and Format Methods

**File:** `developer/conversations/01_shorthand_and_format.py`

**Use case:** The convenience layer: `system()`, `user()`, `assistant()` instead of manual content models. Format output for any LLM provider.

> `system()`, `user()`, `assistant()`, `to_dicts()`, `to_openai()`, `to_anthropic()`

### 02 — Batch

**File:** `developer/conversations/02_batch.py`

**Use case:** A RAG retrieval plus user question must land as one atomic unit — partial state is worse than nothing.

> `with t.batch(): ...`, rollback on exception, clean retry after rollback

### 03 — Status and Token Budget

**File:** `developer/conversations/03_status_and_budget.py`

**Use case:** Track tokens and budget fill percentage without an LLM call.

> `TractConfig(token_budget=TokenBudgetConfig(max_tokens=))`, `status()`, `status.pprint()`

### 04 — Chat and Persist

**File:** `developer/conversations/04_chat_and_persist.py`

**Use case:** Full chat workflow with persistence and session resume.

> `chat()`, `ChatResponse`, `response.pprint()`, persistence with file path + `tract_id`

## History

### 01 — Log and Diff

**File:** `developer/history/01_log_and_diff.py`

**Use case:** Walk history, compare two states, reconstruct past context.

> `log()`, `diff(hash_a, hash_b)`, `compile(at_commit=)`, `compile(at_time=)`

### 02 — Rollback

**File:** `developer/history/02_rollback.py`

**Use case:** Undo recent changes and go back to a known good state.

> `checkout()`, `reset()`

### 03 — Edit History

**File:** `developer/history/03_edit_history.py`

**Use case:** Trace how a message got to its current state.

> `log(include_edits=True)`, edit chain

## Operations

### 01 — Core Compression

**File:** `developer/operations/01_compress.py`
**Tiers:** Manual | Agent

> `compress(content=)`, `compress(review=True)`, `PendingCompress`, `instructions=`

### 02 — Guided Compression and Retention

**File:** `developer/operations/02_guided_compression.py`
**Tiers:** Manual | Agent

> `Priority.IMPORTANT`, `retain_match=`, `retain_match_mode=`, `preserve=`, `max_retries=`

### 03 — Autonomous Compression

**File:** `developer/operations/03_autonomous_compression.py`
**Tiers:** Manual | Agent

> `ToolExecutor`, `t.on("compress", ...)`, `CompressTrigger`, `configure_triggers()`

### 04 — Branch Lifecycle

**File:** `developer/operations/04_branch.py`
**Tiers:** Manual | Agent

> `branch()`, `switch()`, `list_branches()`, `current_branch`, `delete_branch(force=True)`, `BranchTrigger`, `configure_triggers()`

### 05 — Merge Strategies

**File:** `developer/operations/05_merge_strategies.py`
**Tiers:** Manual | Agent

> `merge()`, `MergeResult`, `merge_type`, `no_ff`, `delete_branch=True`, `MergeTrigger`, `configure_triggers()`

### 06 — Merge Conflicts

**File:** `developer/operations/06_merge_conflicts.py`
**Tiers:** Manual | Agent

> `ConflictInfo`, `edit_resolution()`, `commit_merge()`

### 07 — Import Commit

**File:** `developer/operations/07_import_commit.py`
**Tiers:** Manual | Agent

> `import_commit(hash)`, `ImportResult`

### 08 — Rebase

**File:** `developer/operations/08_rebase.py`
**Tiers:** Manual | Agent

> `rebase("main")`, `RebaseResult`, `replayed_commits`, `new_head`

### 09 — GC After Compression

**File:** `developer/operations/09_gc.py`
**Tiers:** Manual | Agent

> `gc(archive_retention_days=)`, `GCResult`

### 10 — Retention Policies

**File:** `developer/operations/10_retention_policies.py`
**Tiers:** Manual | Agent

> `archive_retention_days` parameter

### 11 — Message Reordering

**File:** `developer/operations/11_reorder.py`
**Tiers:** Manual | Agent

> `compile(order=)`, `ReorderWarning`

## Metadata

### 01 — Tags: Classify and Query

**File:** `developer/metadata/01_tags.py`
**Tiers:** Manual | Agent

> `tags=["..."]`, `t.tag()`, `t.untag()`, `register_tag()`, `query_by_tags()`, `log(tags=)`

### 02 — Priority: Pin, Skip, Reset

**File:** `developer/metadata/02_priority.py`
**Tiers:** Manual | Agent

> `annotate(hash, Priority.PINNED/SKIP/NORMAL)`, `Priority` enum

### 03 — Edit in Place

**File:** `developer/metadata/03_edit_in_place.py`
**Tiers:** Manual | Agent

> `system(edit=hash)`, edit-in-place workflow

### 04 — Tool Results: Agentic Loop

**File:** `developer/metadata/04_tool_results.py`
**Tiers:** Manual | Agent

> `set_tools()`, `tool_result()`, `ToolCall`, `compress_tool_calls()`

### 05 — Tool Summarization

**File:** `developer/metadata/05_tool_summarization.py`
**Tiers:** Manual | Agent

> `configure_tool_summarization()`, auto-summarize hooks

### 06 — Offline Tool Management

**File:** `developer/metadata/06_offline_tool_management.py`
**Tiers:** Manual | Agent

> `find_tool_results()`, `find_tool_calls()`, `find_tool_turns()`, `tool_result(edit=)`

### 07 — Manual Reasoning

**File:** `developer/metadata/07_reasoning.py`
**Tiers:** Manual | Agent

> `t.reasoning()`, `ReasoningContent`, `format=`

### 08 — Reasoning Compile Control

**File:** `developer/metadata/08_reasoning_compile.py`
**Tiers:** Manual | Agent

> `compile(include_reasoning=True)`, `annotate()` overrides

### 09 — Reasoning Formatting

**File:** `developer/metadata/09_reasoning_formatting.py`
**Tiers:** Manual | Agent

> `pprint()` rendering for reasoning

### 10 — Reasoning LLM Integration

**File:** `developer/metadata/10_reasoning_llm.py`
**Tiers:** Manual | Agent

> `generate()` auto-extract, `reasoning=False`, `commit_reasoning=False`, `ChatResponse.reasoning`

## Config

### 01 — Per-Call Config

**File:** `developer/config/01_per_call.py`
**Tiers:** Manual | Agent

> `LLMConfig`, `chat(temperature=)`, `generate(llm_config=)`

### 02 — Operation Config

**File:** `developer/config/02_operation_config.py`
**Tiers:** Manual | Agent

> `default_config=`, `configure_operations()`, `OperationConfigs`

### 03 — Operation Clients

**File:** `developer/config/03_operation_clients.py`
**Tiers:** Manual | Agent

> `configure_clients()`, per-operation routing

### 04 — Resolution Chain

**File:** `developer/config/04_resolution_chain.py`
**Tiers:** Manual | Agent

> 4-level resolution chain, `LLMConfig.from_dict()`, alias handling

### 05 — Message Config

**File:** `developer/config/05_message_config.py`
**Tiers:** Manual | Agent

> Auto-message commit message LLM config

### 06 — Budget Guardrail

**File:** `developer/config/06_budget_guardrail.py`
**Tiers:** Manual | Agent

> `status()` in a loop, budget threshold check, `record_usage()`

## Queries

### 01 — Tool Queries

**File:** `developer/queries/01_tool_queries.py`
**Tiers:** Manual | Agent

> `find_tool_results(name=, after=)`, `find_tool_calls(name=)`, `find_tool_turns(name=)`, `ToolTurn`

### 02 — Surgical Edits

**File:** `developer/queries/02_surgical_edits.py`
**Tiers:** Manual | Agent

> `tool_result(edit=)`, surgical replacement

### 03 — Selective Compression

**File:** `developer/queries/03_selective_compression.py`
**Tiers:** Manual | Agent

> `compress_tool_calls(name=)`, targeted compression

### 04 — Config Provenance

**File:** `developer/queries/04_config_provenance.py`
**Tiers:** Manual | Agent

> `query_by_config(model=, temperature=)`, `generation_config`

### 05 — Tool Provenance

**File:** `developer/queries/05_tool_provenance.py`
**Tiers:** Manual | Agent

> `set_tools()`, `get_commit_tools()`, `to_openai_params()`, `to_anthropic_params()`

## Validation

### 01 — Core Retry Primitive

**File:** `developer/validation/01_core_retry.py`
**Tiers:** Manual | Agent

> `retry_with_steering()`, `RetryResult`, `RetryExhaustedError`

### 02 — Chat Validation

**File:** `developer/validation/02_chat_validation.py`
**Tiers:** Manual | Agent

> `chat(validator=, max_retries=, purify=, provenance_note=, retry_prompt=)`

### 03 — Compress Validation

**File:** `developer/validation/03_compress_validation.py`
**Tiers:** Manual | Agent

> `compress(validator=, max_retries=)`, `retain_match=` combo

---

# Agentic

Your agent drives the context. This section covers the spectrum from self-managing (agent has tools in its own loop) to sidecar (companion agent handles context management).

## Self-Managing

The agent gets tract tools alongside its task tools and makes meta-decisions inline. For simple decisions, a good tool description is all you need — no system prompt crutches.

### 01 — Tool Description Hints

**File:** `agentic/self_managing/01_tool_hints.py`
**Tiers:** Manual | Agent

**Use case:** The simplest self-managing pattern: tool descriptions tell the agent when to act.

A `configure_model` tool with description "call BEFORE answering when creative vs precise" reliably triggers the right behavior without system prompt instructions.

> `ToolProfile` customization, description-driven tool selection

### 02 — Lightweight Inline Operations

**File:** `agentic/self_managing/02_lightweight_ops.py`
**Tiers:** Manual | Agent

**Use case:** Agent tags, pins, and checks status as part of its normal workflow.

> `ToolExecutor` for tag/annotate/status, inline agent decisions

### 03 — Budget Awareness

**File:** `agentic/self_managing/03_budget_awareness.py`
**Tiers:** Manual | Agent

**Use case:** Agent monitors its own budget and self-compresses when running hot.

> status() via tools, budget-driven compress decisions, self-adaptation

### 04 — Tool Profiles

**File:** `agentic/self_managing/04_profiles.py`
**Tiers:** Manual | Agent

**Use case:** Scope what an agent can do: full CRUD, oversight only, or read-only monitoring.

> `as_tools(profile=)`, `ToolExecutor` profiles: `"self"`, `"supervisor"`, `"observer"`

## Sidecar

A companion agent (possibly cheaper/smaller model) handles tract operations while the main model focuses on the task. Wins for complex meta-reasoning where the main model shouldn't be burdened.

### 01 — Triggers

**File:** `agentic/sidecar/01_triggers.py`
**Tiers:** Manual | Agent

**Use case:** Threshold-based automation with hook interception.

7 built-in triggers: `CompressTrigger`, `PinTrigger`, `RebaseTrigger`, `GCTrigger`, `MergeTrigger`, `BranchTrigger`, `ArchiveTrigger`. Autonomy spectrum from fully autonomous to collaborative.

> `CompressTrigger`, `PinTrigger`, `BranchTrigger`, `ArchiveTrigger`, `configure_triggers()`, `PendingTrigger`, autonomy spectrum

### 02 — Assessment Loop

**File:** `agentic/sidecar/02_assessment_loop.py`
**Tiers:** Manual | Agent

**Use case:** Auto-assess context health and execute maintenance autonomously.

> `OrchestratorConfig`, `TriggerConfig`, assessment loop, HITL via hooks, `register_trigger`, adaptive trigger policies

### 03 — Toolkit

**File:** `agentic/sidecar/03_toolkit.py`
**Tiers:** Manual | Agent

**Use case:** Expose tract operations as LLM-callable tools for the sidecar.

> `as_tools(format=, profile=)`, `ToolExecutor`, profiles, `register_trigger`, `toggle_triggers`, agent self-configuring triggers

### 04 — Auto-Tagger

**File:** `agentic/sidecar/04_auto_tagger.py`
**Tiers:** Manual | Agent

**Use case:** LLM agent retrospectively tags a conversation using the orchestrator.

> `Orchestrator`, `OrchestratorConfig`, orchestrator-driven tagging

## Multi-Agent

Coordination across multiple agents with parent-child relationships.

### 01 — Parent-Child

**File:** `agentic/multi_agent/01_parent_child.py`
**Tiers:** Manual | Agent

> `parent()`, `children()`, parent-child provenance

### 02 — Delegation

**File:** `agentic/multi_agent/02_delegation.py`
**Tiers:** Manual | Agent

> `compress()` summary, `import_commit()` across tracts, compress-and-ingest pattern

### 03 — Curated Deploy

**File:** `agentic/multi_agent/03_curated_deploy.py`
**Tiers:** Manual | Agent

> `session.deploy()`, `curate=`, merge-back, collapse

---

# Integrations

Use tract with external agent frameworks. These examples require extra dependencies (`agno`, `langchain`, `crewai`). Two universal building blocks:

- **`as_callable_tools()`** exports tract tools as typed Python callables that any framework can introspect — no per-framework adapters needed.
- **`AgentLoop`** protocol lets you swap tract's built-in Orchestrator for an external framework's loop, the same way `LLMClient` lets you swap the LLM transport layer.

### 01 — Callable Tools

**File:** `integrations/01_callable_tools.py`

**Use case:** You want tract's context management tools (compress, branch, gc, etc.) available in any agent framework without writing adapter code.

`as_callable_tools()` returns functions with proper `__name__`, `__doc__`, `__signature__`, and type annotations. Every framework introspects these natively.

> `as_callable_tools()`, `inspect.signature()`, profile filtering, description overrides

### 02 — Agent Loop Protocol

**File:** `integrations/02_agent_loop.py`

**Use case:** You want `t.orchestrate()` to delegate to an external agent loop (Agno, LangGraph, custom) instead of the built-in Orchestrator.

The protocol is minimal: `run(messages, tools, execute_tool) -> AgentLoopResult` + `stop()`. Tract prepares everything, the loop does loop stuff, provenance flows back via the result type.

> `AgentLoop`, `AgentLoopResult`, `configure_agent_loop()`, `Tract.open(agent_loop=)`, provenance

### 03 — Agno

**File:** `integrations/03_agno.py`

**Use case:** You have an Agno agent (web search, reasoning, etc.) and want tract to manage its context window.

Two depths: inject tract tools via `as_callable_tools()`, or build a `TractToolkit` (native Agno Toolkit subclass) with message sync hooks.

> `as_callable_tools()` + Agno Agent, `TractToolkit`, pre/post hooks, `AgnoAdapter`

### 04 — LangChain / LangGraph

**File:** `integrations/04_langchain.py`

**Use case:** You have a LangChain agent or LangGraph graph and want tract tools available alongside task tools.

> `as_callable_tools()` + AgentExecutor, LangGraph tool nodes, callback-based provenance

### 05 — CrewAI

**File:** `integrations/05_crewai.py`

**Use case:** You have a CrewAI multi-agent workflow and want each agent to manage its own context.

> `as_callable_tools()` + CrewAI Agent, per-agent tracts, delegation with `import_commit()`

---

# Hooks

The approval and middleware layer — cross-cutting, used by both developer and agentic patterns. Every `Pending` subclass follows the same `approve()`/`reject()` protocol.

## 01 — Routing

**Files:** `hooks/01_routing/`

Hook registration, three-tier routing, catch-all handlers, and recursion guards.

> `t.on()`, `t.off()`, `Pending` base class, three-tier routing, `review=True`, recursion guard

## 02 — Pending Objects

**Files:** `hooks/02_pending/`

Per-operation `Pending` lifecycle and patterns: compression (lifecycle, handlers, retry), GC (exclude), rebase (replay inspection), merge (conflicts, retry), and tool results (basics, edit, config, routing).

> `PendingCompress`, `PendingGC`, `PendingRebase`, `PendingMerge`, `PendingToolResult`

## 03 — Agent Interface

**Files:** `hooks/03_agent_interface/`

Auto-generated agent-facing interfaces for every `Pending` subclass: serialization, tool schemas for function calling, human-readable docs, and decision dispatch.

> `to_dict()`, `to_tools()`, `describe_api()`, `apply_decision()`

## 04 — Middleware

**Files:** `hooks/04_middleware/`

Token budget enforcement (gates, auto-truncate, dynamic budgets), message ordering (basics, pass-through, conditional, dynamic insertion, full pipeline), and dynamic operations (registration, introspection, review-and-execute).

> Token budget hooks, ordering middleware, dynamic `t.on()`, hook introspection

## 05 — Guidance

**Files:** `hooks/05_guidance/`

Two-stage judgment patterns: `GuidanceMixin` for reasoning before execution, and the full two-stage pipeline.

> `GuidanceMixin`, `judge()`, `Judgment`, two-stage judgment + execution

---

# E2E

End-to-end scenarios combining features from across the cookbook. Each scenario is tagged with its primary pattern.

## self_correcting_agent.py — [self-managing]

**Combines:** validation (retry) + metadata/priority (edit + annotations) + operations/compress + queries/provenance

An agent that validates its own JSON output, retries with steering, and annotates critical decisions with `retain_match=` so they survive compression.

## long_running_session.py — [sidecar]

**Combines:** operations/compress + sidecar/triggers + operations/gc + conversations/chat

A 50+ turn session with `CompressTrigger(threshold=0.8)`, PINNED alert preservation, and `gc(archive_retention_days=30)`.

## ab_testing.py — [developer]

**Combines:** operations/branch + config + history/log_and_diff + queries/provenance

Branch the same conversation, run identical prompts with different configs, diff results and `query_by_config()`.

## context_forensics.py — [developer]

**Combines:** history/log_and_diff + operations/branch + operations/rebase

Walk log to find bad data, time-travel to reconstruct, branch from clean point, cherry-pick good work.

## research_delegation.py — [sidecar + multi-agent]

**Combines:** multi_agent + operations/compress + operations/merge

Three sub-agents research in parallel, compress findings, merge summaries.

## autonomous_steering.py — [sidecar]

**Combines:** sidecar/triggers + hooks + operations/compress

All triggers active at `autonomy="autonomous"`, one hook for human sign-off on large compressions.
