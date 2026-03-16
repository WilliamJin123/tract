# OpenAI Agents SDK -- Targeted Analysis

**Version**: 0.12.2 (March 14, 2026) | **Maturity**: Production (71 releases, weekly cadence, 20k GitHub stars, 228 contributors, MIT)
**Source**: `openai/openai-agents-python` | **Docs**: openai.github.io/openai-agents-python
**Lineage**: Production successor to Swarm (Oct 2024 educational experiment). Launched March 2025.

---

## A. Core Abstractions & Extension Points

Three primitives, deliberately minimal:

1. **Agent**: LLM + instructions + tools + handoffs + guardrails. Declarative dataclass. Supports `clone()` for variant creation. Instructions can be static strings or dynamic callables receiving `RunContextWrapper`.
2. **Runner**: Static execution engine. Three modes: `run()` (async), `run_sync()`, `run_streamed()`. Owns the agent loop: call LLM -> if handoff, switch agent and re-run -> if tool calls, execute and re-run -> if final output, stop.
3. **Handoff**: Agent-to-agent delegation surfaced as a tool to the LLM. The LLM decides routing at runtime via `transfer_to_<agent_name>` tool calls.

Supporting cast: `RunConfig` (per-run overrides for model, guardrails, tracing, handoff behavior), `ModelSettings` (temperature, tool_choice, parallel_tool_calls), `RunHooks`/`AgentHooks` (lifecycle callbacks).

Extension points: custom `Model` implementations, custom `ModelProvider`, `RunHooks` (on_agent_start/end, on_tool_start/end, on_handoff), `tool_use_behavior` strategies, `call_model_input_filter` for pre-LLM input mutation.

**Tract comparison**: OpenAI's unit of work is the *agent run* (LLM call + tool loop); tract's is the *commit*. OpenAI composes via agent handoffs (runtime LLM decisions); tract composes via DAG branches (developer-controlled). OpenAI's `Agent.clone()` parallels tract's `spawn()` for creating variants, but clone is config-only while spawn carries history.

## B. State & Memory Model

Four tiers, progressively sophisticated:

1. **RunContext**: Generic `TContext` type parameter for dependency injection. Passed through `RunContextWrapper[TContext]` to tools, guardrails, instructions. Scoped to a single run.
2. **to_input_list()**: Manual conversation threading -- append result items to next run's input. Zero persistence.
3. **Sessions**: Automatic conversation persistence. 10+ backends: SQLiteSession, RedisSession, SQLAlchemySession, DaprSession, EncryptedSession, AdvancedSQLiteSession (branching + usage tracking), OpenAIConversationsSession, OpenAIResponsesCompactionSession. SessionABC protocol for custom implementations.
4. **Server-managed**: `conversation_id` or `previous_response_id` for OpenAI-hosted state.

Sessions auto-retrieve history pre-run, auto-persist post-run. `session_input_callback` controls history merging. `SessionSettings(limit=N)` bounds retrieval.

**Tract challenge**: OpenAI Sessions are a flat append-only event log with session IDs -- no branching, no diffing, no merge conflicts, no ancestry. `AdvancedSQLiteSession` adds branching from turn N, which gestures toward tract's model but lacks the DAG semantics. Tract's `compare()`, `rebase()`, `merge()` have no Session equivalents. However, OpenAI's session diversity (10+ backends, encryption, compaction) is a maturity advantage in deployment options. The `OpenAIResponsesCompactionSession` auto-compacts long conversations -- similar in intent to tract's compression but server-side and opaque.

## C. Tool/Function Calling Design

`@function_tool` decorator: auto-extracts name, description (from docstring via `griffe`), and JSON schema (from type hints + Pydantic Field constraints). Supports sync/async. First parameter can be `RunContextWrapper` (excluded from schema).

```python
@function_tool
def get_weather(city: str) -> str:
    """Fetch weather for a city."""
    return f"Sunny in {city}"

# With constraints
@function_tool
def score(value: Annotated[int, Field(ge=0, le=100)]) -> str:
    return f"Score: {value}"
```

Advanced: `FunctionTool` class for manual schema control. `defer_loading=True` + `ToolSearchTool()` for lazy tool discovery in large tool surfaces. `timeout` and `failure_error_function` per tool. Multi-modal returns via `ToolOutputImage`/`ToolOutputFileContent`.

Tool guardrails: `@tool_input_guardrail` (pre-execution, can skip/reject), `@tool_output_guardrail` (post-execution, can replace output). Configured on the tool itself.

**Tract challenge**: OpenAI's `@function_tool` and tract's `@tool` decorator are near-identical in ergonomics. Both auto-generate schemas from type hints. Key differences: (1) OpenAI's tool guardrails are per-tool decorators; tract's gates are middleware events that can guard any operation. (2) OpenAI's `defer_loading` for large tool surfaces has no tract equivalent. (3) OpenAI's `tool_use_behavior` (`stop_on_first_tool`, `StopAtTools`) controls post-tool flow -- tract doesn't manage the LLM loop so this isn't applicable. (4) OpenAI's `failure_error_function` is per-tool error formatting; tract has no per-tool error hooks.

## D. Multi-Agent Patterns

Two composition models:

1. **Handoff** (decentralized): Agent declares `handoffs=[agent_a, agent_b]`. LLM sees these as `transfer_to_X` tools and decides routing. On handoff, Runner swaps the active agent, optionally filters input via `input_filter`, fires `on_handoff` callback. `input_type` passes structured metadata (reason, priority) at handoff time.

```python
triage = Agent(
    name="Triage",
    handoffs=[billing_agent, refund_agent],
    instructions="Route to the right specialist.",
)
```

2. **Agent-as-tool** (centralized): `agent.as_tool(tool_name=..., tool_description=...)`. Manager agent calls sub-agent synchronously, gets result back, retains control. No context switch.

`input_filter` controls what the receiving agent sees: `handoff_filters.remove_all_tools` strips tool history, `nest_handoff_history` collapses transcript into a summary block. `RECOMMENDED_PROMPT_PREFIX` provides standard handoff-aware instructions.

**Tract challenge**: Handoffs are OpenAI's signature innovation -- the LLM dynamically routes between specialized agents at runtime. Tract's `spawn()` creates child branches (developer-initiated, structural), while handoffs are LLM-initiated and dynamic. The `input_filter` on handoffs parallels tract's `spawn(include_tags=..., exclude_tags=...)` for selective history inheritance, but operates on conversation items rather than commits. The agent-as-tool pattern (synchronous sub-agent call) has no tract equivalent. If tract enters orchestration, handoffs represent the key pattern to study or integrate.

## E. LLM Client & Streaming

**Provider model**: `ModelProvider` protocol maps model name strings to `Model` instances. Three levels: global client (`set_default_openai_client`), runner-level (`RunConfig(model_provider=...)`), agent-level (`Agent(model=...)`).

Default: OpenAI Responses API. Alternative: Chat Completions API via `set_default_openai_api("chat_completions")`. Multi-provider via `MultiProvider` (prefix-based routing: `openai/gpt-5`, `litellm/anthropic/claude-3.5`). LiteLLM extension covers 100+ providers.

Streaming: `Runner.run_streamed()` returns `RunResultStreaming` with async `stream_events()` iterator. WebSocket transport option (`set_default_openai_responses_transport("websocket")`). Reusable sessions via `responses_websocket_session()`.

Structured output: `Agent(output_type=PydanticModel)` -- the final output must conform to the schema or the run fails.

**Tract challenge**: OpenAI's `Model`/`ModelProvider` protocol is functionally similar to tract's `LLMClient` protocol -- both abstract the provider. Key difference: OpenAI's is runner-managed (the framework calls the model), while tract's is developer-managed (you call `t.chat()`). OpenAI's streaming is tightly integrated with the agent loop; tract's `astream()` is a standalone method. OpenAI's multi-provider routing via `MultiProvider` prefixes is more ergonomic than tract's manual client configuration. The 100+ provider coverage via LiteLLM is a significant distribution advantage.

## F. Control Flow & Error Handling

**Guardrails**: Input guardrails run on first agent (parallel by default, or blocking with `run_in_parallel=False`). Output guardrails run on final agent (always post-execution). Both return `GuardrailFunctionOutput(tripwire_triggered=bool)`. Tripwire raises `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`, halting execution. Guardrails can themselves use agents:

```python
@input_guardrail
async def check_homework(ctx, agent, input):
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        tripwire_triggered=result.final_output.is_homework,
    )
```

**Human-in-the-loop**: Tool approval via `needs_approval` flag. Run pauses with `result.interruptions`. Resume via `state.approve(interruption)` / `state.reject(interruption)`. Durable execution integrations: Temporal, Restate, DBOS.

**Error handling**: `MaxTurnsExceeded` (configurable `max_turns`), `ModelBehaviorError`, `ToolTimeoutError`. `error_handlers={"max_turns": handler}` for graceful degradation. All inherit `AgentsException`.

**Tract challenge**: OpenAI's guardrails (tripwire model) and tract's gates (LLM-judged criteria) solve the same problem differently. OpenAI's run in parallel with agent execution for latency optimization -- tract's gates are pre-operation middleware. OpenAI's guardrails are input/output scoped; tract's gates bind to any of 12 middleware events. OpenAI's tripwire is binary (pass/fail); tract's `GateResult` includes `reason` and `tokens_used`. The human-in-the-loop interruption/resume pattern is a genuine capability tract lacks entirely -- tract has no pause/resume semantics.

## G. API Ergonomics -- Measurable

**Imports for hello-world**: 2 (`Agent`, `Runner`). With tools: 3 (add `function_tool`).

**Lines for basic agent**: 5 (Agent definition + Runner.run_sync + print).

**Lines for multi-agent with handoff**: ~15 (3 agents + handoff list + run).

**Lines for agent with tools + guardrails**: ~30.

**Boilerplate ratio**: Very low. No registry, no configuration object, no initialization ceremony. The `Runner` is stateless static methods.

**Top GitHub issues**: (1) Non-OpenAI model compatibility (thinking blocks lost in conversion, LiteLLM streaming errors). (2) Handoff reliability degradation over long conversations. (3) Reasoning model edge cases (store=True conflicts, missing thought_signature for Gemini).

**Error experience**: Exceptions are typed and descriptive. `AgentsException` base class. `tool_error_formatter` customizes model-visible error messages for approval rejections.

## H. Observability & Debugging

Built-in tracing enabled by default. Auto-traces: agent runs, LLM generations, tool calls, guardrails, handoffs. Traces and spans with parent relationships. Custom traces via `with trace("name"):` context manager. Specialized spans: `generation_span()`, `function_span()`, `guardrail_span()`, `handoff_span()`, `custom_span()`.

Export: default to OpenAI backend for dashboard visualization. `add_trace_processor()` for additional exporters. 20+ integrations (W&B, Arize, MLflow, Langfuse, LangSmith). Sensitive data control via `trace_include_sensitive_data`.

Debug logging: `enable_verbose_stdout_logging()`. Logger names: `openai.agents`, `openai.agents.tracing`.

**Tract angle**: OpenAI's traces are ephemeral observability data -- fire-and-forget spans. Tract's commit history *is* the trace, and it's persistent, queryable, diffable, and branchable. OpenAI needs external tools to inspect what happened; tract's `find()`, `compare()`, and `log()` are built-in. However, OpenAI's integration with 20+ observability platforms provides production-grade monitoring that tract doesn't address. The two approaches are complementary: tract captures *what the context looked like*, OpenAI captures *what the system did*.

## I. Testing

No first-party mock utilities found in public documentation. The SDK's own test suite is internal. Community relies on standard pytest mocking of the `Model` interface or the OpenAI client.

Pydantic AI (a competitor) provides `TestModel` and `FunctionModel` as explicit test doubles -- OpenAI Agents SDK does not. The `Model` protocol is implementable, so custom test models are straightforward but require boilerplate.

**Tract angle**: Tract's test infrastructure (912 pure-core tests, zero runner deps) represents a stronger testing story. The absence of first-party test doubles in OpenAI's SDK is a notable gap -- every user must build their own mock Model implementation or hit the live API.

---

## What's Unique

**OpenAI's signature contribution is the handoff-as-tool pattern.** By representing agent delegation as LLM tool calls (`transfer_to_X`), they collapse the multi-agent routing problem into the tool-calling mechanism LLMs already understand. This is elegant: no new routing DSL, no orchestration config -- the LLM reads handoff descriptions and decides. Combined with `input_filter` for context scoping and `on_handoff` for side effects, it's a complete delegation primitive.

The **guardrail tripwire model** with parallel execution is the second key design: optimistic execution (start agent + guardrail simultaneously) with fast abort on violation. This trades wasted tokens for latency reduction -- a pragmatic production tradeoff.

The **Sessions ecosystem** (10+ backends, encryption, compaction, branching) shows what happens when a framework commits to deployment diversity. The `AdvancedSQLiteSession` with branching is particularly interesting -- it gestures toward tract-like history management within OpenAI's simpler model.

**Key takeaway for tract**: The handoff mechanism validates that LLM-driven dynamic routing is the dominant multi-agent pattern. Tract's `spawn()`/`collapse()` is structural composition; handoffs are runtime delegation. If tract supports agent orchestration, wrapping `spawn()` with an LLM-decidable interface (handoff-as-tool exposing branch creation) would bridge this gap. The guardrail parallel-execution model is worth studying for tract's gates -- currently gates block synchronously, but an optimistic-execution mode could reduce latency. The Sessions diversity is an implementation lesson: tract's SQLite-only storage is a deployment constraint that Session's multi-backend pattern addresses.
