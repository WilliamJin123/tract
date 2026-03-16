# Pydantic AI -- Deep Dive Analysis (Dimensions A-I)

**Version**: 1.68.0 (March 12, 2026) | **Stars**: ~15.5k | **License**: MIT | **Origin**: Pydantic team (Samuel Colvin)

Pydantic AI's thesis: bring the "FastAPI feeling" to agent development -- type safety, dependency injection, and validation as first-class concerns. It is the only framework in this analysis built by the team behind Pydantic itself, which gives it unique leverage over schema generation and structured output. The most instructive comparisons for tract are the DI system, the testing story, and the type-safe tool pattern.

---

## A. Core Abstractions & Extension Points

Three primary abstractions: **Agent** (generic over deps + output type), **RunContext** (DI carrier parameterized by deps type), **Tool** (auto-schema'd from type hints).

```python
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

@dataclass
class Deps:
    db: DatabaseConn
    user_id: int

class Output(BaseModel):
    advice: str
    risk: int

agent = Agent('openai:gpt-5.2', deps_type=Deps, output_type=Output)

@agent.tool
async def get_balance(ctx: RunContext[Deps], include_pending: bool) -> float:
    """Returns account balance."""
    return await ctx.deps.db.balance(ctx.deps.user_id, include_pending)
```

The Agent constructor takes ~25 parameters. Key ones: `output_type` (structured response via Pydantic model), `deps_type` (DI constraint), `tools`, `toolsets`, `instructions`, `model_settings`, `retries`, `end_strategy`, `history_processors`, `instrument`. Run methods: `run()` (async), `run_sync()`, `run_stream()`, `iter()` (node-by-node graph iteration).

**Extension points**: `@agent.tool` / `@agent.tool_plain` decorators, `@agent.instructions` (dynamic system prompts), `@agent.output_validator`, `prepare_tools` callback for dynamic tool filtering, `AbstractToolset` for custom tool collections, MCP server integration, and `Agent.override()` for test-time substitution.

**Composability**: Agents compose via tool-based delegation (agent A calls agent B inside a tool function) or programmatic hand-off (application code sequences agents). A separate **Graph** system (`pydantic_graph`) provides state-machine workflows with typed nodes, persistence, and Mermaid visualization.

**Tract challenge**: Pydantic AI's Agent has a comparable surface area to tract's Tract class (~25 constructor params vs tract's 142 methods). The critical difference is the generic type parameters -- `Agent[Deps, Output]` makes the compiler enforce that tools receive the right deps and runs return the right output type. Tract's `Tract` class is unparameterized; type safety comes from runtime Pydantic validation, not from generic constraints at the class level.

---

## B. State & Memory Model

State flows through **RunContext**, not stored objects. Dependencies are injected at `run()` call time and propagated to all tools/instructions via `ctx.deps`. Message history is explicit: `result.new_messages()` returns the conversation, and you pass `message_history=` to continue.

```python
result1 = agent.run_sync('Question 1', deps=deps)
result2 = agent.run_sync('Follow-up', deps=deps, message_history=result1.new_messages())
```

No built-in persistence -- history is a list of `ModelMessage` objects the caller manages. The Graph system adds optional persistence (`FileStatePersistence`, `FullStatePersistence`, or custom `BaseStatePersistence` subclasses).

**Tract challenge**: Pydantic AI's DI-based state is ephemeral and caller-managed. Tract's commit DAG is persistent and framework-managed. These are complementary, not competing -- tract solves long-lived context evolution (branching, merging, compression), while Pydantic AI solves single-run dependency wiring. The lesson: tract should not try to be the DI system. Instead, tract's compiled output should slot cleanly into Pydantic AI's `message_history` parameter. If `CompiledContext.to_dicts()` returned Pydantic AI-compatible `ModelMessage` objects, tract becomes a context backend for Pydantic AI agents.

---

## C. Tool/Function Calling Design

Tools are functions with type annotations. Schema generation is automatic from the function signature + docstring:

```python
@agent.tool_plain(docstring_format='google')
def search(query: str, max_results: int = 5) -> list[str]:
    """Search the knowledge base.

    Args:
        query: Search query text.
        max_results: Maximum results to return.
    """
    return db.search(query, max_results)
```

Pydantic AI extracts parameter descriptions from Google/NumPy/Sphinx docstrings and embeds them in the JSON schema sent to the LLM. Single-parameter tools with a Pydantic model get simplified schemas. Tools can return anything JSON-serializable.

Advanced features: `prepare` callbacks for dynamic tool filtering per-request, `requires_approval=True` for human-in-the-loop (validation runs before approval, so the LLM fixes bad args without bothering the user), `ToolOutput`/`NativeOutput`/`PromptedOutput` modes for structured results.

**Tract challenge**: Tract's `ToolDefinition` requires explicit `parameters` dict and `description`. Pydantic AI eliminates this boilerplate by deriving both from the function itself. Tract could adopt this: a `@tract.tool` decorator that introspects the function signature and docstring to auto-generate a `ToolDefinition`, keeping the explicit form as an escape hatch.

---

## D. Multi-Agent Patterns

Five tiers of complexity: (1) single agent, (2) agent delegation via tools, (3) programmatic hand-off, (4) graph-based workflows, (5) deep agents (planning + sandboxed execution).

Delegation pattern -- agent B is called inside agent A's tool:

```python
@parent_agent.tool
async def delegate(ctx: RunContext[Deps], task: str) -> str:
    result = await child_agent.run(task, deps=ctx.deps, usage=ctx.usage)
    return result.output
```

Key detail: passing `ctx.usage` to the child aggregates token costs across the delegation chain.

**Tract challenge**: Tract's `spawn()` creates a new branch with filtered context; `collapse()` merges results back. Pydantic AI's delegation is simpler (just call another agent) but loses tract's structural guarantees -- no audit trail of what the child saw, no diff between parent and child context. Tract's approach is better for accountability; Pydantic AI's is better for simplicity.

---

## E. LLM Client & Streaming

Model abstraction uses a `<provider>:<model>` string format (`'openai:gpt-5.2'`, `'anthropic:claude-sonnet-4-5'`). 11 built-in providers plus OpenAI-compatible catch-all. Three-layer architecture: **Model** (API implementation), **Provider** (auth/connection), **Profile** (schema restrictions per model family).

Advanced patterns: `FallbackModel` chains multiple models (switches on HTTP errors), `ConcurrencyLimitedModel` enforces rate limits.

Streaming: `run_stream()` returns `StreamedRunResult` with `stream_text()` and `stream_output()` async iterables. `run_stream_events()` gives raw `PartStartEvent`/`PartDeltaEvent` for fine-grained control. Structured output streams partial Pydantic models as tokens arrive.

**Tract challenge**: Tract's `LLMClient` protocol defines `chat()` and `stream()`. Pydantic AI's model layer is richer -- fallback chains, concurrency limits, and provider/profile separation. Tract should consider a `FallbackClient` wrapper (try model A, fall back to B on error) as a thin utility around its existing protocol.

---

## F. Control Flow & Error Handling

The run loop: send messages -> LLM responds -> if tool calls, execute tools and loop -> if output matches `output_type`, validate and return. Retries are built in at two levels: **tool retries** (tool raises `ModelRetry` with a message, LLM re-attempts) and **output validation retries** (Pydantic validation failure -> error sent back to LLM).

```python
@agent.output_validator
async def validate_sql(ctx: RunContext, output: Output) -> Output:
    try:
        await ctx.deps.db.execute(f'EXPLAIN {output.sql_query}')
    except QueryError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    return output
```

Human-in-the-loop: tools with `requires_approval=True` cause the run to end with `DeferredToolRequests`. The caller approves/rejects, then resumes with `deferred_tool_results`.

`UsageLimits` caps tokens, requests, and tool calls to prevent runaway agents.

**Tract challenge**: Pydantic AI's `ModelRetry` exception is elegant -- a tool says "try again" with context, and the framework handles the retry loop. Tract's `SemanticGate` blocks operations but doesn't have this "retry with feedback" pattern. Adding a `ModelRetry`-like mechanism to tract's middleware (gate says "not good enough, here's why" -> LLM retries) would be a direct improvement.

---

## G. API Ergonomics -- Measurable

**Import count**: Minimal case requires 1 import (`from pydantic_ai import Agent`). Typical case with DI: 3 imports (Agent, RunContext, plus Pydantic's BaseModel).

**Hello world**: 5 lines.

```python
from pydantic_ai import Agent
agent = Agent('openai:gpt-5.2', instructions='Be concise.')
result = agent.run_sync('Hello')
print(result.output)
```

**Structured output**: ~15 lines (define BaseModel + Agent + run).

**Tool agent with DI**: ~30 lines (deps dataclass + output model + agent + tool function + run).

**Top GitHub issues** (by community signal):
1. **State management** (#4322) -- users want guidance on mutable state in agents, race conditions with deps
2. **Empty response handling** (#3105) -- empty LLM responses crash with unintuitive errors
3. **Unified exception handling across models** (#3088) -- model-specific exceptions leak when swapping providers
4. **Orphaned streaming events** (#3108) -- TEXT_MESSAGE_CONTENT events after TEXT_MESSAGE_END
5. **Accidental dependency version bumps** (#3707) -- v1.30.0 broke users on older openai SDK

The state management issue is notable: Pydantic AI's DI system passes deps by reference, so mutation is possible but undocumented and race-prone. This is the gap tract fills -- explicit, versioned state transitions vs implicit mutation.

---

## H. Observability & Debugging

**Logfire integration** is Pydantic AI's flagship observability story:

```python
import logfire
logfire.configure()
logfire.instrument_pydantic_ai()
```

Three lines give you: agent run spans, model request/response traces, tool execution spans, HTTP request capture. Privacy controls via `InstrumentationSettings(include_content=False)`. Supports 10+ alternative backends (Langfuse, W&B Weave, Arize, MLflow, etc.) via OpenTelemetry.

The `iter()` method also serves as a debugging tool -- step through the agent's execution graph node by node.

**Tract angle**: Tract's commit history IS its observability -- every state change is recorded with metadata, tags, and timestamps. This is structurally richer than span-based tracing (you can diff, branch, search commits). But tract lacks real-time visibility into LLM calls. The two are complementary: tract for context evolution audit trail, Logfire/OTEL for LLM call performance.

---

## I. Testing

This is Pydantic AI's strongest differentiator. Two test models eliminate LLM calls entirely:

**TestModel**: Calls all registered tools, returns schema-valid synthetic data. Zero config.

```python
from pydantic_ai.models.test import TestModel

with agent.override(model=TestModel()):
    result = await agent.run('test prompt', deps=test_deps)
    assert isinstance(result.output, MyOutput)
```

**FunctionModel**: Full control over what the "LLM" returns, enabling deterministic tool-call sequences:

```python
from pydantic_ai.models.function import FunctionModel, AgentInfo

def mock_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 1})])
    return ModelResponse(parts=[TextPart('done')])

with agent.override(model=FunctionModel(mock_model)):
    result = await agent.run('test')
```

**capture_run_messages()** records all agent-model interactions for assertion. **ALLOW_MODEL_REQUESTS = False** as a safety guard prevents accidental real LLM calls in tests. **agent.override()** swaps model, deps, or tools without changing application code.

**Tract angle**: Tract has 2717 passing tests with 912 pure-core tests needing zero runner deps -- a strong foundation. But tract lacks Pydantic AI's `FunctionModel` equivalent for testing LLM-dependent paths (gates, maintainers, compression). A `MockLLMClient` that accepts a response function would let users write deterministic tests for tract's LLM operations without hitting real APIs. This is the single highest-value pattern to steal.

---

## Key Takeaways for Tract

| Pydantic AI Pattern | Tract Implication |
|---|---|
| `Agent[Deps, Output]` generic type params | Consider parameterizing Tract for compile output type safety |
| `RunContext[Deps]` DI in tools | Tract tools already receive context; ensure the DI pattern is as clean |
| Auto-schema from function signatures + docstrings | Add `@tract.tool` decorator that introspects signatures, replacing manual ToolDefinition |
| `TestModel` / `FunctionModel` for deterministic testing | Build a `MockLLMClient(response_fn)` for testing gates, maintainers, compression |
| `ModelRetry` exception in tools/validators | Add retry-with-feedback to tract's gate/middleware system |
| `FallbackModel` for provider resilience | Add a `FallbackClient` wrapper around tract's LLMClient protocol |
| `message_history` as explicit list | Ensure `CompiledContext` output is directly usable as Pydantic AI message_history |
| `agent.override()` for test-time substitution | Consider an `override()` context manager on Tract for test-time config swaps |

Sources: [Pydantic AI Docs](https://ai.pydantic.dev/), [GitHub](https://github.com/pydantic/pydantic-ai), [Agent API](https://ai.pydantic.dev/api/agent/), [Dependencies](https://ai.pydantic.dev/dependencies/), [Tools](https://ai.pydantic.dev/tools/), [Testing](https://ai.pydantic.dev/testing/), [Multi-Agent](https://ai.pydantic.dev/multi-agent-applications/), [Models](https://ai.pydantic.dev/models/), [Output Types](https://ai.pydantic.dev/output/), [Graph System](https://ai.pydantic.dev/graph/), [Logfire](https://ai.pydantic.dev/logfire/), [PyPI](https://pypi.org/project/pydantic-ai/)
