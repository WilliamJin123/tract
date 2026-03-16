# Cross-Framework Synthesis

**Frameworks analyzed**: LangGraph 1.1.2, AutoGen 0.7.5, DSPy 3.1.3, Pydantic AI 1.68.0, OpenAI Agents SDK 0.12.2, CrewAI 1.10.1, Semantic Kernel 1.41.0, Google ADK 1.27.1, LlamaIndex 0.14.17, Instructor 1.14.5

**Date**: March 15, 2026

---

## 1. Common Patterns

Patterns adopted by 3+ frameworks. These are industry defaults.

### Tool Definition: Decorators + Type Hints → Auto JSON Schema

**Unanimous.** Every framework derives tool schemas from Python function signatures and docstrings. Manual schema definition exists as escape hatch, not primary path.

| Framework | Pattern | Schema Source | DI for Runtime Context |
|---|---|---|---|
| LangGraph | `@tool` decorator | type hints + docstring | `InjectedState`, `InjectedStore` (hidden from LLM schema) |
| AutoGen | plain async functions | type hints + docstring | None — tools are standalone |
| DSPy | `dspy.Tool(fn)` wrapper | type hints + docstring | None |
| Pydantic AI | `@agent.tool` decorator | type hints + docstring (Google/NumPy/Sphinx) | `RunContext[Deps]` first param (hidden) |
| OpenAI Agents | `@function_tool` decorator | type hints + `griffe` docstring parser | `RunContextWrapper` first param (hidden) |
| CrewAI | `@tool` or `BaseTool` subclass | Pydantic `args_schema` or type hints | None |
| Semantic Kernel | `@kernel_function` | `Annotated[T, "desc"]` type hints | Kernel DI container |
| Google ADK | `FunctionTool(fn)` | type hints + docstring | `ToolContext` auto-injected (hidden) |
| LlamaIndex | `FunctionTool.from_defaults(fn)` | type hints + `Annotated` | None |
| Instructor | Pydantic model IS the schema | field types + validators | `context` param in validators |

**Tract status**: Has `ToolDefinition` + `@callable_tool`, but requires explicit `parameters` dict. Auto-schema from type hints is a gap.

### Multi-Agent via Handoff-as-Tool

The dominant multi-agent pattern: agent delegation surfaced as LLM tool calls (`transfer_to_X`).

| Framework | Handoff Mechanism | LLM-Driven? | Context Scoping |
|---|---|---|---|
| LangGraph | `create_supervisor` generates transfer tools | Yes | Shared message history, `output_mode` controls return |
| OpenAI Agents | `handoffs=[agent_a]` auto-generates tools | Yes | `input_filter` scopes what receiver sees |
| Google ADK | `transfer_to_agent()` via `ToolContext` | Yes | Single-tree hierarchy, shared state |
| LlamaIndex | `can_handoff_to` auto-generates tools | Yes | Full conversation context |
| CrewAI | `allow_delegation=True` on Agent | Yes (hierarchical) | Manager validates outputs |
| AutoGen | `handoffs=["agent_name"]` in Swarm | Yes | Shared flat history (leaky — #6123) |
| Pydantic AI | Agent called inside tool function | No (developer code) | Caller manages via `deps` |
| DSPy | Module composition in `forward()` | No (Python code) | Python scoping |

**Tract status**: `spawn()`/`collapse()` is developer-initiated structural branching, not LLM-driven runtime routing. Different paradigm — see Divergent Choices.

### Structured Output via Pydantic Models

| Framework | Mechanism | Retry on Validation Failure |
|---|---|---|
| Instructor | `response_model=PydanticModel` (core feature) | Yes — append error + retry |
| Pydantic AI | `output_type=PydanticModel` on Agent | Yes — `ModelRetry` exception |
| OpenAI Agents | `output_type=PydanticModel` on Agent | Yes — re-prompt with error |
| AutoGen | `response_format=PydanticModel` on client | No |
| DSPy | Signature types → automatic parsing | Yes — via Adapter layer |
| Semantic Kernel | Via plugin return types | No |

**Tract status**: No structured output at the protocol level. `LLMClient.chat()` returns raw strings/dicts.

### Provider Abstraction via Protocol

| Framework | Protocol | Multi-Provider | Fallback Chain |
|---|---|---|---|
| Pydantic AI | `Model` class hierarchy | 11 built-in + OpenAI-compat | `FallbackModel` |
| AutoGen | `ChatCompletionClient` protocol | OpenAI/Azure built-in, extensions | No |
| OpenAI Agents | `Model`/`ModelProvider` protocol | `MultiProvider` prefix routing | No |
| DSPy | `dspy.LM` (LiteLLM under hood) | 100+ via LiteLLM | No |
| LangGraph | LangChain `ChatModel` | 100+ via langchain-* packages | No |
| Semantic Kernel | AI Service connectors | OpenAI/Azure/HuggingFace | No |
| Google ADK | Model string → LiteLLM | 100+ via LiteLLM | No |

**Tract status**: `LLMClient` protocol with OpenAI/Anthropic implementations. No fallback, no multi-provider routing.

### Observability via Tracing

| Framework | Tracing | Standard | External Dashboard |
|---|---|---|---|
| AutoGen | Auto-instrumented OTel spans | OpenTelemetry | Jaeger, Langfuse, SigNoz |
| Pydantic AI | Logfire integration (3 lines) | OpenTelemetry | Logfire, 10+ alternatives |
| OpenAI Agents | Built-in trace/span system | Custom + OTel export | OpenAI dashboard, 20+ integrations |
| LangGraph | LangSmith integration | Proprietary | LangSmith SaaS |
| DSPy | MLflow + BaseCallback | MLflow | MLflow UI |
| Semantic Kernel | .NET-style telemetry | OpenTelemetry | Azure Monitor |

**Tract status**: No tracing. The DAG *is* a structural trace (queryable via `find()`, `compare()`, `log()`), but emits no OTel spans for integration with standard tooling.

---

## 2. Divergent Choices

### State Model: Flat History vs. Typed Dict vs. Semantic Memory vs. DAG

The most fundamental divergence. Four distinct approaches:

| Approach | Frameworks | Strengths | Weaknesses |
|---|---|---|---|
| **Flat message list** | AutoGen, OpenAI Sessions, SK ChatHistory | Simple, familiar, universal | No branching, no diffing, encapsulation leaks (AutoGen #6123) |
| **Typed dict with reducers** | LangGraph | Type-safe, reducer-driven merging, checkpointable | No cross-branch operations, can't express content-level ops |
| **Semantic memory** | CrewAI, LlamaIndex, ADK Memory | Great for "what do I know about X?", auto-extraction | No history, no rollback, no structural relationships |
| **Commit DAG** | **tract** | Full version control: branch, merge, diff, compress, rebase | More complex, requires understanding git-like concepts |

**Assessment**: Tract's DAG is the richest state model of any framework analyzed. No other framework offers cross-branch comparison, non-destructive compression, selective history inheritance, or merge conflict resolution. The closest is LangGraph's checkpoint time-travel (fork from prior state), but it lacks tract's ancestry semantics.

The risk: richness ≠ adoption. Flat history dominates because it's simple. Tract should preserve its power while offering simpler on-ramps (see Recommendations).

### Control Flow: Graph DSL vs. Python-as-Graph vs. Linear Loop

| Approach | Framework | Unit of Work | Pro | Con |
|---|---|---|---|---|
| Explicit graph DSL | LangGraph | Node execution | Visible, inspectable, checkpointable | Boilerplate, requires learning DSL, can't add nodes at runtime |
| Python control flow | DSPy, Pydantic AI | Module/Agent run | Natural, no new concepts, optimizable | Less inspectable without tracing |
| Event-driven steps | LlamaIndex Workflows | Step execution | Validated at startup, composable | Less mature, still evolving |
| Team orchestration | AutoGen, CrewAI | Agent turn | Natural for multi-agent | Coarse-grained, limited branching |

**Tract relevance**: Tract is a library, not a runtime. It doesn't own control flow. DSPy validates that Python-as-graph works — you don't need a graph DSL to build complex pipelines. Tract should continue to be "the context layer" that any control flow approach can use.

### Multi-Agent: Structural Branching vs. Runtime Delegation

| Approach | Framework | Developer Controls | LLM Controls |
|---|---|---|---|
| **Structural (DAG)** | tract | Branch creation, inheritance rules, merge | Nothing — developer-initiated |
| **Runtime delegation** | OpenAI, ADK, LangGraph, LlamaIndex | Agent definitions, handoff lists | Which agent handles each task |
| **Behavioral** | AutoGen, CrewAI | Team composition, process type | Speaker selection, delegation |

Tract and the runtime-delegation frameworks solve multi-agent from opposite ends:
- **Tract**: "What context does each agent see?" (structural, auditable, mergeable)
- **Others**: "Which agent handles this task?" (dynamic, flexible, opaque)

These are **complementary**, not competing. A production system needs both.

---

## 3. Anti-Patterns

### Shared Mutable State Without Isolation

AutoGen's `SocietyOfMindAgent` wraps an inner team as a single agent — but inner messages leak to outer context (#6123). LangGraph's shared-state graph has similar risks. SK's flat ChatHistory has no isolation mechanism.

**Tract avoidance**: `spawn()` with selective inheritance is architecturally immune to this. Don't regress toward shared mutable state.

### Heavy Dependency Trees

LangGraph requires `langchain-core`, `langgraph`, `langgraph-prebuilt`, plus provider-specific packages (`langchain-openai`, etc.). Version incompatibilities between ecosystem packages generate disproportionate GitHub issues (#4180, #6363). The v1.0.1→v1.0.2 breaking change without version constraints was a production incident.

**Tract avoidance**: Keep the thin dependency profile. Core depends on SQLAlchemy, Pydantic, tiktoken. Runner adds httpx, anthropic, openai. Don't add a LiteLLM dependency.

### Destructive History Operations

SK's `ChatHistoryTruncationReducer` drops messages permanently. LangGraph's state overwrite via reducers is non-recoverable without checkpoints. ADK's state overwrites are fire-and-forget.

**Tract avoidance**: Already avoided — compression is non-destructive (original commits remain in DAG). This is a genuine differentiator.

### Breaking API Rewrites

AutoGen's v0.2→v0.4 rewrite broke the entire API (different import paths, different abstractions, async-first). Community fragmented into AG2 fork. 50k-star project lost momentum.

**Tract avoidance**: Evolve incrementally. The commit-based DAG is sound architecture — it doesn't need an actor model or a graph DSL.

### Missing Test Infrastructure

LangGraph, OpenAI Agents SDK, Google ADK, and DSPy ship no first-party mock/replay utilities for testing agent behavior. Users must build their own or hit live APIs.

**Tract opportunity**: Ship a `MockLLMClient` / `ReplayLLMClient` and become the testing story for agent-adjacent code.

---

## 4. Prioritized Recommendations for Tract

### Must-Do (3 items)

#### 1. `MockLLMClient` + `ReplayLLMClient` for Testing

Every Tier 1 framework except Pydantic AI lacks proper test utilities. Pydantic AI's `TestModel`/`FunctionModel` and AutoGen's `ReplayChatCompletionClient` are the gold standard. Tract's LLM-dependent paths (gates, maintainers, compression) currently require live API calls or ad-hoc mocking.

**Files**: `src/tract/llm/testing.py` (new)

**Before** (current — no testing utilities):
```python
# Users must mock at protocol level
from unittest.mock import MagicMock
mock_client = MagicMock(spec=LLMClient)
mock_client.chat.return_value = "approved"
```

**After**:
```python
from tract.llm.testing import MockLLMClient, ReplayLLMClient

# Deterministic function-based mock
mock = MockLLMClient(lambda messages, **kw: "approved")
t = Tract(llm_client=mock)

# Replay recorded responses in sequence
replay = ReplayLLMClient(["response 1", "response 2", "tool_call:get_weather"])
t = Tract(llm_client=replay)
```

**Complexity**: Small. Protocol is already defined; just need two implementations.

#### 2. `@tract.tool` Auto-Schema Decorator

All 10 frameworks derive tool schemas from function signatures + docstrings. Tract's `ToolDefinition` requires explicit `parameters` dict — unnecessary boilerplate that every competitor has eliminated.

**Files**: `src/tract/toolkit/definitions.py` (modify), `src/tract/toolkit/decorators.py` (new)

**Before**:
```python
ToolDefinition(
    name="get_weather",
    description="Get weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    handler=get_weather_fn,
)
```

**After**:
```python
from tract.toolkit import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return fetch_weather(city)

# Auto-generates ToolDefinition with name, description, parameters from signature
# Explicit ToolDefinition remains as escape hatch
```

**Complexity**: Small-medium. Inspect function signature, extract docstring, generate JSON schema from type hints. `griffe` or manual introspection.

#### 3. Retry-with-Feedback on Gates

Three frameworks (Pydantic AI `ModelRetry`, DSPy `Assert`/`Suggest`, Instructor validation→retry) implement "reject with reason, retry with error context appended." Tract's `SemanticGate` blocks operations but doesn't retry with feedback — the operation simply fails.

**Files**: `src/tract/gate.py` (modify)

**Before**:
```python
# Gate blocks the operation — no retry, no feedback
gate_result = GateResult(passed=False, reason="Content too vague")
# Operation raises BlockedError
```

**After**:
```python
t.gate(
    "post_commit",
    criteria="Content must be specific and actionable",
    max_retries=2,  # NEW: retry with feedback
    retry_strategy="feedback",  # append gate reason to next attempt
)
# On failure: gate reason injected into LLM context, operation retried
# After max_retries exhausted: raises BlockedError as before
```

**Complexity**: Medium. Requires wiring gate rejection reason back into the operation's LLM call, which means gates need to participate in the retry loop rather than being pure middleware.

### Should-Consider (5 items)

#### 1. Tool Dependency Injection (LangGraph `InjectedState` Pattern)

LangGraph, Pydantic AI, OpenAI Agents, and Google ADK all inject runtime context into tools via typed parameters hidden from the LLM schema. This separates "what the LLM sees" from "what the tool needs."

**Files**: `src/tract/toolkit/definitions.py`, `src/tract/toolkit/executor.py`

```python
# Before: tool must access tract externally
def my_tool(query: str) -> str:
    # How does the tool access the tract's state?
    pass

# After: inject tract context
from tract.toolkit import tool, TractContext

@tool
def my_tool(query: str, ctx: TractContext) -> str:
    """Search documents."""  # ctx excluded from LLM schema
    history = ctx.tract.log(limit=5)
    return search(query, context=history)
```

**Complexity**: Medium. Need to detect annotated params, exclude from schema, inject at execution time.

#### 2. `FallbackClient` Wrapper

Pydantic AI's `FallbackModel` chains providers: try A, on HTTP error try B. Simple but high-value for production resilience.

**Files**: `src/tract/llm/fallback.py` (new)

```python
from tract.llm import FallbackClient

client = FallbackClient(
    OpenAIClient(model="gpt-4o"),
    AnthropicClient(model="claude-sonnet-4-5-20250929"),
)
# Tries OpenAI first; on error, falls back to Anthropic
```

**Complexity**: Small. Thin wrapper around existing protocol.

#### 3. Structured Output at Protocol Level

AutoGen, Pydantic AI, and OpenAI Agents all support `response_format=PydanticModel` at the client/agent level. Tract's `LLMClient.chat()` returns raw responses.

**Files**: `src/tract/llm/protocols.py` (modify)

```python
# Add optional response_format to chat()
class LLMClient(Protocol):
    def chat(self, messages, *, response_format=None, **kwargs): ...

# Usage
from pydantic import BaseModel
class Analysis(BaseModel):
    summary: str
    confidence: float

result = t.chat("Analyze this", response_format=Analysis)
# result is an Analysis instance
```

**Complexity**: Medium. Each client implementation needs response_format handling.

#### 4. Token Budget Splitting (LlamaIndex Pattern)

LlamaIndex's `chat_history_token_ratio` splits the token budget between recent messages and historical/compressed context. Simple knob, big impact.

**Files**: `src/tract/operations/compilation.py` (modify), `src/tract/models/config.py` (modify)

```python
t.configure(
    token_budget=4000,
    recent_ratio=0.7,  # 70% for recent commits, 30% for compressed history
)
```

**Complexity**: Small. The compile pipeline already manages token budgets; this adds a split point.

#### 5. OpenTelemetry Span Emission

AutoGen auto-instruments all operations with OTel spans. Pydantic AI integrates with Logfire (OTel-based). OpenAI Agents has its own tracing with OTel export. Tract's DAG is a structural trace, but emits nothing for standard observability tools.

**Files**: `src/tract/telemetry.py` (new), instrumentation in `tract.py`, `operations/*.py`

```python
# Optional — only activates if opentelemetry-api is installed
import tract
tract.configure_telemetry(tracer_provider=my_provider)

# Operations now emit spans:
# tract.commit (content_type, token_count, branch)
# tract.compile (strategy, token_budget, result_tokens)
# tract.merge (source_branch, target_branch, strategy)
```

**Complexity**: Medium. Need conditional import, span creation in hot paths, minimal overhead when disabled.

### Explicitly Reject (4 items)

#### 1. Graph DSL for Control Flow

**Pattern**: LangGraph's `StateGraph` with `add_node()`/`add_edge()`/`add_conditional_edges()`.

**Why reject**: DSPy proves that Python control flow (loops, conditionals, function calls) is sufficient for complex agent pipelines — and is optimizable. LangGraph itself created `create_react_agent` as an escape hatch from its own graph boilerplate. The manual `StateGraph` path requires 35+ lines and a routing function for what is fundamentally a while loop. Tract is a library; adding a graph DSL would make it a framework. Users who need graph-based orchestration should use LangGraph with tract as the context layer.

#### 2. Flat Shared Message History as Default State Model

**Pattern**: AutoGen teams, SK AgentGroupChat, and most frameworks default to a single shared message list.

**Why reject**: AutoGen's `SocietyOfMindAgent` encapsulation leak (#6123) demonstrates the failure mode — inner agent messages pollute outer context. SK's ChatHistory has no isolation mechanism. Tract's branch-based isolation is architecturally superior. The DAG is harder to learn but prevents an entire class of bugs. Don't simplify by regressing to shared mutable state.

#### 3. Framework-Owned Agent Loop as Core Abstraction

**Pattern**: Every framework except Instructor and DSPy owns the agent loop (call LLM → check tools → execute → loop).

**Why reject**: Tract is a library for context management, not an execution runtime. The runner/loop layer exists for convenience but should remain optional. Users who want a managed loop can use Pydantic AI, LangGraph, or OpenAI Agents SDK with tract providing the context. Tract's value is orthogonal to the loop — it's about what the LLM sees, not how the loop runs.

#### 4. Automatic Semantic Memory / RAG as Core Feature

**Pattern**: CrewAI's unified memory, LlamaIndex's FactExtractionMemory, ADK's Memory service, AutoGen's ChromaDB integration.

**Why reject**: Tract manages within-conversation context evolution (the compile window). RAG and semantic memory manage cross-conversation knowledge retrieval — a different concern. Adding vector stores, embedding models, and retrieval logic would double tract's dependency surface and blur its purpose. Users who need semantic memory should integrate a purpose-built tool (LlamaIndex, ChromaDB) alongside tract.

### Open Questions (3 items)

#### 1. Should Tract Offer a Stateless Compile Path?

Instructor's success comes from being a thin, stateless patch on existing clients. Tract requires instantiating a `Tract` object with SQLite storage, session management, etc. Could `tract.compile_messages(commits)` work as a pure function — no storage, no session — making tract composable as a building block?

**Why it's open**: This would lower the adoption barrier significantly but might sacrifice the DAG semantics that are tract's differentiator. Need to prototype whether a stateless compile is useful without branching/merging/compression.

#### 2. Should Tract Support LLM-Driven Dynamic Routing?

The handoff-as-tool pattern (OpenAI, ADK, LangGraph) is the dominant multi-agent primitive. Tract's `spawn()`/`collapse()` is developer-initiated. Should tract expose `spawn()` as an LLM-callable tool, enabling dynamic delegation where the LLM decides which specialized context branch to create?

**Why it's open**: This bridges tract's structural model with the runtime delegation model but risks making tract an orchestration framework rather than a context library. The right answer might be a recipe/example rather than a core feature.

#### 3. Could Tract Serve as a Persistence Layer for DSPy Optimization?

DSPy's compiled programs (optimized prompts, few-shot examples) are saved as JSON files with no versioning. Tract's DAG could store optimization progression — each compile producing a commit, enabling rollback and diff between optimization runs.

**Why it's open**: This is a genuine integration opportunity (not competitive overlap), but the value depends on DSPy adoption patterns. Need to prototype a `tract-dspy` bridge to validate whether the overhead justifies the versioning benefit.

---

## 5. Gaps

### Where No Framework Excels

**Testing agent behavior deterministically.** Only Pydantic AI (TestModel/FunctionModel) and AutoGen (ReplayChatCompletionClient) ship first-party test utilities. LangGraph, OpenAI Agents, DSPy, CrewAI, ADK, SK, LlamaIndex — all punt testing to the user. This is the biggest gap in the ecosystem. A framework that makes agent testing as easy as unit testing regular code would have a significant advantage.

**Context evolution audit trail.** Every framework treats conversation history as a mutable buffer to be truncated, summarized, or discarded. None provide structural versioning (branch, diff, merge) of context state. Tract is the only system that treats context as a first-class versioned artifact. This is tract's moat.

**Non-destructive compression.** SK truncates, LangGraph overwrites, CrewAI consolidates, LlamaIndex replaces — all destroy the original content. Tract's compression preserves the original commits in the DAG. No other framework offers this.

**Cross-agent context isolation with merge semantics.** AutoGen's encapsulation leak (#6123) is the canonical failure. LangGraph uses subgraph namespaces. Most frameworks share flat history. Tract's `spawn(include_tags=..., exclude_tags=...)` with `MergeStrategy` is the most sophisticated isolation + reconciliation mechanism in the ecosystem.

### Where Tract Is Already Ahead

| Capability | Tract | Closest Competitor | Gap |
|---|---|---|---|
| Context versioning (branch/merge/diff) | Full DAG | LangGraph checkpoints (fork only) | Large |
| Non-destructive compression | Preserves originals | None | Unique |
| Selective context inheritance | `spawn(include_tags, filter_func)` | OpenAI `input_filter` (shallow) | Significant |
| Content-level operations | APPEND, EDIT with priorities | Everyone: append-only flat list | Large |
| Pure-core test coverage | 912 tests, zero runner deps | None measure this | Unique |

### Greenfield Opportunity

**Tract as the context layer for any framework.** No framework owns context management well. They all default to flat message lists because context is "someone else's problem." Tract could position itself as the universal context backend — the way SQLAlchemy is the universal database layer. The integration path: `CompiledContext.to_openai()` / `.to_anthropic()` outputs slot directly into any framework's message input. Make this the primary on-ramp.

**Agent test infrastructure built on DAG history.** Tract stores the full context evolution. This is inherently a test artifact — you can assert on exact sequences of operations, replay from any point, compare branches. No framework offers this. A `tract.testing` module that provides snapshot testing, replay testing, and regression testing for agent behavior would be genuinely novel.

---

## Appendix: Additional Framework Analyses

- **Smolagents** (HuggingFace): Full analysis in [`smolagents.md`](smolagents.md). Code-first agent paradigm (LLM writes Python instead of JSON tool calls). Key unique insight: code generation outperforms JSON tool-calling on benchmarks. Six sandbox options for safe execution. Flat memory with no structural state management — sharpest contrast with tract's DAG.
- **Anthropic Agent SDK** (`claude_agent_sdk`): Full analysis in [`anthropic_agent_sdk.md`](anthropic_agent_sdk.md). Not a general-purpose agent framework — it's a Claude Code orchestration wrapper. Key tract-relevant insight: **prompt caching** (prefix-based, 0.1x cost for cache reads) creates a natural integration point where tract's compile could place `cache_control` breakpoints at the stable/volatile boundary. Extended thinking with interleaved mode and computer use tools are unique to Anthropic.
- **Mastra** (TypeScript): Workflow primitives (`step()`, `then()`, `parallel()`) are more composable than Python equivalents. The `suspend()`/`resume()` for human-in-the-loop is cleaner than LangGraph's `interrupt()`. Worth watching if tract ever considers a TypeScript port.
- **ControlFlow**: Prefect-backed. Interesting for treating agent tasks as observable workflow steps, but Prefect dependency is heavy.
- **Haystack**: Pipeline-based. Similar to LlamaIndex's deprecated QueryPipeline. No novel agent patterns.
- **Marvin**: Lightweight extraction (similar to Instructor). No novel patterns beyond Instructor's analysis.
