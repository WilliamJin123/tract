# Smolagents -- Targeted Analysis (Dimensions A-D)

**Version**: 1.24.0 (January 2026) | **Stars**: ~25.5k | **License**: Apache 2.0 | **Origin**: Hugging Face

Smolagents is HuggingFace's minimalist agent library built on one contrarian thesis: **agents should write Python code, not JSON tool calls**. The entire library's agent logic fits in ~1,000 lines. The code-first paradigm is the key differentiator and the only thing worth studying here.

---

## A. Core Abstractions & Extension Points

Three core abstractions: **MultiStepAgent** (ReAct loop base class), **Model** (LLM wrapper), **Tool** (callable with typed schema).

Two agent subclasses implement the paradigm split:

- **CodeAgent**: LLM generates Python code snippets that call tools as functions. The code is parsed and executed, with results fed back as observations.
- **ToolCallingAgent**: LLM generates JSON tool calls (standard OpenAI-style). Structured, validated, predictable.

The **unit of work** is a ReAct step: think -> act -> observe. Agents run multi-step loops up to `max_steps` (default 20). Periodic planning steps can be enabled via `planning_interval`.

```python
from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def search_docs(query: str) -> str:
    """Search internal docs. Args: query: search string."""
    return db.search(query)

agent = CodeAgent(tools=[search_docs], model=InferenceClientModel())
agent.run("Find docs about authentication and summarize them")
```

What the LLM actually generates (CodeAgent):
```python
results = search_docs("authentication")
summary = "\n".join(results[:3])
final_answer(summary)
```

This is the core insight: the LLM composes tools using native Python -- loops, conditionals, variable assignment, string manipulation -- rather than emitting one JSON blob per tool call. Research cited (arXiv 2402.01030) shows code actions outperform JSON on agentic benchmarks.

**Extension points**: Custom `PromptTemplates` (system, planning, managed agent, final answer), `step_callbacks` for per-step hooks, `final_answer_checks` for output validation, custom `PythonExecutor` implementations. Agents are serializable to Hub (`push_to_hub`/`from_hub`).

**Composability**: High for CodeAgent (Python is the composition language). Low for ToolCallingAgent (one JSON call at a time, no chaining).

---

## B. State & Memory Model

Memory is an **ordered list of MemoryStep objects** stored in `AgentMemory`:

- `SystemPromptStep` -- the system prompt
- `TaskStep` -- the user task
- `ActionStep` -- LLM output + code/tool execution + observations (one per ReAct step)
- `PlanningStep` -- periodic planning outputs

This is a **flat, append-only log**. No branching, no DAG, no structural relationships between steps. Memory is scoped to a single agent run; calling `run(reset=True)` clears it. Calling `run(reset=False)` continues the conversation.

Memory manipulation is imperative: step callbacks receive the agent and can mutate `agent.memory.steps` directly. The documented pattern for managing context windows is manual pruning in callbacks (e.g., removing old screenshots to save tokens).

**Persistence**: Agents can be serialized to disk/Hub via `save()`/`push_to_hub()`, but this saves the agent _definition_ (tools, prompts, config), not the runtime memory. There is no built-in persistence for conversation state across process restarts.

**Challenge tract**: Smolagents memory is structurally primitive -- a flat list with manual mutation. No commits, no branches, no merge, no ancestry traversal, no priority annotations. Tract's DAG provides _structural_ state management (branch, diff, rebase, selective spawn) where smolagents provides _none_. The code-first paradigm does not affect state management; it only affects how tool calls are expressed within a single step. Tract and smolagents operate at entirely different layers -- tract manages what _context_ the LLM sees across interactions, while smolagents manages what _actions_ the LLM takes within a single run.

---

## C. Tool/Function Calling Design

Two ways to define tools:

**Decorator** (quick): Function with type hints + docstring with `Args:` section. The `@tool` decorator extracts schema automatically.

```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city. Args: city: The city name."""
    return weather_api.get(city)
```

**Class** (flexible): Subclass `Tool`, define `name`, `description`, `inputs` dict, `output_type`, and `forward()` method. Useful when the tool needs initialization state.

**Dual mode trade-offs**: CodeAgent tools become callable Python functions in the execution environment. ToolCallingAgent tools become JSON schemas. The same `Tool` object works with both agent types -- the agent type determines the calling convention.

**Code execution sandboxing** is the critical concern. Five options, escalating security:

1. **LocalPythonExecutor** (default): AST-based interpreter with import allowlists and operation caps. NOT a security boundary -- can be bypassed.
2. **Docker**: Jupyter kernel in a container. Memory/CPU limits, `cap_drop=ALL`.
3. **E2B**: Cloud sandbox. Simple setup (`executor_type="e2b"`), but no multi-agent support.
4. **Modal**: Cloud sandbox with similar trade-offs.
5. **Blaxel**: Hibernated VMs, <25ms cold start, scales to zero.
6. **Wasm** (Pyodide+Deno): WebAssembly sandbox, strongest isolation model.

**Challenge tract**: Tract's toolkit system (`@tool` decorator, `ToolDefinition`, `ToolExecutor`) is structurally similar to smolagents' tool definition. The key difference: tract tools are wired into the context DAG (tool call results become commits, can be compressed, annotated, branched). Smolagents tools are ephemeral -- results exist only as step observations in the flat memory log. Tract has no equivalent to the code execution sandbox concern because tract does not execute arbitrary LLM-generated code.

---

## D. Multi-Agent Patterns

Multi-agent is **hierarchical delegation**. Any agent with `name` and `description` can be passed as a `managed_agents` argument to a manager agent:

```python
web_agent = CodeAgent(
    tools=[WebSearchTool()], model=model,
    name="web_search_agent",
    description="Runs web searches. Give it your query as an argument."
)
manager = CodeAgent(tools=[], model=model, managed_agents=[web_agent])
manager.run("Who is the CEO of Hugging Face?")
```

The manager calls managed agents like tools -- in CodeAgent, literally `result = web_search_agent("query")` in generated code. Each agent has **isolated memory and tools**, which is the stated benefit: a code-generation agent's context is not polluted by a web-search agent's scraped pages.

Communication is **call-return only**: manager sends a task string, managed agent returns a result string. No shared state, no event bus, no pub/sub. The `provide_run_summary` flag controls whether managed agents summarize their run for the manager.

Limitation: When using sandboxed executors (E2B, Docker), multi-agent requires running the _entire_ agent system inside the sandbox because managed agent calls require LLM API credentials. The simple `executor_type=` parameter only sandboxes code snippets, not the orchestration layer.

**Challenge tract**: Tract's `spawn()` creates a child branch that _inherits_ parent context (with selective filtering via `include_tags`, `exclude_tags`, `filter_func`). `collapse()` merges child work back. This is fundamentally richer than smolagents' call-return pattern: tract children share structural lineage with their parent and can merge results back into the parent's DAG with conflict resolution. Smolagents' multi-agent is fire-and-forget delegation with string results -- there is no shared context graph, no merge semantics, no selective inheritance. However, smolagents' isolation-by-default is a valid design choice for preventing context pollution, which tract would need to achieve through selective spawn filtering.
