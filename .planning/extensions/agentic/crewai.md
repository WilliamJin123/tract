# CrewAI -- Targeted Analysis (Dimensions A-D)

**Version**: 1.10.1 (March 4 2026) | **Stars**: 46.2k | **License**: MIT

---

## A. Core Abstractions & Extension Points

Four primary classes: **Agent** (role/goal/backstory persona triple), **Task** (description + expected_output contract), **Crew** (agent team + process strategy), **Flow** (event-driven orchestrator above Crews).

**Unit of work** is the Task -- declaratively defined with explicit `expected_output` strings that act as completion criteria. Tasks wire dependencies via `context=[other_task]`, forming an implicit DAG. Agents are assigned per-task, so the same agent can serve multiple tasks with different tool sets.

**Declarative-first**: YAML config for agents/tasks (`agents.yaml`, `tasks.yaml`), `@CrewBase` decorator hydrates from config. Imperative fallback via direct Python construction.

**Extension points**: `step_callback` / `task_callback` / `before_kickoff` / `after_kickoff` hooks on Crew; `guardrail` callables on Task for output validation with retry; `@tool` decorator or `BaseTool` subclass for capabilities; custom `Process` implementations; `StorageBackend` protocol for memory.

**Composability**: Flows compose Crews as atomic steps in larger workflows. `@start`/`@listen`/`@router` decorators build an event graph. `or_()` and `and_()` combinators express fan-in logic. This is a two-tier architecture: Crews handle agent collaboration, Flows handle workflow orchestration.

**Key insight for tract**: CrewAI's `expected_output` string on every Task is a lightweight contract that guides both the agent and the guardrail system. Tract has no equivalent per-commit completion criterion -- the closest analog would be a directive, but those are scoped to the tract, not to individual operations.

## B. State & Memory Model

**Unified Memory** (recently redesigned): single `Memory` class with composite scoring (semantic 0.5 + recency 0.3 + importance 0.2). Replaces the older short-term/long-term/entity split. LLM-powered on save (infers scope, categories, importance) and on deep recall (query decomposition). Storage: LanceDB by default, pluggable via `StorageBackend` protocol.

**Hierarchical scoping**: memories organize into a tree (`/project/alpha/decisions`). `MemoryScope` restricts to subtree; `MemorySlice` enables cross-branch reads. Private memories support multi-tenant isolation.

**State flow between tasks**: after each task, the crew auto-extracts atomic facts into memory; before each task, relevant context is recalled and injected. This is implicit -- no manual wiring needed.

**Flow state**: Pydantic-backed `FlowState` (structured) or dict (unstructured), persisted via `@persist` decorator (SQLite default). State is the data bus between Flow steps.

**Challenge to tract**: CrewAI's memory is a semantic search layer -- content-addressed, importance-weighted, with LLM-assisted consolidation. Tract's DAG is history-addressed -- append-only commits with explicit parentage, branches, and merge operations. CrewAI's model is better for "what do I know about X?" queries; tract's is better for "how did this context evolve?" and rollback/branching workflows. The consolidation threshold (0.85 similarity triggers dedup) is interesting -- tract's compression is explicit and user-initiated, while CrewAI's is automatic and continuous.

## C. Tool/Function Calling Design

Two patterns: **`BaseTool` subclass** (Pydantic schema via `args_schema`, implement `_run`) or **`@tool` decorator** (infers schema from function signature and docstring).

```python
@tool("search_web")
def search_web(query: str) -> str:
    """Search the web for current information."""
    return requests.get(f"https://api.example.com/search?q={query}").text
```

Tools attach at agent level (`Agent(tools=[...])`) or overridden per-task (`Task(tools=[...])`). Task-level tools replace (not extend) the agent's tool set for that task. Built-in caching with optional `cache_function` for custom invalidation logic. Async tools supported via `async def _run`.

**Separate `function_calling_llm`**: agents can use a cheaper/faster model specifically for tool invocation decisions, distinct from the reasoning LLM. This is a pragmatic cost optimization.

**Challenge to tract**: Tract's toolkit system (`ToolDefinition` + `ToolExecutor`) is structurally similar but sits behind the runner boundary. CrewAI's per-task tool override is more granular than tract's per-agent assignment. The `function_calling_llm` split has no tract equivalent -- tract uses a single LLM per tract instance. Both use Pydantic for schema; CrewAI's `@tool` decorator is slightly more ergonomic than tract's `@callable_tool`.

## D. Multi-Agent Patterns

**Process.sequential**: linear task chain, output feeds forward. Simple, predictable.

**Process.hierarchical**: manager agent (auto-generated or custom) dynamically delegates tasks to worker agents based on capabilities. Manager validates outputs before proceeding. Tasks are not pre-assigned -- the manager decides at runtime.

**Process.consensual** (planned, not implemented): collaborative decision-making.

**Delegation**: controlled by `allow_delegation=True` on Agent. When enabled, agents can hand off work to peers. In hierarchical mode, delegation is the primary mechanism.

**Flows as multi-crew orchestration**: a Flow can launch multiple Crews in parallel branches, merge results via `and_()`, route conditionally via `@router`. This is the real multi-agent pattern -- Crews are the micro level, Flows are the macro level.

**Challenge to tract**: Tract's `spawn()` creates a child branch inheriting (optionally filtered) context; `collapse()` merges results back. This maps loosely to CrewAI's hierarchical process where a manager dispatches subtasks and collects results. However:
- Tract's spawn/collapse is structural (DAG branches), CrewAI's is behavioral (agent delegation).
- Tract has no role-based agent personas -- context is shared via the DAG, not via agent identity.
- CrewAI's Flow `@router` decorator is analogous to tract's stage transitions with `t.configure()` per stage, but CrewAI's is more expressive (arbitrary conditional branching vs. linear stage progression).
- CrewAI's `and_()` combinator for fan-in has no tract equivalent -- collapse() is always one child merging back to parent.

The hierarchical process with runtime delegation is CrewAI's most distinctive pattern. Tract would need to model this as a parent tract spawning children with filtered context and custom directives per child -- possible but verbose.

---

## Key Takeaways for Tract

1. **Expected output contracts**: per-task completion criteria are cheap and powerful. Tract could benefit from optional per-commit or per-directive success criteria.
2. **Unified memory with composite scoring**: the semantic+recency+importance blend is more sophisticated than tract's priority annotations (SKIP/NORMAL/IMPORTANT/PINNED). Tract's compile could optionally weight by a similar composite.
3. **Two-LLM strategy**: separating reasoning LLM from function-calling LLM is a practical cost optimization tract should consider.
4. **Flow-level orchestration**: the `@start`/`@listen`/`@router` decorator pattern for composing Crews is elegant. Tract's profiles/stages could learn from this event-driven wiring.
5. **Automatic fact extraction**: CrewAI auto-extracts atomic facts from task outputs into memory. Tract requires explicit commits -- an auto-commit middleware could bridge this gap.
