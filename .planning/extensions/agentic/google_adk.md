# Google Agent Development Kit (ADK) -- Targeted Analysis

**Version**: 1.27.1 (March 13, 2026) | **Maturity**: Stable (50+ releases, bi-weekly cadence, 18.4k GitHub stars, 256 contributors, Apache 2.0)
**Source**: `google/adk-python` | **Docs**: google.github.io/adk-docs

---

## A. Core Abstractions & Extension Points

ADK's hierarchy: `BaseAgent` -> three branches:

1. **LlmAgent** (aliased as `Agent`): LLM-powered reasoning. Properties: `model`, `instruction`, `tools`, `sub_agents`, `output_key`. Non-deterministic.
2. **Workflow Agents**: `SequentialAgent`, `ParallelAgent`, `LoopAgent`. Deterministic orchestrators with no LLM calls -- pure control flow.
3. **Custom Agents**: Subclass `BaseAgent`, override `_run_async_impl()`.

Extension points: callbacks (lifecycle hooks), plugins (pre-packaged behaviors), artifacts, and the `ToolContext` injection pattern. Instructions support `{state_key}` template substitution from session state.

**Tract comparison**: ADK's unit of work is the *agent invocation*; tract's is the *commit*. ADK composes agents hierarchically; tract composes context via DAG branches. ADK's `instruction` templating is roughly analogous to tract's directives, but ADK's is agent-scoped while tract's is context-scoped.

## B. State & Memory Model

Three-tier design:

- **Session**: Conversation-scoped event log (`Session.events`). Managed by `SessionService` (InMemory, Database, VertexAI backends).
- **State**: Key-value scratchpad on sessions. Four prefixes: unprefixed (session), `user:` (cross-session per user), `app:` (global), `temp:` (invocation-only, discarded after). Mutations tracked via `EventActions.state_delta`.
- **Memory**: Long-term cross-session archive. `add_session_to_memory()` ingests completed sessions. `search_memory()` retrieves via keyword (InMemory) or semantic search (VertexAiMemoryBankService). Two built-in tools: `PreloadMemory`, `LoadMemory`.

Values must be JSON-serializable primitives. No structured versioning.

**Tract challenge**: ADK state is a flat key-value store with scope prefixes -- no history, no diffing, no branching. Tract's DAG gives you `compare()`, `rebase()`, `merge()`, and full ancestry traversal. ADK's `temp:` prefix solves a narrow problem that tract handles with branch-scoped commits. The Memory service adds cross-session recall that tract doesn't target (tract manages *within* a context window, not across conversations). This is a genuine gap -- but also a different design goal.

## C. Tool/Function Calling Design

`FunctionTool` wraps plain Python functions. Type hints become parameter schemas; docstrings become descriptions. `ToolContext` is injected automatically (excluded from LLM schema) and provides state access, `transfer_to_agent()`, artifact I/O, and `search_memory()`.

Built-in tools: Google Search, Code Execution (Vertex sandbox), RAG retrieval. Also supports OpenAPI spec tools and MCP tool servers.

Return convention: structured dicts with success/failure indication, with instructions to guide LLM interpretation of return values.

**Tract challenge**: ADK's function-to-tool conversion is nearly identical to tract's `@tool` decorator pattern. Both auto-extract schemas from type hints. ADK's `ToolContext` is richer (state + agent transfer + artifacts), while tract's `ToolContext` is lighter (focused on commit/compile). ADK's built-in Google Search and Code Execution are genuine value-adds that tract doesn't replicate -- but they're Google Cloud lock-in points.

## D. Multi-Agent Patterns

Three communication patterns:

1. **Shared state**: Agents in a workflow read/write session state keys. Sequential agents pass data via `output_key` -> `{state_key}` template reads.
2. **Agent transfer**: LLM generates `transfer_to_agent(agent_name='X')`. Framework intercepts via `AutoFlow`, calls `root_agent.find_agent()`, switches execution. Requires agents to be in the same hierarchy.
3. **AgentTool**: Wrap an agent as a tool in another agent's `tools` list. Synchronous call-and-return. State/artifact changes propagate back.

Workflow agents provide the structural patterns: Sequential (pipeline), Parallel (fan-out with shared state, unique keys to avoid races), Loop (iterate until `max_iterations` or `exit_loop` tool call). Nestable -- e.g., ParallelAgent inside SequentialAgent for fan-out/fan-in.

Constraint: an agent instance can only have one parent (single-tree hierarchy, no DAG).

**Tract challenge**: ADK's `transfer_to_agent` is dynamic delegation -- the LLM decides routing at runtime. Tract has no equivalent; `spawn()` is developer-initiated, not LLM-initiated. ADK's `ParallelAgent` with fan-out/fan-in has no direct tract analog. However, ADK's single-tree constraint means no shared sub-agents or cross-hierarchy references -- tract's DAG allows arbitrary branch topologies. ADK's `LoopAgent` with `exit_loop` is a pattern tract could support via staged workflows but doesn't natively provide. The agent-transfer mechanism is ADK's most distinctive feature and represents a fundamentally different model from tract's branch-based composition.

---

## What's Unique

**Google's distinctive contribution is treating agent *routing* as a first-class LLM decision.** The `transfer_to_agent` mechanism lets the model dynamically choose which specialized agent handles a task, turning the agent hierarchy into a live routing graph. Combined with the three workflow agents, this creates a "compose agents like functions" paradigm that's structurally simple but powerful.

The Vertex AI integration (persistent sessions, semantic memory via RAG, code execution sandbox) creates a strong cloud-native story -- but also the primary lock-in vector. The state prefix system (`app:`, `user:`, `temp:`) is a pragmatic solution to scope management that avoids the complexity of tract's DAG at the cost of losing history and diffing.

**Key takeaway for tract**: ADK validates that agent orchestration needs both *structural* patterns (sequential/parallel/loop) and *dynamic* routing (LLM-driven transfer). Tract's `spawn()`/`collapse()` handles the structural side but lacks dynamic delegation. If tract enters the agent orchestration space, the transfer mechanism is the gap to study.
