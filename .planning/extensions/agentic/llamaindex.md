# LlamaIndex -- Targeted Analysis (Dimensions A-D)

**Version**: llama-index-core 0.14.17 (March 12 2026) | **Stars**: 47.6k | **Downloads**: ~3M/month | **License**: MIT

---

## A. Core Abstractions & Extension Points

Two generations coexist. **Legacy (deprecated)**: AgentRunner (task dispatch) / AgentWorker (step execution) -- split enabled step-wise debugging and human-in-the-loop between reasoning steps. **Current**: Workflows + AgentWorkflow. Workflows are the foundation: `@step`-decorated async methods consume/emit Pydantic Events, validated at startup. AgentWorkflow is a pre-configured Workflow managing agents (FunctionAgent, ReActAgent, CodeActAgent -- all `BaseWorkflowAgent` subclasses), tool execution, and handoffs. QueryPipeline also deprecated in favor of Workflows.

**Key insight for tract**: LlamaIndex's `@step` + custom Events route control flow proactively; tract's 12 middleware events react to existing operations. Steps are composable routing primitives; tract hooks are lifecycle interceptors. Different purposes, but the event graph startup validation is a safety net tract lacks.

## B. State & Memory Model

**Memory** (new `Memory` class, replaces deprecated `ChatMemoryBuffer`): merges short-term chat history with long-term blocks -- `StaticMemory` (fixed), `FactExtractionMemory` (LLM-extracted facts), `RetrievalBasedMemory` (vector-backed). Assembled into structured XML injected before LLM calls. Token budget: `token_limit` (30k default) split by `chat_history_token_ratio` (0.7) between recent chat and long-term retrieval. Persistence via SQLite or PostgreSQL.

**Challenge to tract**: LlamaIndex assembles context by token-budgeted merge of heterogeneous memory sources. Tract assembles by DAG traversal with priority annotations and compression. LlamaIndex's `FactExtractionMemory` is more sophisticated (auto-extracts facts via LLM); tract's DAG provides history/rollback/branching that LlamaIndex memory entirely lacks. The `chat_history_token_ratio` knob -- splitting budget between recent and historical context -- is a simple idea tract's compile could adopt.

## C. Tool/Function Calling Design

`FunctionTool.from_defaults(fn)` auto-infers schema from signature + `Annotated` hints. `QueryEngineTool` wraps any query engine or agent as a tool. `LoadAndSearchToolSpec` splits large tool outputs into load + search steps to manage context overflow. `return_direct` flag bypasses reasoning loop. Tools attach at agent level.

**Challenge to tract**: `QueryEngineTool` (agent-as-tool wrapping) has no tract equivalent -- tract tools are leaf functions, not composable agent abstractions. `LoadAndSearchToolSpec` solves the same problem as tract's tool summarization differently: indexing large outputs for retrieval vs. compressing them. Both are valid strategies.

## D. Multi-Agent Patterns

Three patterns: (1) **AgentWorkflow** -- agents declare `can_handoff_to` lists, framework auto-generates handoff tools, routes conversation with full context. (2) **Orchestrator** -- sub-agents exposed as tools to a top-level agent. (3) **Custom planner** -- LLM outputs structured plan, user code executes imperatively.

**Challenge to tract**: LlamaIndex handoffs are behavioral (conversational routing) with no structural history. Tract's spawn/collapse is structural (DAG branches) preserving full provenance. LlamaIndex's agent-as-tool pattern (orchestrator wrapping sub-agents) cannot be expressed natively in tract -- would require a composed spawn + run + collapse operation. `can_handoff_to` is more ergonomic for delegation but discards provenance that tract's DAG preserves.

---

## Key Takeaways for Tract

1. **Token budget splitting**: `chat_history_token_ratio` splitting compile budget between recent commits and compressed history -- simple, effective.
2. **Fact extraction**: auto-extracting atomic facts via LLM into memory is more sophisticated than explicit commits. A middleware could bridge this.
3. **Agent-as-tool**: wrapping a child tract as a callable tool (spawn + run + collapse) is a missing composition pattern.
4. **LoadAndSearchToolSpec**: indexing large tool outputs (vs. tract's compression approach) -- consider offering both strategies.
