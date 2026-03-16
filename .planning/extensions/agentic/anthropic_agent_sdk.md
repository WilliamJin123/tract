# Anthropic Agent SDK — Tier 2 Analysis (Dimensions A-D)

**Package**: `claude-agent-sdk` v0.1.48 (March 7, 2026) | **Python**: >=3.10 | **License**: MIT
**Also relevant**: `anthropic` SDK v0.84.0 (raw API client), Anthropic Messages API (tool_use, prompt caching, extended thinking)

> Anthropic does NOT ship a general-purpose agent framework like OpenAI Agents SDK. Instead they provide (1) a low-level API client (`anthropic`), (2) a Claude Code orchestration SDK (`claude-agent-sdk`), and (3) opinionated documentation on agent patterns. The "framework" is the Messages API itself -- tool_use, prompt caching, extended thinking, and computer use are primitives you compose yourself.

---

## A. Core Abstractions & Extension Points

**Two-tier architecture:**

1. **`anthropic` SDK** -- thin API client. `client.messages.create()` with `tools=[]`, `thinking={}`, `cache_control={}`. No agent loop, no state, no orchestration. You build the loop.

2. **`claude-agent-sdk`** -- Claude Code orchestration layer (wraps the CLI). Two entry points:
   - `query(prompt, options)` -- stateless one-shot, returns `AsyncIterator[Message]`
   - `ClaudeSDKClient` -- stateful bidirectional session with `connect/query/receive_response/interrupt/disconnect`

**Key difference from OpenAI Agents SDK**: OpenAI provides `Agent`, `Runner`, `Handoff`, `GuardRail` as first-class classes with declarative composition. Anthropic provides *primitives* (tool_use, thinking, caching) and lets you build the loop. The Claude Agent SDK is specifically a Claude Code wrapper, not a general agent framework -- it assumes file/bash/edit tools and a coding-agent mental model.

**Extension points:**
- **Hooks** (`PreToolUse`, `PostToolUse`, `Stop`, `SubagentStart`, `PreCompact`, etc.) -- intercept agent behavior at ~10 lifecycle events
- **Custom tools** via `@tool` decorator + `create_sdk_mcp_server()` -- in-process MCP servers, no subprocess overhead
- **`can_use_tool` callback** -- custom permission handler per tool invocation
- **`AgentDefinition`** -- define named subagents with independent tools/model/prompt

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions

@tool("search_docs", "Search documentation", {"query": str})
async def search_docs(args):
    return {"content": [{"type": "text", "text": f"Results for: {args['query']}"}]}

server = create_sdk_mcp_server(name="docs", version="1.0.0", tools=[search_docs])
options = ClaudeAgentOptions(
    mcp_servers={"docs": server},
    allowed_tools=["mcp__docs__search_docs"],
    max_turns=20,
    max_budget_usd=0.50,
)
```

**Composability**: Limited. No declarative graph, no workflow DSL. Composition happens through the agent loop itself and subagent spawning. The `AgentDefinition` dict is the closest thing to declarative multi-agent config.

---

## B. State & Memory Model

**Claude Agent SDK**: Session state lives in the Claude Code CLI process. `ClaudeSDKClient` maintains conversation across `query()` calls within a session. `list_sessions()` and `get_session_messages()` provide persistence/retrieval. `enable_file_checkpointing=True` allows rewinding file state to specific message IDs.

**Anthropic Messages API**: Stateless. Every request sends the full conversation history. There is no server-side session.

**Prompt Caching** (the unique Anthropic primitive):
- Cache breakpoints via `cache_control: {"type": "ephemeral"}` on content blocks
- 5-minute TTL (default) or 1-hour TTL (2x cost)
- Cache reads cost 0.1x base input price -- massive savings for long conversations
- Up to 4 explicit breakpoints for content that changes at different rates
- **Automatic caching**: single `cache_control` at request level, breakpoint auto-advances as conversation grows
- Minimum cacheable: 1024-4096 tokens depending on model
- Tools, system prompts, and message history are all cacheable
- Changes at any level invalidate that level and all subsequent levels

**Challenge tract**: Prompt caching is deeply complementary to tract's compile. Tract's `ContextCompiler` produces the message array; Anthropic's cache can then avoid re-processing the stable prefix. The key interaction: tract's SKIP/PINNED priority annotations control *what* goes into the compiled output, while cache breakpoints control *how much the API re-processes*. A tract integration could place `cache_control` breakpoints at the boundary between historical (stable) and recent (changing) compiled content, getting both semantic compression (tract) and prefix caching (Anthropic) simultaneously. The 5-minute TTL aligns well with interactive agent sessions. Tract's DAG is strictly more expressive than Anthropic's flat message array -- branching, merging, and selective spawn inheritance have no cache equivalent.

---

## C. Tool/Function Calling Design

**Messages API tool definition**:
```python
tools=[{
    "name": "get_weather",
    "description": "Get current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    },
    "strict": True  # Optional: guaranteed schema conformance
}]
```

**Tool execution**: Claude returns `stop_reason: "tool_use"` with `tool_use` content blocks. You execute and return `tool_result` blocks. The agent loop is yours to write.

**`strict: true`** (Structured Outputs): Guarantees tool call inputs match schema exactly. Production-critical for agents where invalid params cause failures.

**Computer use tools** (beta): `computer_20251124` (screenshot + mouse/keyboard), `text_editor_20250124`, `bash_20250124`. Anthropic-defined tool types -- you declare them, Anthropic provides the schema, Claude generates the actions. Unique to Anthropic; OpenAI has no equivalent.

**Claude Agent SDK tools**: Built-in `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebSearch`, `WebFetch`, `Agent` (subagent spawning), `NotebookEdit`. Custom tools via `@tool` decorator creating in-process MCP servers.

**Extended thinking + tools**: Claude can reason between tool calls with interleaved thinking (beta `interleaved-thinking-2025-05-14`). Thinking blocks must be preserved and passed back in subsequent requests. Budget applies across all thinking blocks in a turn.

**Challenge tract**: Tract's toolkit system (`@tool` decorator, `ToolDefinition`, `ToolExecutor`, `ToolProfile`) is structurally similar to the Claude Agent SDK's `@tool` + `create_sdk_mcp_server()` pattern. Key difference: tract's toolkit integrates with the DAG (tool results become commits, tool call summarization compresses results), while the Claude Agent SDK's tools are ephemeral within the message stream. Tract adds value by persisting tool results as versioned content that survives compression and can be branched/merged.

---

## D. Multi-Agent Patterns

**Claude Agent SDK**:
```python
options = ClaudeAgentOptions(
    agents={
        "researcher": AgentDefinition(
            description="Research information from the web",
            prompt="You are a research expert.",
            tools=["WebSearch", "WebFetch"],
            model="sonnet"
        ),
        "analyst": AgentDefinition(
            description="Analyze complex data",
            prompt="You are a data analyst.",
            tools=["Read", "Bash"],
            model="opus"
        ),
    },
    allowed_tools=["Agent"],
)
```

The main agent spawns subagents via the `Agent` tool. Each subagent gets independent tools, model, and system prompt. Hooks include `SubagentStart` and `SubagentStop` events. This is an orchestrator-worker pattern where the main Claude instance decides when to delegate.

**Anthropic's documented patterns** (from "Building Effective Agents" blog):
1. Prompt chaining (sequential with gates)
2. Routing (classify and dispatch)
3. Parallelization (sectioning + voting)
4. Orchestrator-workers (dynamic delegation)
5. Evaluator-optimizer (generate + critique loop)

None of these are SDK abstractions -- they are patterns you implement with the Messages API.

**No handoff primitive**: Unlike OpenAI's `Handoff` class, Anthropic has no first-class agent-to-agent transfer. Subagents in the Claude Agent SDK are fire-and-forget tasks, not conversational handoffs.

**Challenge tract**: Tract's `spawn()` creates a child branch that inherits filtered context (via `include_tags`, `exclude_tags`, `include_types`, `filter_func`). `collapse()` merges the child's work back. This is structurally richer than the Claude Agent SDK's `AgentDefinition` which provides isolated sessions without shared history or merge-back semantics. Tract could model the Claude Agent SDK's subagent pattern as spawn (filtered context) -> independent work -> collapse (merge results), adding persistence, conflict resolution (`MergeStrategy`), and audit trail that the Claude Agent SDK lacks entirely.

---

## Summary: What's Unique About Anthropic's Approach

| Aspect | Anthropic | OpenAI Agents SDK |
|--------|-----------|-------------------|
| Philosophy | Primitives, not framework | Declarative framework |
| Agent loop | You build it | `Runner.run()` |
| State | Stateless API + CLI sessions | `RunContext` + handoffs |
| Caching | Prompt caching (prefix-based, TTL) | None equivalent |
| Thinking | Extended thinking with budget | No equivalent |
| Computer use | Native tool type | No equivalent |
| Multi-agent | `AgentDefinition` + `Agent` tool | `Handoff` class |
| Tool schema | `strict: true` for guarantees | `strict: True` (similar) |
| MCP | First-class, in-process servers | Not integrated |

**The tractable insight**: Anthropic's prompt caching is the single most relevant primitive for tract. A tract adapter that emits `cache_control` breakpoints at the stable/volatile boundary of compiled context would deliver compounding savings -- tract compresses semantically, caching avoids re-tokenization of the stable prefix. No other framework in this analysis has an equivalent interaction point.
