# Agentic Framework Research Prompt

## Objective

Research the API design and programmatic patterns of leading agentic AI frameworks. Extract concrete design lessons — what works, what doesn't, and what tract can adopt or deliberately avoid. The output is a structured analysis per framework plus an opinionated synthesis with prioritized recommendations.

## Research Targets

### Tier 1 — Deep Dive (all dimensions, ~500-700 words each)

These have the most divergent and instructive design decisions:

1. **LangGraph** — Graph-based agent orchestration (LangChain ecosystem)
2. **AutoGen / AG2** — Microsoft's multi-agent conversation framework (analyze newest stable version only; note what changed from v1 and why)
3. **DSPy** — Programmatic prompt optimization (Stanford NLP)
4. **Pydantic AI** — Type-safe agent framework built on Pydantic
5. **OpenAI Agents SDK** — OpenAI's agent framework (formerly Swarm)

### Tier 2 — Targeted Analysis (dimensions A, B, C, D only; ~300 words each)

These mostly echo Tier 1 patterns but may have unique angles worth capturing:

6. **CrewAI** — Multi-agent role-based collaboration
7. **Semantic Kernel** — Microsoft's AI orchestration SDK
8. **Google ADK** (Agent Development Kit) — Google's agent framework
9. **LlamaIndex** — Agent + retrieval framework (focus on AgentRunner, AgentWorker, context assembly patterns — directly relevant to tract's compile pipeline)
10. **Instructor** — Structured output / type-safe LLM interactions (interesting for typing patterns and validation, not agents per se)

### Why These, Not Others

- **Smolagents**: Too thin an API surface to justify analysis. Skim and note anything surprising in synthesis.
- **Anthropic Agent SDK**: If a public framework exists at research time, analyze it at Tier 2. If it's just API docs for tool_use, skip.
- **Mastra**: TypeScript-only. Note in synthesis if its workflow primitives offer something Python frameworks don't.
- **ControlFlow, Haystack, Marvin**: Mention in synthesis only if they introduce patterns not seen in the 10 above.

## Dimensions to Analyze

### A. Core Abstractions & Extension Points
- What are the 3-5 primary classes/concepts users interact with?
- How are they composed? (inheritance, composition, configuration, decoration)
- What is the "unit of work"? (a step, a node, a turn, a task)
- On a spectrum from "fully declarative / config-driven" to "fully imperative / code-driven", where does this framework land? Give a concrete example of something that requires fighting the framework's opinions.
- What are the explicit extension points? What requires subclassing vs. configuration vs. monkey-patching?
- How composable are components? Can you mix-and-match pieces or is it all-or-nothing?

### B. State & Memory Model
- How does conversation/session state flow between steps?
- Is there a concept of persistent memory vs. ephemeral context?
- How is state scoped? (global, per-agent, per-conversation, per-step)
- What serialization/persistence story exists?
- **Challenge tract**: If you were building tract from scratch today, would you adopt this framework's state model instead of tract's commit-based DAG? Why or why not? What would you lose?

### C. Tool/Function Calling Design
- How are tools defined? (decorators, classes, schemas, dicts)
- How is tool execution handled? (automatic, manual, hybrid)
- How are tool results fed back into the conversation?
- Is there tool-level middleware/hooks?
- How does streaming interact with tool calls? Can you observe/modify context mid-stream?
- **Challenge tract**: Compare to tract's ToolDefinition/ToolProfile/ToolExecutor pattern. Is tract over- or under-abstracting?

### D. Multi-Agent Patterns
- How do agents communicate? (direct call, message passing, shared state, event bus)
- How is delegation/handoff modeled?
- Is there a supervisor/orchestrator pattern?
- How are agent boundaries enforced?
- **Challenge tract**: Compare to tract's spawn()/collapse() and session model. Is the DAG-branching metaphor natural for multi-agent, or are other frameworks onto something better?

### E. LLM Client & Streaming (Tier 1 only)
- How tightly coupled is the framework to specific LLM providers?
- Is there a provider abstraction layer? How clean is it?
- How are model configs (temperature, max_tokens, etc.) managed?
- How does streaming work end-to-end? How do partial tool calls, streaming function arguments, and incremental state updates interact with the framework's state model?
- Structured output support?
- **Challenge tract**: Compare to tract's LLMClient protocol + OpenAI/Anthropic implementations.

### F. Control Flow & Error Handling (Tier 1 only)
- How are multi-step workflows defined? (graphs, chains, loops, state machines)
- How is branching/conditional logic expressed?
- How are retries, fallbacks, and error recovery handled?
- Is there human-in-the-loop support?
- What do error messages look like when things go wrong? Show an example of a common misconfiguration error and the resulting message.
- **Challenge tract**: Compare to tract's middleware events, gates, and maintainers for control flow.

### G. API Ergonomics (Tier 1 only) — Measurable
Do not write subjective impressions. Measure:
- **Import count**: How many imports for a basic agent loop?
- **Line count**: Lines of code for (1) single-turn tool use, (2) multi-turn conversation with memory, (3) multi-agent handoff. Use official examples, not contrived minimal versions.
- **Boilerplate ratio**: What % of code is framework ceremony vs. user logic?
- **Top 5 GitHub issues** by thumbs-up/reactions tagged 'bug' or 'enhancement' — what do users actually complain about?
- **Error experience**: What happens when you misconfigure a tool? Forget a required field? Use the wrong type?

### H. Observability & Debugging (Tier 1 only)
- How do you inspect what the agent did after a multi-step run?
- Is there built-in tracing? (LangSmith, Weave, etc.)
- What does the debug experience look like? Can you set breakpoints in the agent loop?
- How does state inspection work at runtime vs. post-mortem?
- **Tract angle**: Tract's DAG is inherently a trace. How do other frameworks' observability tools compare to just reading the commit history?

### I. Testing (Tier 1 only)
- How do you unit-test an agent built with this framework?
- Are there test utilities, mocking helpers, or replay capabilities?
- How do you regression-test prompt changes?
- What does CI look like for an agent codebase?
- **Tract angle**: Tract stores full context history. Could it be the foundation for agent test infrastructure (replay, snapshot testing, etc.)?

## Cross-Framework Synthesis

After individual analyses, produce a synthesis structured as:

### 1. Common Patterns
What do 3+ frameworks agree on? These are likely industry defaults. Present as a comparison table with one row per framework.

### 2. Divergent Choices
Where do frameworks make fundamentally different decisions? What are the trade-offs? (e.g., graph-based vs. imperative control flow, typed state vs. dict state)

### 3. Anti-Patterns
What do frameworks get wrong? Recurring community pain points. What should tract explicitly avoid?

### 4. Prioritized Recommendations for Tract

Structure as:

**Must-do** (0-3 items): Changes where the research reveals a clear deficiency in tract that most frameworks handle better. For each:
- The specific tract file(s) that would change
- A before/after code sketch (pseudocode is fine)
- Estimated complexity (small / medium / large)

**Should-consider** (0-5 items): Patterns worth adopting that would improve DX. Same structure as must-do.

**Explicitly reject** (0-5 items): Patterns that are popular but wrong for tract, with reasoning for why. Rejecting with rationale is a valid and valuable output.

**Open questions** (0-3 items): Areas where the research is inconclusive and further prototyping is needed. If a recommendation can't be specified to the level of "which files change and roughly how," it belongs here.

### 5. Gaps
What do none of the frameworks handle well? Where is tract already ahead? Where is there greenfield opportunity?

## Output Format

```
.planning/agentic/
  prompt.md            # (this file)
  langgraph.md         # Tier 1 — full analysis (A-I)
  autogen.md           # Tier 1
  dspy.md              # Tier 1
  pydantic_ai.md       # Tier 1
  openai_agents.md     # Tier 1
  crewai.md            # Tier 2 — targeted (A-D only)
  semantic_kernel.md   # Tier 2
  google_adk.md        # Tier 2
  llamaindex.md        # Tier 2
  instructor.md        # Tier 2
  SYNTHESIS.md         # Cross-framework synthesis + prioritized recommendations
```

Individual framework files should focus on what's **unique or instructive** about that framework. If a dimension has nothing interesting to say, write "Nothing notable — follows the [X] pattern described in [other framework].md" and move on. Do not pad.

The SYNTHESIS.md is the primary deliverable. Framework files are reference appendices.

## Research Method

- **Primary sources**: Official documentation, GitHub repos, actual source code
- **Community signal**: GitHub issues (sort by most-reacted), Discord, Reddit, blog posts
- **Source code**: For each Tier 1 framework, read the source for: (1) the main agent/runner loop, (2) the tool execution path, (3) the state serialization mechanism. Note the actual implementation, not just the documented API.
- **Concrete examples over marketing**: Prefer showing code. If the docs show a clean 5-line example, check whether the real-world version is 50 lines.
- **Version pinning**: Note exact version analyzed. For frameworks undergoing major rewrites (AutoGen/AG2), analyze newest stable only. Note what changed and why, but don't analyze the old version.

## Scope Constraint

Tract is a library, not a framework. Recommendations should respect this boundary but flag cases where the boundary is worth questioning. We are not copying any framework's architecture — we are looking for API ergonomics, naming conventions, composition patterns, state management ideas, and tool design patterns that make tract a better building block.

## Reference: Tract's Current Design (for comparison)

Tract is a **library** providing git-like version control for LLM context windows:

- **DAG primitives**: commit, compile, branch, merge, rebase, compress, spawn/collapse
- **Content type system**: 10 typed content classes (dialogue, instruction, tool_call, etc.)
- **Compile pipeline**: token budget management, priority annotations, compile strategies
- **Middleware**: 12 lifecycle events with pre/post hooks
- **Semantic gates/maintainers**: LLM-powered quality enforcement and context maintenance
- **Convenience API**: t.system(), t.user(), t.chat(), t.generate() for the happy path
- **Runner layer**: OpenAI/Anthropic clients, toolkit system, agent loop
- **Multi-agent**: spawn() for child sessions with selective inheritance, collapse() to merge back
- **Persistence**: SQLite with automatic schema migrations
- **Config system**: t.configure() for LLM settings, compile strategies, token budgets

## Success Criteria

The research is successful if:
1. Each recommendation in SYNTHESIS.md specifies which tract files would change and includes a before/after code sketch
2. The "explicitly reject" section exists and has substantive reasoning (not just "we don't need this")
3. Comparison tables let you see how all frameworks handle a given concern at a glance
4. A reader can make API design decisions for tract without re-researching any framework
