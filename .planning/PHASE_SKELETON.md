# Trace — Roadmap

## Phase 0: Foundations

**Goal**: Define the core primitives and make architectural decisions.

- Define the core data model
  - Commit structure: content block, commit message, token count, timestamp, parent pointer(s), commit type (append/edit/pin)
  - DAG structure for commit history
  - Branch and HEAD references
- Choose storage backend (SQLite to start using sqlalchemy)
- Design the internal Python API surface — this is the primary interface, not the CLI
- Set up project scaffolding, testing infrastructure
- Define serialization format for commits and context snapshots

**Exit criteria**: Can programmatically create, store, and retrieve commits with token counts. Core data model is stable.

---

## Phase 1: Single Agent, Linear History

**Goal**: Prove the value of checkpoint/restore and clean context on real agentic tasks.

- Implement core operations:
  - `trace init` — initialize a context trace
  - `trace commit` — commit a context snapshot (append/edit/pin types)
  - `trace log` — view commit history with token counts
  - `trace status` — current HEAD, token budget usage
  - `trace diff` — compare two commits
  - `trace reset --soft` — move HEAD, keep content available
  - `trace reset --hard` — move HEAD, discard forward history
  - `trace checkout <commit>` — restore context to a specific checkpoint
- Context materialization: define the "read path" — how a checked-out context becomes an actual prompt/context window (concatenation of commits in order)
- Token budget tracking: warn when approaching limits, show cost of each commit
- Integrate with one agentic framework (Claude Code, OpenAI Agents SDK, or LangGraph)
- Build basic CLI wrapper around the Python API

**Exit criteria**: Can manage a single agent's context through a real multi-turn task, roll back mistakes, and demonstrate cleaner outputs vs. unmanaged context.

---

## Phase 2: Branching, Merging & Compression

**Goal**: Enable exploration without pollution, and lossy compression with human/agent control.

### Branching & Merging

- `trace branch <name>` — create a named branch from current HEAD
- `trace switch <name>` — switch active branch
- `trace merge <branch>` — merge branch into current (preserves history, adds merge commit)
- `trace rebase <branch>` — rebase current onto target (clean history, branch disappears)
- LLM-mediated semantic merge strategy
  - Define what a "conflict" means in context (contradictory information, redundant content)
  - Implement merge driver that calls an LLM to reconcile
  - Track inference cost of merge operations

### Compression

- `trace compress <range>` — summarize a range of commits into a single commit
- `trace compress --respect-pins` — compress but preserve pinned commits verbatim
- Hierarchical summarization: compress at different granularities
- Human-controllable strategies: "combine these commits," "drop these," "keep this one verbatim"
- Validation: mechanism to check that compression preserved critical information (at minimum: pinned content integrity, token count reporting)

### Commit Reordering

- `trace reorder` — rearrange commits (context is order-sensitive, unlike git content)
- Safety checks: warn when reordering changes semantic meaning

**Exit criteria**: Can branch for exploration, merge/rebase results, and compress history while preserving pinned invariants. Demonstrated on real tasks (architecture exploration, debugging death spirals).

---

## Phase 3: Multi-Agent

**Goal**: Extend Trace to coordinate multiple agents with independent context windows.

- **Spawn pointers**: a commit in a parent agent's history can reference the root of a child agent's Trace
  - `trace spawn <agent-name>` — create a subagent trace linked to current commit
  - Each subagent gets its own full Trace repo
- **Context collapse**: when a subagent completes, parent receives a collapse commit
  - Contains: final output, compressed reasoning trace, provenance pointer to full subagent history
  - `trace collapse <agent-name>` — generate collapse commit
  - `trace expand <collapse-commit>` — pull subagent commits inline for debugging
- **One-to-many²**: a single parent commit can spawn multiple parallel subagents, each with multiple internal commits
- **Session persistence & crash recovery**
  - Durable storage of all agent traces
  - Ability to resume from last committed state after process restart
- **Garbage collection**
  - Define retention policies (e.g., prune subagent history after collapse, keep only provenance pointer)
  - `trace gc` — run garbage collection per policy

**Exit criteria**: Can coordinate a head agent + 2-3 subagents on a real task, collapse results, and trace provenance of information across agents.

---

## Phase 4: Tooling & Observability

**Goal**: Make the system inspectable by humans.

- **Enhanced CLI**
  - `trace graph` — visualize the commit DAG (like `git log --graph` but for agent traces)
  - `trace audit <query>` — answer questions like "what did agent-3 know when agent-1 decided X?"
  - `trace blame <content>` — trace origin of specific context content
- **Web/Desktop Viewer** (the "GitHub of Trace")
  - Visual DAG explorer for multi-agent traces
  - Timeline view: temporal evolution of each agent's context
  - Diff viewer for context snapshots
  - Token budget visualization over time
  - Inspired by OpenTelemetry trace viewers
- **Export & Interop**
  - Export traces to JSON/JSONL for analysis
  - Integration hooks for logging platforms

**Exit criteria**: A human can visually inspect a multi-agent trace, understand what each agent knew at each point, and identify where information was lost.

---

## Phase 5: Autonomous Context Management

**Goal**: Agents manage their own context without human intervention.

- **Policy engine**: define rules for automatic context management
  - Auto-branch on detected tangents
  - Auto-compress when token budget exceeds threshold
  - Auto-pin based on heuristics (user requirements, file paths, constraints)
  - Auto-rebase when exploration branch is abandoned
- **Context management agent**: a dedicated agent whose job is to manage another agent's Trace
  - Monitors context health (relevance, coherence, token usage)
  - Proposes and executes context operations
  - Can be overridden by human
- **Benchmarking**: measure impact of autonomous context management on task completion quality
  - A/B comparisons: managed vs. unmanaged context on standardized agentic tasks
  - Token efficiency metrics

**Exit criteria**: An agent can complete a complex multi-step task with autonomous context management that demonstrably outperforms unmanaged context, with no human intervention on context operations.