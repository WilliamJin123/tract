# Trace — Git for Context

## Motivation

Anyone who does agentic engineering realizes the following:

- **Longer context window = degrading performance**
- **Low signal tokens ⇒ Diluted Context ⇒ degrading performance** (keep relevance and coherence high)
- **Things might go off track** ⇒ often times we want to rollback / reset context windows ⇒ both soft and hard resets, as well as cutting out / "refactoring" sections of context

The final solution produced by an agent is best if the context window before important execution / implementation is **clean, coherent, relevant, and concise**.

---

## The Idea

Git for context:

- **Hash pointers** — commits to upserts of context + commit messages
- **Branches** for exploration
  - **Merging** — without cleaning up history, leaving meta information that a path was deliberately taken
  - **Rebasing** — cleaning up history as if the branch never existed
- **Status queries**, git diffs, etc.
- **Soft and hard resets**, selecting different commits for context "checkpoints"
- Clean, simple, git-inspired or git-synonymous commands, and straightforward toolkit implementations for dedicated context management agents

For context we would be doing a lot of **preemptive branching** to avoid any bad context changes.

### Extensions Beyond Git

- **Spawn pointers** for subagents
- **Categorized / clustered commits** for retrieval
- **Dynamic rearranging** of commits (order matters semantically in context windows)
- **Three commit types**: Append, Edit, Pin (compression-resistant)
- **LLM powered** compression / summaries
- **Token accounting** as a first-class primitive — commits carry token counts, operations report token deltas

---

## Use Cases

- A software dev wants something built and the agent brings up niche knowledge we have no background with, and we want to ask **conceptual questions** about what it actually is
  - Branch this off and remove it after
- An agent pursued a certain implementation / workflow / reasoning path that led to a **local minima / negative attractor basin**
  - The sensation that we keep asking the agent to "fix something" or try again but it keeps circling the same broken approaches
  - Branch this part off and start again with a summary of what went wrong and to avoid XYZ in a new branch
- We want to clearly **"chunk" context** in a way that reflects importance of information for future compression / summary
  - e.g. a user name, a filepath, something "important" should be in its own commit
  - Large run-on abstract conversation turns are their own commit
- The above enables:
  - **Hierarchical summarization** (scope-adaptive)
  - **Exceptions** — summarize everything, but keep this specific pinned commit unchanged for crucial reasons
  - **Human-controllable context compression** — combine these commits, throw away those ones
  - **Context-specific edits** — change this context commit (add/remove/rewrite with XYZ in mind)
  - **Auditability** — clear meta-insights into context over conversation turns, tool calls, etc.

---

## Case Studies

### 1. Conceptual Tangent Pollution

A developer asks an agent to implement authentication. The agent suggests OAuth2 with PKCE, mentions JWTs, refresh tokens. The developer asks clarifying questions. 15 turns later, they're deep in cryptographic theory.

- **Without**: Context is polluted with conceptual tangents. Agent over-engineers everything on return.
- **With**: Detour is branched, explored, then collapsed to a summary: "user wants simple auth, session-based is fine."

### 2. Debugging Death Spiral

An agent is debugging an error. It tries fix after fix, each introducing a new problem. The developer keeps saying "try again" and the agent circles the same 3 approaches.

- **Without**: Context is saturated with failed approaches, interleaved with frustration and red herrings. Agent is stuck in a loop.
- **With**: Inject a constraint commit ("previous approaches failed — investigate elsewhere"), remove the loop, explore other solution spaces.

### 3. Forgotten Invariants

A user mentions at turn 3: "All outputs must go to this specific S3 bucket, partitioned by date." By turn 40, the agent has forgotten.

- **Without**: User has to keep re-reminding. Critical invariants get lost in the noise.
- **With**: Pinned commit survives compression verbatim. Agent never forgets.

### 4. Architecture Exploration

A team is deciding between two database architectures. They want the agent to fully explore both before deciding.

- **Without**: By the time they evaluate option B, context is contaminated with option A's assumptions. Agent mixes concerns.
- **With**: Each option gets unpolluted exploration on its own branch. Merge brings in only conclusions.

### 5. Multi-Agent Accountability

A head agent coordinates subagents for research, code generation, and testing. The final code has a bug.

- **Without**: Logs are flat. Correlating what each agent knew at each moment requires painful manual reconstruction.
- **With**: Full causal traceability via spawn pointers. Pinpoint exactly where information was lost.

### 6. Retroactive Correction

An agent is halfway through a feature when the developer realizes it's been operating under a wrong assumption from 20 turns ago (thinks REST, actually GraphQL).

- **Without**: Developer either restarts entirely or corrects inline, leading to schizophrenic context mixing both paradigms.
- **With**: Surgical correction of a historical commit. No restart, no pollution. Fix propagates forward cleanly.

### 7. Mid-Flight Pivot in a Swarm

A swarm of agents is migrating a frontend from React to Next.js. Subagent A refactors 50+ components while Subagent B sets up the App Router. Halfway through, the developer switches from Styled Components to Tailwind.

- **Without**: Context contains 40 turns of Styled Components logic followed by a pivot. Agents hallucinate old syntax.
- **With**: Rebase on the root context commit specifying CSS library. A Context Refactorer surgically updates intermediate drafts. Agents resume as if Tailwind was always the plan.

---

## Open Questions

### 1. Dynamic Prompt Injection & External Context

MD files, RAG, system prompts, Jinja templates — these don't fit neatly into the commit model. Options:

- Limit Trace to managing actual context windows only, track everything else in git
- Encapsulate the entire system including external files
- Track based on what's actually materialized in context at runtime
- How to reconcile with RAG, dynamic variables, dependency injection?

### 2. Read Path vs. Write Path

Git separates object store (write) from working tree (read). What's the equivalent for context?

- When an agent "checks out" a context, what's the materialization step?
- Concatenation of commits in order? Template rendering?
- This is where the Jinja/RAG/system prompt question lives, and it's load-bearing.

### 3. Agent Scaffolding

If the scaffolding is ever-evolving, how do we reconcile git tracking with Trace tracking?

### 4. Multi-Agent Systems

Clean interface for one agent. Gets messy with a human entry point agent and a deeper multi-agent system where each agent has its own context window.

#### Proposed Solution: Hierarchical Commit Graph

- **Head agent** maintains its own linear commit history for direct human interaction
- Each commit can contain **spawn pointers** — references to root commits of subagent context trees
- Subagent histories are **full Trace repos themselves**
- **One-to-many²**: A single head commit can spawn multiple parallel subagents, and point to multiple commits within each
- **Context Collapse**: When subagent completes, head receives a collapse commit (summary + compressed trace + provenance pointer)
  - Head can accept collapse (clean) or expand inline (debug)
  - Subagent histories can be pruned aggressively once collapsed
- **Audit Trail**: Full system trace is a DAG, not a tree
  - Enables: "What was agent-3 thinking when agent-1 made decision X?" → follow spawn pointer, find concurrent commit by timestamp

### 5. Compression is the Hard Problem

Lossy compression of reasoning traces is where the actual value and difficulty live.

- Who generates the summaries?
- How do you validate compression preserved the important bits?
- This is arguably the core research problem of the project.

### 6. Semantic Conflict Resolution

Git merges work because text has line-based diffing. Context is semantic. Merging branches of exploration likely requires LLM-mediated merge strategies — meaning the version control system itself requires inference calls. Different cost model than git.

### 7. Garbage Collection & Retention

Git keeps everything forever. We can't. Need explicit retention policies for subagent histories, pruned branches, collapsed traces.

---

## Key Design Decisions

- **Python-first** — context management is IO-bound (LLM calls, storage), not compute-bound. Don't prematurely optimize with C/Rust.
- **API-first** — primary consumers are agents, not humans. Python SDK is the primary interface. CLI is for debugging/inspection.
- **SQLite for storage** — simple, sufficient to start.
- **Token-aware from day one** — every commit carries a token count. Every operation reports token deltas.
- **Three commit types**: Append (new info), Edit (modify existing), Pin (compression-resistant).