# Feature Landscape

**Domain:** Git-like version control library for LLM context windows
**Researched:** 2026-02-10
**Overall confidence:** MEDIUM-HIGH (strong academic and industry evidence for core patterns; some features novel enough that confidence is lower)

---

## Existing Tools and Approaches

Before defining features, it is critical to understand what already exists. Trace enters a space where several tools address pieces of context management, but none provide the full version-control abstraction Trace proposes.

### Tool 1: MemGPT / Letta

**What it does:** OS-inspired virtual context management. Three memory tiers -- main context (RAM), recall memory (searchable conversation DB), and archival memory (vector-backed long-term storage). The LLM itself moves data between tiers via function calls. Self-editing memory lets agents update their own persona and user models.

**What it does NOT do:** No branching, no commit history, no merge/rebase, no diff. Memory is a mutable scratchpad, not a versioned resource. No rollback semantics. The agent manages memory implicitly through tool calls rather than explicit version-control operations. No token budgeting beyond the context window hard limit.

**Relevance to Trace:** Letta validates that structured memory management dramatically improves agent performance. But Letta treats memory as a database, not a version-controlled resource. Trace's commit/branch/merge model is orthogonal and could potentially sit alongside Letta's memory tiers.

**Confidence:** HIGH (official docs, peer-reviewed paper)

### Tool 2: LangChain/LangGraph Memory

**What it does:** Multiple memory types -- ConversationBufferMemory (full history), ConversationSummaryMemory (LLM-summarized), ConversationSummaryBufferMemory (hybrid: recent messages as-is, older messages summarized). LangGraph adds checkpointing with time-travel: snapshot entire graph state at each super-step, fork from any checkpoint, resume alternate paths.

**What it does NOT do:** No semantic merge. No commit semantics (checkpoints are automatic, not intentional). No compression-resistant pinning. No explicit branching model -- forking is a side effect of resuming from a past checkpoint, not a first-class operation. No token budgeting or awareness beyond truncation. Memory types are conversation-centric, not general context management.

**Relevance to Trace:** LangGraph's checkpointer validates the value of state snapshots and time-travel for debugging. But checkpoints are opaque state blobs, not semantically meaningful commits with messages and types. Trace's intentional commit model with messages, types, and parent pointers provides richer history than automatic checkpointing.

**Confidence:** HIGH (official LangChain/LangGraph docs, multiple sources)

### Tool 3: Git Context Controller (GCC)

**What it does:** Academic framework that gives LLM agents COMMIT, BRANCH, MERGE, and CONTEXT operations. Structures agent memory as a persistent file system. Agents checkpoint meaningful progress, branch to explore alternatives, merge to synthesize reasoning paths, and retrieve historical context at varying resolutions.

**Results:** 48% on SWE-Bench-Lite (outperforming 26 systems). Agents spontaneously adopted disciplined behaviors -- modularizing code, testing before committing, branching for experiments. 40.7% vs 11.7% task resolution with vs without GCC.

**What it does NOT do:** GCC operates at the file-system level -- agents write actual files, not manage in-memory context. No token awareness. No compression. No pinning. No materializer concept. Designed for coding agents specifically, not general context management.

**Relevance to Trace:** GCC is the closest existing system to Trace's vision and strongly validates the git mental model for agent context. But GCC manages files on disk, while Trace manages context windows in memory. GCC's results prove the version-control metaphor improves agent performance. Trace should study GCC's operation semantics carefully.

**Confidence:** HIGH (peer-reviewed paper with benchmark results)

### Tool 4: ContextBranch

**What it does:** Conversation management with four primitives -- checkpoint (capture state), branch (isolated exploration), switch (change active context), inject (selectively merge messages between branches). Treats checkpoints as immutable snapshots with branches maintaining independent futures.

**Results:** 2.5% higher response quality, +4.6% focus, +6.8% context awareness vs linear conversations. 58.1% context size reduction through branching. Benefits concentrated in complex scenarios.

**What it does NOT do:** No commit semantics beyond checkpointing. No compression. No merge conflict resolution (inject is manual selection). No token awareness. No persistence. Designed for human-in-the-loop exploratory programming, not agent frameworks.

**Relevance to Trace:** Validates branching and isolated exploration. The inject primitive (selective merge) is particularly interesting -- it is a simpler model than full merge that may be more practical for context. The 58% context reduction from branching alone is a strong signal that branch-based context management works.

**Confidence:** HIGH (peer-reviewed paper with controlled experiment)

### Tool 5: Mem0

**What it does:** Universal memory layer for AI apps. Extracts and consolidates salient facts from conversations into compressed memory representations. Three memory scopes: user (cross-session), session (within conversation), agent (per-instance). Graph-based memory for relational structures. Conflict detection and resolution for contradictory information.

**What it does NOT do:** No version control semantics. No branching, diffing, or rollback. Memory is extracted facts, not managed context. No token budgeting. Optimized for personalization and recall, not for context window composition.

**Relevance to Trace:** Mem0's conflict detection for contradictory facts is relevant to Trace's merge conflict handling. The 90% token savings from extracting only relevant memories validates aggressive compression. But Mem0 extracts facts into a separate store; Trace manages the context window itself.

**Confidence:** HIGH (official docs, peer-reviewed paper, production deployment)

### Tool 6: Twigg

**What it does:** Desktop app providing visual tree diagram of LLM conversations. Branch from any point, merge branches, cut/copy/delete nodes. Multi-model support. Claims 30-60% token cost reduction through selective context inclusion.

**What it does NOT do:** Human-facing UI tool, not a developer library/SDK. No programmatic API. No compression. No commit semantics. No token-aware operations. Not designed for agent integration.

**Relevance to Trace:** Validates the visual tree model for conversation management. The branch-merge workflow mirrors git closely. But Twigg is a consumer product, not a developer tool. Trace serves a fundamentally different audience (agent framework developers).

**Confidence:** MEDIUM (product page, HN discussion, no peer review)

### Tool 7: Context Folding

**What it does:** Academic framework where agents can branch into sub-trajectories for subtasks, then fold (collapse) intermediate steps while retaining a summary. Trained via RL (FoldGRPO) to learn when to fold. 62% on BrowseComp-Plus, 58% on SWE-Bench Verified with only 32K token budget.

**What it does NOT do:** Requires custom RL training. Not a library. No persistence. No multi-agent. No explicit version control semantics.

**Relevance to Trace:** The fold/collapse operation directly maps to Trace's compress and collapse concepts. The distinction between fold (agentic, at subtask boundaries) vs summarization (post-hoc, generic) is important -- Trace should support both patterns.

**Confidence:** HIGH (peer-reviewed paper with benchmark results)

### Tool 8: ACON (Agent Context Optimization)

**What it does:** Gradient-free context compression framework that uses compression guidelines in natural language. Analyzes paired trajectories (full vs compressed) to learn what to preserve. 26-54% memory reduction while preserving task success.

**What it does NOT do:** Not a general-purpose library. Requires calibration on task trajectories. No version control semantics.

**Relevance to Trace:** ACON's guideline-based compression approach could inform Trace's compress operation. The finding that naive summarization loses critical details is a strong signal that Trace needs smarter compression than "just summarize."

**Confidence:** HIGH (peer-reviewed paper with benchmarks)

---

## Git Primitives Mapping to Context Management

The following table maps git concepts to context management, with assessment of how well each maps.

| Git Primitive | Context Analog | Maps Well? | Notes |
|---------------|---------------|------------|-------|
| `init` | Create new context/trace | Yes | Direct mapping. Initialize empty context with metadata. |
| `commit` | Snapshot context state with message | Yes, with adaptation | Core operation. Unlike git, context commits have *types* (append/edit/pin) because context is not just "changed files." |
| `log` | View context history | Yes | Direct mapping. Show commit chain with messages, token counts, timestamps. |
| `status` | Show current context state | Yes | Show HEAD, current branch, token count, uncommitted changes. |
| `diff` | Compare context states | Partially | Text diff is insufficient -- context changes are *semantic*. Need semantic diff (what meaning changed) alongside textual diff. |
| `branch` | Create isolated context line | Yes | Strong mapping. Validated by ContextBranch (58% context reduction) and GCC (spontaneous disciplined behavior). |
| `checkout`/`switch` | Change active context | Yes | Direct mapping. Switch HEAD to different branch. |
| `merge` | Combine context from branches | Yes, but harder | Git merges files; Trace merges *meaning*. Requires LLM-mediated semantic merge because context is order-sensitive natural language. Automatic three-way merge does not work for prose. |
| `rebase` | Restructure context history | Partially | Rebase in git replays commits on new base. For context, this means re-applying context changes in a different order. Useful but dangerous -- context is order-sensitive, so reordering can change meaning. Needs semantic safety checks. |
| `reset` | Undo context changes | Yes | Soft (move HEAD, keep content) and hard (move HEAD, discard content). Direct mapping. |
| `stash` | Temporarily shelve changes | Partially | Less clear mapping for context. Could mean "set aside some context temporarily." Lower priority. |
| `tag` | Mark important states | Yes | Pin a particular context snapshot. Maps to Trace's "pin" commit type. |
| `cherry-pick` | Apply specific commit to another branch | Yes | Selectively inject specific context from one branch to another. Maps to ContextBranch's "inject" primitive. |
| `blame` | Track origin of content | Partially | "Which commit introduced this context?" Useful for debugging but complex for natural language. |
| `gc` | Garbage collection | Yes | Clean up unreachable commits, expired summaries. Direct mapping with token-awareness. |
| `clone`/`fork` | Copy context for another agent | Yes | Maps directly to Trace's "spawn" for multi-agent. Give a sub-agent a copy of (some of) the parent's context. |
| `revert` | Undo a specific commit | Partially | "Remove the effect of commit X." Hard for natural language -- can't just reverse a patch. Would need LLM to determine what "undoing" an edit means semantically. |
| `.gitignore` | Exclude from context | Partially | Could map to filtering rules for materialization, but context is opt-in not opt-out. |

### Key Insight: Where Git Breaks Down

Git manages *files* that are *independently addressable* and *order-insensitive* (file A does not change meaning based on position relative to file B). Context is *order-sensitive natural language* where:

1. Position affects meaning (instructions at the top vs bottom get different attention)
2. Content is interrelated (removing one section can make another incoherent)
3. "Merge conflicts" are semantic, not textual
4. Compression is lossy by nature (unlike git's lossless storage)

This means Trace must add capabilities that git does not need:
- **Token awareness** (git has no concept of "size budget")
- **Semantic operations** (merge, diff, rebase need LLM mediation)
- **Compression** (git stores everything; Trace must summarize)
- **Materializers** (git outputs files; Trace outputs prompts)
- **Order sensitivity** (reorder operations need semantic safety checks)

---

## Table Stakes

Features users expect. Missing = developers will not adopt the library.

| Feature | Why Expected | Complexity | Confidence | Notes |
|---------|--------------|------------|------------|-------|
| **Commit with message and metadata** | Core promise of "git for context." Without commits, there is no version control. Must capture content, message, timestamp, parent pointer, token count. | Low | HIGH | Every comparable system (GCC, ContextBranch, LangGraph checkpointer) has this. Three commit types (append/edit/pin) are a Trace-specific innovation on top. |
| **Linear history (log)** | Developers expect to see what happened. Log is the most basic inspection tool. | Low | HIGH | Direct git mapping. Include token counts per commit and cumulative. |
| **Branch and switch** | Validated by GCC (+348% task resolution), ContextBranch (58% context reduction, +6.8% context awareness). Branching is what makes version control useful, not just logging. | Medium | HIGH | Must be cheap (pointer-based, not copy). Copy-on-write semantics. |
| **Reset (soft and hard)** | Undo is fundamental. Agents and developers need to roll back when exploration fails. Every version control system has this. | Low | HIGH | Soft = move HEAD, keep content accessible. Hard = move HEAD, discard forward commits from working state. |
| **Token counting** | Context management without token awareness is like file management without knowing file sizes. Every decision depends on "how many tokens is this?" The entire value proposition depends on token-awareness. | Medium | HIGH | tiktoken default + pluggable tokenizer for model-specific counting. Must be on every commit and operation. |
| **Materialize / render context** | The whole point is producing a context window for the LLM. "Give me the current context as a string/messages" is the read operation. Without this, the library has no output. | Medium | HIGH | Simple concatenation default. Must be pluggable for users who want structured prompts (XML tags, role prefixes, etc). |
| **Persistence (save/load)** | Agents crash. Sessions end. Context history must survive. In-memory-only context management is a toy. | Medium | HIGH | SQLite via SQLAlchemy is the right default. Must support session recovery. |
| **Merge (basic)** | If you have branches, you need to combine them. Merge is the complement of branch. Without merge, branches are dead ends. | High | HIGH | This is where Trace diverges from git. Context merge is semantic, not textual. Need LLM-mediated merge strategy. Simpler "inject" (cherry-pick specific content) should be available too for cases where full merge is overkill. |
| **Diff (basic)** | "What changed?" is the most basic debugging question. Developers expect diff between any two states. | Medium | MEDIUM | Textual diff is straightforward. Semantic diff (what *meaning* changed) is a differentiator, not table stakes. Basic textual diff is table stakes. |
| **Checkout (read-only)** | Navigate to any point in history to inspect it. Read-only checkout for debugging and inspection. | Low | HIGH | Direct git mapping. |

---

## Differentiators

Features that set Trace apart. Not expected but highly valued. These are what make developers choose Trace over rolling their own.

| Feature | Value Proposition | Complexity | Confidence | Notes |
|---------|-------------------|------------|------------|-------|
| **Typed commits (Append/Edit/Pin)** | No existing tool distinguishes *how* context changed. Append adds new info, Edit modifies existing, Pin marks compression-resistant content. This enables smarter compression (never summarize pinned content), smarter diffs (edit vs addition is meaningful), and smarter materialization. | Medium | MEDIUM | Novel concept. Validated by project design but not by existing tools. The pin type is particularly valuable -- tells the compressor "this matters, do not lose it." |
| **Token-budget-aware compression** | Not just "summarize old stuff" but "compress to fit within N tokens while preserving pinned content and maximizing information retention." ACON shows naive summarization loses critical details. Hierarchical compression (summarize summaries) enables aggressive compression. | High | MEDIUM | ACON (26-54% reduction), Context Folding (58% on SWE-Bench with 32K budget), Mem0 (90% token savings) all validate compression. But *budget-aware* compression that respects pin types is novel. |
| **Pluggable materializers** | Different frameworks need different context formats. Claude wants XML tags, ChatGPT wants message arrays, some want markdown, some want structured templates. Materializers decouple storage from rendering. | Medium | MEDIUM | No existing tool has this. MCP provides prompt templates but not pluggable rendering. This is a framework-agnostic enabler. |
| **Semantic merge with LLM mediation** | Merging natural language context is fundamentally different from merging code. Trace can use the LLM itself to resolve "merge conflicts" where two branches added contradictory or overlapping information. | High | MEDIUM | GCC has basic MERGE. ContextBranch has inject (manual). No tool does LLM-mediated semantic merge. High complexity, high value. |
| **Spawn/collapse for multi-agent** | Give a sub-agent a subset of context (spawn), get back their results as a collapsed summary (collapse). Maps to git clone + squash merge. Enables multi-agent coordination without shared mutable state. | High | MEDIUM | Letta's shared memory blocks, collaborative memory frameworks, and Memory-as-a-Service papers all validate multi-agent memory as a critical need. But spawn/collapse via version control semantics is novel. |
| **Rebase with semantic safety checks** | Reorder context history while checking that the new order preserves semantic coherence. Unlike git rebase which replays patches, context rebase must verify meaning is preserved when order changes. | High | LOW | Novel concept. No existing tool has this. Addresses the order-sensitivity problem that makes context different from code. High complexity. |
| **Compression-resistant pinning** | Mark specific context as "never compress this." System prompt instructions, critical facts, user preferences that must survive any compression pass. No existing tool has this concept. | Low | MEDIUM | Simple to implement (flag on commit), high value. Prevents the #1 complaint about context compression -- losing important information. |
| **Commit reordering with safety** | Rearrange context order (e.g., move instructions to top) while verifying semantic safety. Context is order-sensitive; reordering can change meaning. | High | LOW | Novel. No existing tool. Addresses a real need (LLMs attend to positions differently) but complex to implement safely. |
| **Garbage collection with retention policies** | Automatic cleanup of unreachable commits, expired summaries, orphaned branches. Token-aware: keep context that fits budget, gc the rest. | Medium | MEDIUM | Git has gc. Trace's gc is token-aware and respects pin types. |
| **Semantic diff** | Not just "what text changed" but "what meaning changed." LLM-powered diff that explains the semantic difference between two context states. | High | LOW | The llm-prompt-semantic-diff tool shows early work here. Novel for context management. Very useful for debugging agents but high complexity. |

---

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Do not replace conversation history** | Trace manages *context*, not conversations. Chat history is one input to context. If Trace tries to own the conversation loop, it competes with every agent framework (LangChain, CrewAI, Agno, etc.) and nobody will adopt it. | Be a layer that agent frameworks compose. Accept conversation messages as commits. Let the framework own the conversation loop. |
| **Do not build a RAG system** | RAG retrieves external knowledge into context. That is a well-solved problem (LlamaIndex, LangChain, Pinecone, etc.). Building RAG into a context version-control library creates scope creep and inferior RAG. | Accept RAG-retrieved content as commits. Let dedicated RAG tools handle retrieval. Trace manages what happens to content after it enters the context window. |
| **Do not build a vector database or embeddings store** | Mem0, Letta, and dozens of tools already handle vector storage for long-term memory. This is not Trace's job. | If long-term memory is needed, integrate with existing stores. Trace manages the *working context*, not the knowledge base. |
| **Do not build an agent framework** | CrewAI, LangGraph, Agno, OpenAI Agents SDK, and others already handle agent orchestration. Building framework features into a context library dilutes focus. | Be a library that any framework can use. Provide clean Python SDK. No opinions about agent architecture. |
| **Do not build a GUI in v1** | Twigg is a GUI-first product. Trace is SDK-first. Building a GUI before the SDK is stable wastes effort on the wrong audience. | SDK and CLI first. GUI is a future milestone. Design the data model to support future visualization. |
| **Do not auto-manage context without user control** | Letta's approach (LLM self-manages memory via function calls) is powerful but opaque. Agent framework developers want *control*, not magic. If Trace autonomously compresses or rearranges context, developers cannot debug or predict behavior. | Provide explicit operations. The user (or their agent) decides when to commit, branch, merge, compress. Offer convenience helpers but never auto-mutate without explicit invocation. |
| **Do not try to be framework-specific** | Building adapters for LangChain, CrewAI, etc. in v1 couples the library to specific frameworks and their release cycles. | Build a clean, framework-agnostic Python SDK. Framework adapters are a post-v1 concern. |
| **Do not compete with the LLM provider's context window** | Trying to extend context beyond the model's window (like MemGPT does with virtual context) is complex and model-dependent. Trace should *manage what fits*, not try to extend the window. | Work within the context window. Compress to fit. Branch to isolate. But do not implement virtual memory / paging of context in and out of the window. That is Letta's job. |
| **Do not implement real-time streaming** | Context version control is a state-management concern, not a streaming concern. Trying to version-control streaming token output adds massive complexity for unclear value. | Accept completed content as commits. Do not try to version-control partial/streaming outputs. |

---

## Feature Dependencies

```
                    init
                     |
                   commit -----> log
                   / | \           |
                  /  |  \          v
              status diff  checkout
                |
                v
            branch ---------> switch
                |                |
                v                v
              merge          reset
              / | \
             /  |  \
            v   v   v
     compress  pin  rebase
         |            |
         v            v
        gc     reorder (safety)

            spawn ---------> collapse
                               |
                               v
                          multi-agent

    materialize (depends on commit, reads from any state)
    persistence (depends on commit, wraps storage)
```

Key dependency chains:

1. **Core chain:** init -> commit -> log/status/diff/checkout -> branch/switch -> merge -> compress -> gc
2. **Compression chain:** commit (with types) -> compress (respects pins) -> hierarchical compress -> gc
3. **Multi-agent chain:** branch -> spawn -> collapse (which is compress + merge)
4. **Rendering chain:** commit -> materialize (pluggable) -> token counting -> budget validation

Features that are *independent* and can be built in parallel:
- log, status, diff, checkout (all read operations on commit history)
- branch and reset (both pointer operations)
- materialize and persistence (both wrap the core data model)

Features that *must* be sequential:
- commit before anything else
- branch before merge
- compress before gc
- spawn before collapse
- merge before rebase (rebase is merge's harder sibling)

---

## MVP Recommendation

For MVP, prioritize the **core chain** that proves the value proposition: agents produce better output when context is version-controlled.

### Must have (Phase 0-1):
1. **init, commit (with types), log, status** -- The core data model and inspection
2. **diff (textual)** -- Basic "what changed"
3. **reset (soft/hard)** -- Undo capability
4. **checkout** -- Navigate history
5. **Token counting on every operation** -- The differentiating substrate
6. **Materialize (simple concat)** -- The read/render operation
7. **Persistence (SQLite)** -- Survive sessions

### Should have (Phase 2):
8. **Branch and switch** -- Validated by GCC and ContextBranch as high-value
9. **Merge (basic + LLM-mediated)** -- Complement to branching
10. **Compress (with pin respect)** -- Token budget management
11. **CLI for inspection** -- Developer experience

### Nice to have (Phase 3):
12. **Spawn and collapse** -- Multi-agent support
13. **Rebase with safety checks** -- Advanced history management
14. **Commit reordering** -- Advanced context optimization
15. **Garbage collection** -- Cleanup

### Defer to post-v1:
- **Semantic diff** -- High complexity, uncertain value in v1
- **Framework adapters** -- Need stable SDK first
- **GUI/visualization** -- SDK must be proven first
- **Autonomous context policies** -- Explicit control first, automation later
- **Virtual context / paging** -- Let Letta own this

---

## Sources

### Academic Papers (HIGH confidence)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Git Context Controller: Manage the Context of LLM-based Agents like Git](https://arxiv.org/abs/2508.00031)
- [ContextBranch: Context Branching for LLM Conversations](https://arxiv.org/abs/2512.13914)
- [ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/abs/2510.00615)
- [Context Folding: Scaling Long-Horizon LLM Agent](https://arxiv.org/abs/2510.11967)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- [Collaborative Memory: Multi-User Memory Sharing in LLM Agents](https://arxiv.org/abs/2505.18279)
- [Memory as a Service (MaaS)](https://arxiv.org/html/2506.22815v1)
- [Intrinsic Memory Agents](https://arxiv.org/html/2508.08997v1)
- [ComprExIT: Context Compression via Explicit Information Transmission](https://arxiv.org/html/2602.03784)

### Official Documentation (HIGH confidence)
- [Letta/MemGPT Docs](https://docs.letta.com/concepts/memgpt/)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangChain Context Engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)
- [LangChain ConversationSummaryBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary_buffer.ConversationSummaryBufferMemory.html)
- [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Mem0 Official Docs](https://docs.mem0.ai/platform/overview)

### Industry Sources (MEDIUM confidence)
- [Letta Blog: Rearchitecting Letta's Agent Loop](https://www.letta.com/blog/letta-v1-agent)
- [Letta Blog: Guide to Context Engineering](https://www.letta.com/blog/guide-to-context-engineering)
- [Martin Fowler: Context Engineering for Coding Agents](https://martinfowler.com/articles/exploring-gen-ai/context-engineering-coding-agents.html)
- [LangChain Blog: Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)
- [LangChain: State of Agent Engineering](https://www.langchain.com/state-of-agent-engineering)
- [MongoDB: Why Multi-Agent Systems Need Memory Engineering](https://www.mongodb.com/company/blog/technical/why-multi-agent-systems-need-memory-engineering)
- [Factory.ai: Compressing Context](https://factory.ai/news/compressing-context)

### Product/Community Sources (LOW-MEDIUM confidence)
- [Twigg: Git for LLMs (Product Hunt)](https://www.producthunt.com/products/twigg)
- [HN: Show HN: Git for LLMs](https://news.ycombinator.com/item?id=45682776)
- [Context Llemur (GitHub)](https://github.com/jerpint/context-llemur)
- [ContextLab (GitHub)](https://github.com/Siddhant-K-code/ContextLab)
- [Claude Code Conversation Branching Feature Request](https://github.com/anthropics/claude-code/issues/16236)
