# Domain Pitfalls

**Domain:** Context management / version control library for LLM agents (Python)
**Researched:** 2026-02-10

---

## Critical Pitfalls

Mistakes that cause rewrites or major issues. Each of these, if not caught early, will require significant rearchitecture.

---

### Pitfall 1: Over-Extending the Git Metaphor Where It Breaks

**What goes wrong:** The team maps git concepts 1:1 onto context management, but several git assumptions do not hold for LLM context. The resulting API confuses users or produces semantically incorrect results.

**Why it happens:** Git is the mental model driving the project. It is easy to inherit git's assumptions without noticing they are context-hostile:

1. **Git is order-agnostic; context is order-sensitive.** Git treats files as sets of lines. A commit's content does not depend on the order of other commits. But LLM context is a sequence where position matters. The "Lost in the Middle" phenomenon (documented by Liu et al., 2023; confirmed by MIT research in 2025) shows LLMs exhibit a U-shaped attention curve -- information at the start and end of context gets more weight than information in the middle. This means `reorder`, `merge`, and `rebase` are not neutral operations. Moving a commit from position 3 to position 15 changes its effective weight in the LLM's reasoning, even if the content is identical.

2. **Git merge resolves line-level conflicts; context "conflicts" are semantic.** Two branches might contain contradictory assumptions ("use REST" vs "use GraphQL"). Line-level diffing cannot detect this. LLM-mediated merge is required, but research shows that even SOTA LLM-based merge approaches struggle with semantic-level conflicts (CHATMERGE, 2023; ConGra benchmark, 2024). The merge operation itself is unreliable.

3. **Git diff is well-defined; context diff is ambiguous.** Diffing two context states does not have an obvious meaning. Is it a textual diff? A semantic comparison? A token count delta? Users expecting git-like diffs will be confused.

4. **Git checkout restores file state; context checkout changes what the LLM "knows."** Checking out an old commit in git is a filesystem operation. Checking out an old context state means the agent loses access to everything committed after that point. This is much more destructive than it feels in git terms.

**Consequences:** API that maps poorly to real usage. Users build mental models from git that produce unexpected behavior. Core operations (merge, rebase, reorder) silently degrade LLM performance because they change context positioning.

**Prevention:**
- Document explicitly where the git analogy applies and where it breaks. Create a "Git vs Trace" conceptual guide as part of the SDK documentation.
- For `reorder`: always warn about positional sensitivity. Consider a "semantic safety check" that uses an LLM to evaluate whether reordering changes meaning. This is already in the PHASE_SKELETON but should be treated as a first-class concern, not an afterthought.
- For `merge`/`rebase`: design these as explicitly LLM-mediated operations from day one. Do not attempt textual merge as a fallback -- it will produce subtly wrong context. Accept that merge quality depends on LLM quality and expose confidence/cost to the user.
- For `diff`: define it clearly as token-level delta + optional semantic comparison. Do not try to replicate git's line-based diff.
- For `checkout`: make the destructive nature explicit in the API. Consider a "detached HEAD" warning equivalent.

**Detection:** Users reporting unexpected agent behavior after merge/rebase/reorder operations. Agent outputs that contradict information from commits that were repositioned in context.

**Phase relevance:** Phase 0 (data model must account for order sensitivity), Phase 2 (merge/rebase/reorder are the highest-risk operations)

**Confidence:** HIGH -- position sensitivity is well-documented in research; the git analogy limitations follow directly from Trace's design premises.

---

### Pitfall 2: Lossy Compression That Silently Drops Critical Information

**What goes wrong:** Context compression (summarization) loses specific details that the agent later needs. The agent produces confidently wrong outputs because the information it needs was compressed away. This failure is silent -- there is no error, no warning, just degraded quality.

**Why it happens:** LLM summarization is inherently lossy. Research from 2025 shows that all compression methods result in sizeable performance losses (3% to 55% relative difference), with the highest drops on tasks requiring multi-hop reasoning and information aggregation. The types of information most vulnerable to compression loss are:

- **Specific values**: file paths, variable names, version numbers, thresholds, exact requirements
- **Conditional logic**: "if X then Y, but if Z then W" gets flattened
- **Negations and constraints**: "do NOT use library X" becomes "use library X" or disappears
- **Temporal relationships**: "first do A, then B" loses ordering
- **Cross-references**: relationships between entities that span multiple commits

The project already identifies this as "arguably the core research problem" in the PREPLAN. But the pitfall is specific: it is tempting to ship compression early with a simple summarization prompt and discover the failure mode only when agents misbehave in production.

**Consequences:** Agent produces subtly wrong outputs. Users lose trust in the system. The failure is hard to diagnose because the compressed context looks reasonable to a human reviewer -- the problem only manifests when the agent reasons over it.

**Prevention:**
- **Pin commits are the right primitive** -- the project already has this. Enforce that pinned commits survive compression verbatim, with integrity verification (hash comparison).
- **Compression validation**: after compressing, run a verification step that checks whether key facts from the original are preserved in the summary. This can be a simple LLM call: "Given this original and this summary, what information was lost?" This adds cost but prevents silent degradation.
- **Hierarchical compression with explicit layers**: do not flatten everything to one summary. Keep 2-3 layers of detail (full -> detailed summary -> brief summary). Allow the agent to "zoom in" when it needs specifics.
- **Expose compression metrics**: token reduction ratio, estimated information retention score. Let the user decide if the tradeoff is acceptable rather than making it silently.
- **Never auto-compress without user/agent opt-in**. Compression should be an explicit operation, not an automatic background process in v1.

**Detection:** Agent asking for information that was previously in context but no longer present after compression. Regression in agent task completion quality after compression operations. A/B testing compressed vs uncompressed context on the same task.

**Phase relevance:** Phase 2 (compression implementation). This should be the most heavily tested feature in the entire project.

**Confidence:** HIGH -- compression loss is well-documented in research (arxiv:2503.19114, JetBrains research blog 2025, mem0.ai summarization guide 2025).

---

### Pitfall 3: Token Counting That Disagrees With the LLM Provider

**What goes wrong:** Trace's internal token counts diverge from the provider's actual token consumption. Users hit context window limits unexpectedly, or waste budget because Trace's accounting is wrong.

**Why it happens:** Token counting is model-specific and harder than it appears:

1. **Each provider has a different tokenizer.** tiktoken works for OpenAI models but produces wrong counts for Anthropic (Claude) and Google (Gemini) models. The project plans to use tiktoken as default, which means counts will be wrong for non-OpenAI users from day one.

2. **Even within OpenAI, tiktoken can diverge.** Community reports from May 2025 document significant discrepancies between tiktoken counts and API-reported usage for o4-mini and gpt-4o-mini models. New model releases can introduce new tokenizer versions that tiktoken does not immediately support.

3. **System prompts, tool definitions, and message framing add hidden tokens.** The raw content token count does not include the overhead the provider adds for message structure (role tags, tool schemas, etc.). This overhead varies by provider and by API call structure.

4. **Token counts are stale after compression.** After an LLM compresses content, the token count of the result is only known after the compression completes. Any pre-computed token counts on the compressed commit are approximate until the operation finishes.

**Consequences:** Users cannot trust Trace's token budget reporting. Context window overflows cause API errors. Budget forecasting is unreliable.

**Prevention:**
- **Do not make tiktoken the sole truth source.** Design the token counting interface as a pluggable protocol from day one. Provide tiktoken as a default with a clear warning that it is only accurate for specific OpenAI models.
- **Prefer API-reported token counts when available.** After any LLM operation (compression, merge), use the provider's reported usage to update token counts. This is already in the PROJECT.md ("API response extraction when available") -- make sure this is the primary mechanism, not a fallback.
- **Report token counts as estimates with a confidence indicator.** `token_count: 1523 (estimated, tiktoken cl100k)` vs `token_count: 1523 (provider-reported)`. Let consumers know what they are getting.
- **Add a "materialization cost" calculation** that includes message framing overhead. This requires provider-specific knowledge, but even a rough estimate (e.g., +10% overhead) is better than ignoring it.
- **Test token count accuracy** against real API calls for each supported model as part of CI. This catches tokenizer drift on new model releases.

**Detection:** Users reporting "context window exceeded" errors when Trace reports they are within budget. Token counts that differ from provider invoices by more than 5%.

**Phase relevance:** Phase 0 (design the pluggable interface), Phase 1 (implement and test with real API calls)

**Confidence:** HIGH -- token count discrepancies are documented in OpenAI community forums and GitHub issues (openai/openai-python#2538).

---

### Pitfall 4: Multi-Agent State Corruption and Orphaned Traces

**What goes wrong:** In multi-agent scenarios, concurrent operations on related traces produce corrupted state. Subagent traces become orphaned (no parent reference), spawn pointers reference deleted commits, or concurrent writes corrupt the SQLite database.

**Why it happens:** Multi-agent LLM systems have a documented failure rate of 41-86.7% in production (Cemri et al., 2025). The coordination layer itself is a major source of failures -- 36.94% of all failures in the MAST taxonomy are coordination failures, and 79% of problems originate from specification and coordination issues. Trace adds its own coordination surface: the storage layer must handle concurrent access from multiple agents that may be in separate threads or processes.

Specific failure modes:
1. **Concurrent writes to SQLite**: even with WAL mode, only one writer can proceed at a time. If multiple agents commit simultaneously, writes will serialize (best case) or fail with "database is locked" errors (worst case, especially with long-running transactions).
2. **Orphaned traces**: if a parent agent crashes after spawning a subagent but before recording the spawn pointer, the subagent's trace exists but has no parent reference. It becomes unreachable through the normal API.
3. **Stale spawn pointers**: if a subagent's trace is garbage collected but the parent still holds a spawn pointer to it, following that pointer produces an error or dangling reference.
4. **Context collapse races**: two subagents completing simultaneously both try to collapse into the parent. Without proper serialization, the parent's commit history can become inconsistent.
5. **Conversation resets**: the MAST taxonomy identifies "unexpected conversation resets" as a key multi-agent failure mode. If a subagent trace is lost or corrupted, the parent has no way to recover the subagent's reasoning.

**Consequences:** Data loss. Corrupted audit trails that undermine Trace's core value proposition. "Database is locked" errors in production that force retries or crash recovery. Orphaned traces that leak storage.

**Prevention:**
- **WAL mode is mandatory**, not optional. Enable it at database creation time with `PRAGMA journal_mode=WAL`. Set a generous busy_timeout (5-15 seconds) for write contention.
- **Use SQLAlchemy's NullPool for SQLite** to avoid connection pooling complications. Each operation gets its own connection, which aligns with SQLite's single-writer model.
- **Design spawn/collapse as atomic operations.** The spawn pointer and the subagent's root commit must be created in the same transaction. The collapse commit and the deactivation of the subagent trace must be in the same transaction.
- **Add a cleanup/repair command**: `trace repair` that finds orphaned traces, broken spawn pointers, and inconsistent state. Run this on startup and expose it as a user-facing operation.
- **Implement write serialization at the application level** for multi-agent scenarios. Do not rely solely on SQLite's locking. A simple mutex or queue for write operations prevents contention.
- **For multi-process scenarios** (multiple agents in separate processes), consider file-based locking or a dedicated write queue. SQLite across processes with WAL mode works but requires careful timeout tuning.

**Detection:** "Database is locked" errors in logs. Traces that appear in storage but are not reachable via any HEAD or branch reference. Spawn pointers that resolve to nonexistent commits. Inconsistent parent-child relationships.

**Phase relevance:** Phase 3 (multi-agent), but Phase 0 must design the storage layer with concurrent access in mind. Retrofitting concurrency safety is a rewrite-level change.

**Confidence:** HIGH for SQLite concurrency issues (well-documented in SQLite official docs and SQLAlchemy GitHub discussions). MEDIUM for specific multi-agent failure modes (extrapolated from MAST taxonomy research -- Trace is a novel context, so exact failure modes may differ).

---

## Moderate Pitfalls

Mistakes that cause delays or technical debt but are recoverable without full rewrites.

---

### Pitfall 5: Sync/Async API Bifurcation

**What goes wrong:** The library starts with a synchronous API, then users request async support. Adding async after the fact leads to either (a) duplicated code for sync/async variants, or (b) ugly wrappers that run sync code in threads.

**Why it happens:** Many Trace operations involve LLM API calls (compression, semantic merge) which are inherently IO-bound and naturally async. Agent frameworks increasingly use async patterns (LangGraph, OpenAI Agents SDK). But writing async-first is harder during initial development, and Python's async ecosystem requires choosing between asyncio, trio, etc.

**Prevention:**
- **Write async-first internally**, then provide sync wrappers using `asyncio.run()` or similar. This is the approach used by httpx and other modern Python libraries.
- Consider using **anyio** as the async abstraction to avoid locking into asyncio specifically.
- Alternatively, use the "token removal" pattern: write async code, then use a script to strip `async`/`await` tokens to generate the sync version. This keeps both APIs in sync automatically.
- **At minimum, design the internal interfaces to be async-compatible from Phase 0**, even if the initial public API is sync-only. This means: no blocking calls in core logic, IO operations behind interfaces that can later become async.

**Detection:** Users requesting async support. Agent frameworks requiring async callbacks that Trace cannot satisfy.

**Phase relevance:** Phase 0 (architectural decision). If the internal architecture is sync-only, adding async later requires touching every IO path.

**Confidence:** MEDIUM -- based on common Python library evolution patterns (httpx, databases, SQLAlchemy's own async migration).

---

### Pitfall 6: Leaky Storage Abstraction

**What goes wrong:** SQLite-specific assumptions leak into the core domain logic. When users need a different storage backend (PostgreSQL for multi-process, file-based for simplicity, in-memory for testing), the refactoring is extensive.

**Why it happens:** SQLAlchemy provides backend abstraction in theory, but in practice:
- SQLite-specific PRAGMAs (WAL mode, busy_timeout, journal_size_limit) are embedded in initialization
- SQLite's type system differs from PostgreSQL's (e.g., no native datetime, no native JSON in older versions)
- SQLite's concurrency model (single-writer) influences architectural decisions that would differ for PostgreSQL
- SQLAlchemy's ORM can hide differences, but raw SQL or SQLite-specific features create coupling

**Prevention:**
- Define a **storage protocol** (Python Protocol or ABC) that the core domain logic uses. The protocol should define: create_commit, get_commit, get_branch, update_head, etc. SQLite is just one implementation.
- Keep SQLite-specific configuration (PRAGMAs, connection settings) in the SQLite adapter, not in shared code.
- **Test with an in-memory SQLite variant** that exercises the protocol interface. This both validates the abstraction and provides fast test execution.
- Do not use raw SQL. Use SQLAlchemy's ORM/Core consistently so the query layer is portable.

**Detection:** Functions that import sqlite3 directly. PRAGMA statements outside the storage adapter. Tests that only work with a file-based database.

**Phase relevance:** Phase 0 (define the storage protocol alongside the data model)

**Confidence:** MEDIUM -- standard software engineering practice, verified by SQLAlchemy's own documentation recommending dialect-agnostic patterns.

---

### Pitfall 7: Storing Large Context Blobs Inefficiently

**What goes wrong:** Commit content stored as large text blobs in SQLite degrades performance as traces grow. Queries slow down, database file grows excessively, and operations like diff/log that scan many commits become sluggish.

**Why it happens:** Each commit in Trace contains the full text content of a context block. For rich agent conversations, a single commit might be 1-10KB of text. A long trace with hundreds of commits across branches and subagents can easily reach tens of MB of text content. SQLite documentation shows that blobs under 256KB perform well in SQLite (35% faster than filesystem), but performance degrades for larger objects. More importantly, queries that touch many rows (log, diff across ranges) must deserialize all those blobs even if only metadata is needed.

**Prevention:**
- **Separate metadata from content** in the schema. Store commit metadata (hash, parent, timestamp, token_count, type, message) in one table and content blobs in another, joined by commit hash. This allows metadata-only queries (log, status, branch listing) to be fast.
- **Consider content deduplication.** If the same context block appears in multiple commits (e.g., after checkout and recommit), store it once and reference by hash, git-style.
- **Set a max content size** for commits. If a single commit exceeds a threshold (e.g., 100KB), warn the user -- they are probably committing an entire conversation when they should be committing individual turns.
- **Use incremental blob I/O** (sqlite3.Blob) for large content rather than loading entire rows into memory.
- **Consider compression** at the storage layer (zlib/lz4 on text content). Text compresses well (typically 3-5x) and the CPU cost is negligible compared to LLM API latency.

**Detection:** Slow `trace log` or `trace status` commands. Database file growing faster than expected. Memory spikes when scanning traces.

**Phase relevance:** Phase 0 (schema design). This is a schema-level decision that is painful to change after data exists.

**Confidence:** HIGH for the general principle (SQLite official docs on blob sizing). MEDIUM for the specific thresholds for Trace's usage patterns.

---

### Pitfall 8: Breaking API Changes After Early Adopters Commit

**What goes wrong:** The SDK API changes in incompatible ways after users have integrated Trace into their agent frameworks. Users pin to old versions or abandon the library.

**Why it happens:** Python library ecosystem has a documented pattern of excessive breaking changes (discussed on Hacker News, 2023). Early-stage libraries are especially prone because the API is still being discovered. Trace has additional pressure because the git-inspired API must balance familiarity with correctness for the context domain.

**Prevention:**
- **Version the API as 0.x until Phase 3 is complete.** Semantic versioning allows breaking changes in 0.x. Communicate this clearly in docs.
- **Separate the public API from internal implementation** from Phase 0. Use `__all__` and the underscore convention to mark internals. The public surface should be as small as possible.
- **Design for extension, not modification.** Materializers are already pluggable (good). Apply the same pattern to merge strategies, compression strategies, and storage backends. New behaviors should be addable without changing existing interfaces.
- **Deprecation before removal.** Even in 0.x, give at least one minor version of deprecation warnings before removing or renaming public API elements.
- **Type hints from day one.** Type-annotated APIs are harder to accidentally break because the type checker catches signature changes. Python typing adoption is increasing (Meta's Python Typing Survey 2025 shows quality and flexibility as top adoption reasons).

**Detection:** GitHub issues about API breakage. Users vendoring or pinning specific versions. Complaints about upgrade difficulty.

**Phase relevance:** Phases 0-1 (API design), ongoing through all phases.

**Confidence:** MEDIUM -- based on general Python ecosystem patterns, not Trace-specific data.

---

### Pitfall 9: Testing LLM-Dependent Operations Is Non-Deterministic

**What goes wrong:** Tests for compression, semantic merge, and other LLM-mediated operations are flaky because LLM outputs are non-deterministic. CI becomes unreliable. Developers skip tests or ignore failures.

**Why it happens:** LLMs produce different outputs for the same input across runs. Temperature=0 reduces but does not eliminate non-determinism (provider-side batching, model updates, etc.). The traditional testing approach of asserting exact outputs does not work.

**Prevention:**
- **Layer the testing strategy:**
  1. **Unit tests (deterministic):** Test all non-LLM logic with mocked LLM responses. Commit creation, DAG operations, branch management, storage, token counting, materialization -- all of these are deterministic and should have extensive unit tests with mocks.
  2. **Contract tests:** For LLM-mediated operations, test the contract, not the output. E.g., "after compression, the result should be shorter than the input and should contain these key terms." Use fuzzy assertions (contains, length ranges, semantic similarity scores).
  3. **Integration tests (expensive, infrequent):** Run against real LLM APIs in a separate CI job, not on every push. Accept some flakiness but track pass rates over time.
  4. **Snapshot tests with human review:** For semantic operations, capture LLM outputs and review them periodically. This catches quality regressions even if individual runs vary.

- **Mock at the LLM boundary, not deeper.** Define a `LLMClient` protocol with a `complete()` method. In tests, inject a mock that returns canned responses. This tests all Trace logic while eliminating non-determinism.
- **Use recorded cassettes** for integration tests (like VCR.py). Record real LLM responses once, replay them in CI.

**Detection:** Flaky CI. Tests that pass locally but fail in CI. Tests that developers skip or disable. Test suites that take minutes due to real API calls.

**Phase relevance:** Phase 0 (testing infrastructure), critical for Phase 2 (compression/merge tests).

**Confidence:** HIGH -- LLM testing challenges are extensively documented (Confident AI, Langfuse, LangChain testing guides, arxiv:2508.20737).

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable without significant rework.

---

### Pitfall 10: Commit Hash Collisions With Naive Hashing

**What goes wrong:** Using a simple hash of content for commit IDs leads to collisions when the same content is committed at different times or in different branches. Alternatively, including timestamp in the hash means the same content produces different hashes, breaking deduplication.

**Prevention:** Use a composite hash that includes: content + parent hash + timestamp + commit type. This mirrors git's approach (content + parent + author + timestamp). SHA-256 is sufficient; no need for cryptographic strength, just collision resistance.

**Phase relevance:** Phase 0 (data model)

**Confidence:** HIGH -- standard practice.

---

### Pitfall 11: CLI-First Design When SDK Is the Product

**What goes wrong:** Developer time is spent polishing CLI ergonomics when the primary consumers are agent frameworks using the Python SDK. CLI becomes the tested path while the SDK has gaps.

**Prevention:** Build and test the SDK first. The CLI should be a thin wrapper that calls SDK functions. Every CLI command maps to exactly one SDK call. If the SDK cannot do it, the CLI should not do it either. The PROJECT.md already states this intent -- enforce it in code review.

**Phase relevance:** Phase 1 (when CLI is introduced alongside core operations)

**Confidence:** HIGH -- follows directly from PROJECT.md's "API-first" constraint.

---

### Pitfall 12: Garbage Collection That Deletes Reachable Data

**What goes wrong:** GC policy deletes subagent traces or old branches that are still referenced by spawn pointers or that users intend to revisit. Data loss that cannot be recovered.

**Prevention:**
- GC must perform reachability analysis before deletion. Any commit reachable from any HEAD, branch, or spawn pointer is protected.
- Implement GC in two phases: mark (identify unreachable) then sweep (delete). Show the user what will be deleted before doing it (`trace gc --dry-run`).
- Add a "quarantine" period: unreachable data is marked for deletion but not actually deleted for N days. This provides a recovery window.

**Phase relevance:** Phase 3 (GC implementation)

**Confidence:** MEDIUM -- standard GC design principles applied to Trace's specific reference structure.

---

### Pitfall 13: Materialization Strategy That Ignores Prompt Engineering Best Practices

**What goes wrong:** The default materializer simply concatenates commits in order, producing a context window that is suboptimal for LLM consumption. Important information ends up in the "lost in the middle" zone.

**Prevention:**
- The default materializer should be simple concatenation (as planned), but it should be well-documented as a baseline, not an optimal strategy.
- Provide built-in "smart" materializers as optional strategies: one that places pinned commits at the start/end of context (where attention is highest), one that orders by relevance score.
- Expose the materializer protocol clearly so framework developers can implement domain-specific strategies.

**Phase relevance:** Phase 1 (context materialization design)

**Confidence:** HIGH for the "lost in the middle" problem (well-documented research). MEDIUM for the specific materializer strategies (requires experimentation).

---

### Pitfall 14: Edit Commits With Unclear Semantics

**What goes wrong:** The "Edit" commit type (modify existing context) has ambiguous behavior. Does it replace the original commit's content? Does it create a new commit that overrides the original during materialization? If the original is pinned, can it be edited? If the edit is compressed, does the original reappear?

**Prevention:**
- Resolve edit semantics decisively in Phase 0. The PROJECT.md already flags this as TBD ("in-place replacement vs override commit"). The recommendation: **override commit** semantics. An edit commit carries a reference to the commit it modifies and replaces its content during materialization. The original commit remains in the DAG for history/audit. This is cleaner than in-place mutation because it preserves the immutable commit model.
- Pin + Edit interaction must be defined: editing a pinned commit should produce a new unpinned commit that overrides it. The pin flag remains on the original.
- Document edge cases explicitly: what happens when you compress a range that includes both an original and its edit? (The edit's content should survive, the original's should not.)

**Phase relevance:** Phase 0 (data model design, edit commit semantics)

**Confidence:** MEDIUM -- this is a design decision, not a researched finding. The recommendation is based on git's immutable object model as precedent.

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|---------------|------------|
| Phase 0: Foundations | Designing the data model without accounting for order sensitivity (Pitfall 1) | Include an `order` or `position` field in the commit model. Test that materialization respects commit ordering within a branch. |
| Phase 0: Foundations | Token counting locked to tiktoken only (Pitfall 3) | Define `TokenCounter` as a Protocol. Provide tiktoken implementation but design for pluggability. |
| Phase 0: Foundations | Storage schema that mixes metadata and content (Pitfall 7) | Separate tables for commit metadata and commit content. Benchmark with 1000+ commits. |
| Phase 0: Foundations | No async-compatible internal design (Pitfall 5) | Use `async def` internally or ensure all IO goes through an interface that can become async. |
| Phase 1: Linear History | CLI absorbs development focus over SDK (Pitfall 11) | SDK-first development. CLI is a thin wrapper. Test the SDK directly. |
| Phase 1: Linear History | Materialization ignores position sensitivity (Pitfall 13) | Implement the simple materializer but document its limitations. Design the materializer protocol for extension. |
| Phase 1: Linear History | Edit commit semantics unresolved (Pitfall 14) | Resolve in Phase 0. Do not ship edit commits with ambiguous behavior. |
| Phase 2: Branching & Compression | Merge/rebase produces subtly wrong context (Pitfall 1) | LLM-mediated merge from day one. No textual merge fallback. Expose merge confidence to users. |
| Phase 2: Branching & Compression | Compression loses critical details silently (Pitfall 2) | Compression validation step. Pin commit integrity checks. Expose compression metrics. |
| Phase 2: Branching & Compression | Tests for LLM operations are flaky (Pitfall 9) | Layered testing strategy. Mock LLM boundary. Contract tests for semantic operations. |
| Phase 3: Multi-Agent | SQLite write contention from concurrent agents (Pitfall 4) | WAL mode, application-level write serialization, generous busy_timeout. |
| Phase 3: Multi-Agent | Orphaned traces from crashed agents (Pitfall 4) | Atomic spawn/collapse operations. `trace repair` command. Startup integrity check. |
| Phase 3: Multi-Agent | GC deletes reachable subagent data (Pitfall 12) | Reachability analysis, dry-run mode, quarantine period. |
| All Phases | Breaking API changes (Pitfall 8) | 0.x versioning, small public surface, type hints, deprecation warnings. |

---

## Sources

### HIGH Confidence (Official Documentation, Research Papers)
- [SQLite WAL mode documentation](https://sqlite.org/wal.html) -- concurrent access limitations
- [SQLite Internal vs External BLOBs](https://sqlite.org/intern-v-extern-blob.html) -- 256KB threshold for blob performance
- [SQLite thread safety documentation](https://www.sqlite.org/threadsafe.html) -- multi-thread modes
- [SQLAlchemy SQLite dialect docs](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html) -- connection pooling, NullPool default
- [Liu et al., "Lost in the Middle" (2023)](https://arxiv.org/abs/2307.03172) -- position bias in LLM context
- [MIT research on position bias (2025)](https://news.mit.edu/2025/unpacking-large-language-model-bias-0617) -- architectural roots of position bias
- [Understanding and Improving Information Preservation in Prompt Compression (2025)](https://arxiv.org/html/2503.19114) -- 3-55% performance loss from compression
- [Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (2025)](https://arxiv.org/html/2503.13657v1) -- MAST taxonomy, 14 failure modes
- [OpenAI tiktoken token counting discrepancy](https://community.openai.com/t/discrepancy-in-token-counts-between-tiktoken-and-api-usage-for-o4-mini-gpt-4o-mini/1271170) -- documented mismatch
- [Token Counting Guide: tiktoken, Anthropic, Gemini (2025)](https://www.propelcode.ai/blog/token-counting-tiktoken-anthropic-gemini-guide-2025) -- cross-provider tokenizer differences

### MEDIUM Confidence (Verified Across Multiple Sources)
- [JetBrains Research: Efficient Context Management (2025)](https://blog.jetbrains.com/research/2025/12/efficient-context-management/) -- context management strategies
- [mem0.ai: LLM Chat History Summarization Guide (2025)](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025) -- summarization approaches and limitations
- [Langfuse: Testing for LLM Applications (2025)](https://langfuse.com/blog/2025-10-21-testing-llm-applications) -- testing strategies
- [Seth Larson: Designing Libraries for Async and Sync IO](https://sethmlarson.dev/designing-libraries-for-async-and-sync-io) -- async/sync dual API patterns
- [Ben Hoyt: Designing Pythonic Library APIs](https://benhoyt.com/writings/python-api-design/) -- API ergonomics
- [Galileo: Why Multi-Agent LLM Systems Fail](https://galileo.ai/blog/multi-agent-llm-systems-fail) -- production failure analysis
- [SQLAlchemy "database is locked" discussion](https://github.com/sqlalchemy/sqlalchemy/discussions/11524) -- thread safety solutions

### LOW Confidence (Single Source, Requires Validation)
- [Factory.ai: Compressing Context](https://factory.ai/news/compressing-context) -- context compression approaches
- [Redis: Context Window Overflow (2026)](https://redis.io/blog/context-window-overflow/) -- overflow handling patterns
- [Agenta: Techniques to Manage Context Length](https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms) -- management strategies
