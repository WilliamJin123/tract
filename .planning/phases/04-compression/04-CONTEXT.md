# Phase 4: Compression - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Token-budget-aware compression of context history. Users can shrink commit chains into summaries while preserving pinned commits verbatim, reorder compiled context via compile-time parameters, and run garbage collection on unreachable commits. Compression is non-destructive by default (originals archived with queryable provenance).

</domain>

<decisions>
## Implementation Decisions

### Compression Granularity
- Default compress() summarizes all NORMAL priority commits; PINNED preserved verbatim in place; SKIP ignored entirely
- Scoping options: all commits (default), specific commits list, or from_commit/to_commit range
- Call-time `preserve` parameter acts as temporary pin for that invocation — does NOT override stored annotations
- PINNED is sacred: no call-time override can compress pinned commits. User must explicitly annotate(NORMAL) to un-pin first
- Compression can produce multiple summary commits — LLM clusters content naturally rather than flattening into one blob
- PINNED commits within a compressed range preserve their original position; summaries fill around them
- No automatic hierarchical compression — range support enables the pattern via composition (compress old stuff aggressively, recent stuff gently, in separate calls)

### Provenance & Archives
- Original commits stay in DB as unreachable after compression (non-destructive)
- CompressionRecord as first-class relational entity (not JSON metadata) for queryable provenance:
  - CompressionRow: compression_id, tract_id, created_at, original_tokens, compressed_tokens
  - CompressionSourceRow: compression_id + commit_hash (N source commits)
  - CompressionResultRow: compression_id + commit_hash + position (M summary commits)
- Queryable in both directions: "what sources produced this summary?" and "was this commit ever compressed?"
- Summary commits are plain APPEND operations with provenance tracked via CompressionRecord (no new operation type — APPEND/EDIT enum unchanged)

### Budget Targeting
- Compression is "make this smaller," not strict budget targeting
- Optional `target_tokens` parameter as hint/guideline to the LLM, not a hard constraint
- Token budget enforcement lives at the tract level (TractConfig.token_budget) at compile time, not at compression time
- Approximate targeting — single LLM pass, accept some variance, no expensive retry loops
- When content can't be meaningfully compressed (e.g., pinned commits alone are large): best effort + warning, never fail
- Rich CompressResult returned: original_tokens, compressed_tokens, source_commits, summary_commits, preserved_commits, compression_id

### Summary Format & Quality
- Summaries are single-message prose digests (flattened), not role-structured conversation replays
- Pinned commits preserve original role structure; summaries fill gaps between them as narrative prose
- Basic sanity validation: not empty, shorter than original, valid content type. No quality judgment — the review step handles that
- Optional `instructions` parameter appends to default prompt (e.g., "focus on architectural decisions")
- Optional `system_prompt` parameter overrides default prompt entirely
- Two separate parameters: `instructions` appends, `system_prompt` overrides. Clear intent from caller
- Default summarization prompt stored in a separate file (format at Claude's discretion)
- LLM infrastructure: reuse Phase 3 configure_llm() + resolver pattern by default, with option to swap in custom summarizer callable

### Autonomy Spectrum (Core Value #2)
- **Autonomous (DEFAULT)**: compress() — LLM summarizes and commits directly. Safe because originals are archived
- **Collaborative**: compress(auto_commit=False) — returns pending result with draft summaries, user reviews/edits, calls result.approve()
- **Manual**: compress(content='...') — user writes their own summary, no LLM involved
- Autonomous default differs from merge (which defaults to collaborative) because compression is non-destructive

### Commit Reordering (Compile-Time, Not DAG Mutation)
- Reordering is a parameter on compile(), NOT a standalone operation or DAG mutation
- User passes desired order: compile(order=[hash3, hash1, hash5, hash2, hash4])
- Commit history (DAG) stays truthful; only the compiled output order changes
- Full autonomy spectrum for ordering:
  - Manual: user provides exact order list
  - Heuristic: built-in strategies (e.g., group by role, important last)
  - LLM-suggested: Trace asks LLM for optimal ordering
- Two-tier semantic safety checks:
  - Structural (free/fast): EDIT before its target, broken response_to chains
  - LLM-powered (optional): "does this reordering change semantic meaning?"
- Satisfies roadmap success criteria #3 without a standalone reorder() operation

### Garbage Collection
- Two categories of unreachable commits:
  - **Archived**: compressed originals with CompressionRecord provenance — protected by default (never auto-cleaned)
  - **Truly orphaned**: no branch, no provenance pointer — fair game for GC
- Time-based retention (git model): GC only deletes orphans older than retention period
  - Default: orphan_retention=7d, archive_retention=never
  - User tunes: gc(orphan_retention="1d") or gc(archive_retention="30d")
- Explicit invocation only — no auto-GC. Developers prefer explicit control; easy to add auto later if demand shows
- Reachability wins over pins: if a pinned commit is truly unreachable (no branch, no compression record), GC can clean it after retention period. Pins protect against compression, not GC
- Tract-wide by default, optional branch scoping: gc(branch="main")
- Returns GCResult with stats: commits_removed, tokens_freed, archives_removed
- No dry-run — retention period IS the safety net

### Claude's Discretion
- Summary message role (system vs assistant)
- Default summarization prompt content and file format
- Natural preamble style for summaries (framing for downstream LLMs)
- Exact CompressResult / GCResult field names
- Compression clustering strategy (how LLM decides to split into multiple summaries)

</decisions>

<specifics>
## Specific Ideas

- Compression follows the same LLM client + custom callable pattern as merge (Phase 3) — consistent across all LLM-powered operations
- The pending result / approve() pattern for collaborative mode mirrors how merge review works
- Reordering via compile() avoids a whole new operation — "same building blocks, new parameter"
- CompressionRecord is relational (not JSON metadata) specifically for queryability — the user explicitly wanted provenance to be queryable

</specifics>

<deferred>
## Deferred Ideas

- Automatic hierarchical compression (Trace decides recency tiers and budgets automatically) — future enhancement if demand shows
- Auto-GC triggered after compress() — future enhancement if developers request it
- LLM-powered compression quality scoring (beyond basic sanity checks) — future enhancement
- Saved compile ordering presets / named views — future enhancement

</deferred>

---

*Phase: 04-compression*
*Context gathered: 2026-02-16*
