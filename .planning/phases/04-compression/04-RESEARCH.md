# Phase 4: Compression - Research

**Researched:** 2026-02-16
**Domain:** LLM context compression, commit history management, garbage collection
**Confidence:** HIGH

## Summary

Phase 4 adds four capabilities to Tract: (1) LLM-powered compression of commit ranges into summary commits, (2) pinned commit preservation during compression, (3) compile-time commit reordering with semantic safety checks, and (4) garbage collection of unreachable commits. This research analyzed the existing codebase thoroughly to understand integration points and confirm implementation patterns.

The codebase is well-prepared for Phase 4. The operations/ package pattern (established in Phase 2, extended in Phase 3) provides the exact blueprint for new compression and GC operations. The LLM client infrastructure (OpenAIClient + ResolverCallable protocol + OpenAIResolver) established in Phase 3 can be directly reused for summarization. The existing schema migration system (schema_version 1->2 in Phase 3) provides the pattern for adding CompressionRecord tables. The compiler already handles priority filtering (SKIP/NORMAL/PINNED) that compression must respect.

**Primary recommendation:** Implement in 3-4 plans: (1) CompressionRecord schema + storage + models, (2) compression engine + Tract facade, (3) compile-time reordering + GC, (4) optional CLI commands. Reuse the Phase 3 LLM client pattern (configure_llm + callable protocol) for summarization.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | >=2.0.46,<2.2 | CompressionRecord schema (3 new tables) | Already in use; proven pattern for schema evolution |
| Pydantic | >=2.10,<3.0 | CompressResult, GCResult, PendingCompression models | Already in use for all domain models |
| tiktoken | >=0.12.0 | Token counting for compression budgets | Already in use; TiktokenCounter protocol established |
| httpx | >=0.27,<1.0 | LLM calls for summarization | Already in use via OpenAIClient |
| tenacity | >=8.2,<10 | Retry logic for LLM summarization calls | Already in use via OpenAIClient |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| uuid | stdlib | compression_id generation | For CompressionRow primary keys |
| datetime | stdlib | Retention period calculations for GC | For time-based retention comparisons |
| dataclasses | stdlib | GCResult, OrderingSafetyResult | Lightweight frozen result types |

### Alternatives Considered

No new libraries needed. All Phase 4 requirements are satisfiable with the existing dependency set. The LLM summarization uses the same httpx+tenacity+OpenAI-compatible pattern from Phase 3.

## Architecture Patterns

### Recommended Project Structure

```
src/tract/
  models/
    compression.py     # CompressResult, PendingCompression, GCResult, ReorderWarning
  storage/
    schema.py          # +CompressionRow, CompressionSourceRow, CompressionResultRow (3 new tables)
    repositories.py    # +CompressionRepository ABC
    sqlite.py          # +SqliteCompressionRepository
  engine/
    compiler.py        # Extended: compile(order=...) parameter
  operations/
    compression.py     # compress(), gc(), reorder safety checks
  prompts/
    summarize.py       # Default summarization prompt (Python module, not text file)
  tract.py             # +compress(), +gc(), extended compile()
```

### Pattern 1: Operations Module Pattern (established Phase 2+3)

**What:** Higher-level operations are standalone functions in `operations/` that receive repository and engine dependencies as parameters, not methods on a god class.

**When to use:** Any new composite operation that coordinates multiple storage primitives.

**Example (from existing merge.py):**
```python
# operations/compression.py follows the exact same pattern as operations/merge.py:
# - Standalone functions receiving repos as params
# - Returns rich result dataclass
# - Tract facade method delegates to operations function

def compress_range(
    tract_id: str,
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    annotation_repo: AnnotationRepository,
    ref_repo: RefRepository,
    commit_engine: CommitEngine,
    token_counter: TokenCounter,
    compression_repo: CompressionRepository,
    *,
    commits: list[str] | None = None,
    from_commit: str | None = None,
    to_commit: str | None = None,
    target_tokens: int | None = None,
    preserve: list[str] | None = None,
    auto_commit: bool = True,
    llm_client: LLMClient | None = None,
    content: str | None = None,
    instructions: str | None = None,
    system_prompt: str | None = None,
) -> CompressResult | PendingCompression:
    ...
```

**Source:** `src/tract/operations/merge.py` lines 238-400, `src/tract/tract.py` lines 859-944

### Pattern 2: LLM Client Reuse Pattern (established Phase 3)

**What:** Phase 3 established `configure_llm()` which stores `self._llm_client` and creates `self._default_resolver`. For compression, the same client is reused for summarization calls. Custom summarizer callables follow the same pluggability pattern as `ResolverCallable`.

**When to use:** Any LLM-powered operation.

**Key integration points:**
- `Tract.configure_llm(client)` already stores `self._llm_client`
- `OpenAIClient.chat()` for sending summarization prompts
- `OpenAIClient.extract_content()` for extracting summary text from response
- Custom summarizer callable follows `ResolverCallable`-like protocol

### Pattern 3: Schema Migration Pattern (established Phase 3)

**What:** `init_db()` checks `schema_version` in `_trace_meta` and runs migrations. Phase 3 added migration from v1->v2 (CommitParentRow table). Phase 4 will add v2->v3 migration (CompressionRecord tables).

**When to use:** Any time new tables or columns are added.

**Example (from existing engine.py):**
```python
# init_db() already handles v1->v2 migration
# Phase 4 adds v2->v3:
elif existing.value == "2":
    # Migrate v2 -> v3: create compression record tables
    for table_name in ["compressions", "compression_sources", "compression_results"]:
        Base.metadata.tables[table_name].create(engine, checkfirst=True)
    existing.value = "3"
    session.commit()
```

**Source:** `src/tract/storage/engine.py` lines 48-72

### Pattern 4: Result Model + Review Flow (established Phase 3 merge)

**What:** MergeResult demonstrates the "return rich result for review, then commit" pattern. CompressResult and PendingCompression follow the same pattern for the collaborative autonomy mode.

**When to use:** When an operation has autonomous/collaborative/manual modes.

**Example (from existing merge.py):**
```python
# MergeResult has committed=False initially
# User reviews, calls commit_merge()
# Same pattern for PendingCompression:
result = t.compress(auto_commit=False)  # Returns PendingCompression
result.edit_summary(0, "Better summary text")
result.approve()  # Commits the summaries
```

**Source:** `src/tract/models/merge.py` lines 48-82, `src/tract/tract.py` lines 946-1014

### Pattern 5: Compile Parameter Extension (compile() already supports at_time, at_commit)

**What:** `compile()` already accepts optional parameters that modify compilation behavior. Adding `order` follows the same pattern -- it's just another parameter that alters how commits are assembled into messages.

**When to use:** Extending compilation behavior without changing the core protocol.

**Key consideration:** The `ContextCompiler` protocol signature in `protocols.py` (line 87) defines the interface. Adding `order` must either be optional in the protocol or handled before the compiler is called (at the Tract facade level by reordering the snapshot/compiled result).

**Recommendation:** Handle reordering at the **Tract facade level**, not in the compiler protocol. The compiler returns messages in commit order; `Tract.compile()` reorders them post-compilation. This avoids protocol changes and keeps the compiler simple. The `ContextCompiler` protocol stays unchanged.

### Anti-Patterns to Avoid

- **New operation type for summaries:** CONTEXT.md explicitly states "no new operation type -- APPEND/EDIT enum unchanged." Summary commits use `CommitOperation.APPEND` with provenance tracked via `CompressionRecord`.
- **Strict budget enforcement at compression time:** CONTEXT.md says "Compression is make-this-smaller, not strict budget targeting." Never fail because the LLM produced output above target_tokens. Accept variance.
- **DAG mutation for reordering:** CONTEXT.md says "Reordering is a parameter on compile(), NOT a standalone operation or DAG mutation." The commit chain stays truthful.
- **Auto-GC triggers:** CONTEXT.md says "Explicit invocation only -- no auto-GC."
- **Compressing PINNED commits:** CONTEXT.md says "PINNED is sacred: no call-time override can compress pinned commits."

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Token counting | Custom tokenizer | `TiktokenCounter.count_text()` / `count_messages()` | Already handles message overhead, encoding differences |
| Commit creation | Manual blob/hash/ref management | `CommitEngine.create_commit()` | Handles blob dedup, hash computation, HEAD update, auto-annotations |
| Commit chain walking | Manual SQL parent traversal | `CommitRepository.get_ancestors()` / DAG utilities | Batch-loads all tract commits, handles merge parents |
| LLM calls | Raw httpx calls | `OpenAIClient.chat()` | Has retry logic, error classification, rate limit handling |
| Schema migration | Manual table creation | `init_db()` migration path | Handles version checking, idempotent creation |
| Reachability analysis | Custom graph traversal | `dag.get_all_ancestors()` / `dag._bfs_walk()` | Already handles merge parents, cycle-safe BFS |

**Key insight:** Phase 4 builds on Phase 1-3 infrastructure. Nearly every "hard" problem (token counting, commit creation, LLM calls, DAG traversal) already has a tested solution. The new work is in the orchestration layer (operations/compression.py) and new schema (CompressionRecord tables).

## Common Pitfalls

### Pitfall 1: HEAD Update After Compression

**What goes wrong:** After compressing commits and creating summary commits, HEAD must be updated to point to the new chain tip. If the compressed range includes the current HEAD, the new HEAD must be the last summary commit.

**Why it happens:** Compression removes commits from the reachable chain and inserts new ones. If the original HEAD was in the compressed range, the chain is severed.

**How to avoid:** After compression: (1) Create summary commits as children of the pre-range parent, (2) If any PINNED commits were in the range, interleave them with summaries in position order, (3) The last summary/pinned commit becomes the new chain tip, (4) Update the branch ref to point to the new tip. Use `CommitEngine.create_commit()` which handles HEAD updates automatically -- commit summaries sequentially so each one's parent is the previous one.

**Warning signs:** Tests where `t.head` returns a commit hash from the original (now-unreachable) chain.

### Pitfall 2: Session Commit Timing for Multi-Commit Compression

**What goes wrong:** Compression creates multiple DB objects (CompressionRow, CompressionSourceRow, CompressionResultRow, new BlobRow, new CommitRow) and must be atomic.

**Why it happens:** `CommitEngine.create_commit()` does NOT call `session.commit()` -- the `Tract.commit()` method does. But compression needs to create commits AND record provenance atomically.

**How to avoid:** Use the same pattern as `Tract.batch()`: defer `session.commit()` until all summary commits and CompressionRecord entries are created. Then call `session.commit()` once at the end of `Tract.compress()`. The individual `CommitEngine.create_commit()` calls use `session.flush()` internally, which stages changes without committing.

**Warning signs:** Partial compression records in the database (some source rows without results or vice versa).

### Pitfall 3: PINNED Commit Interleaving in Compressed Ranges

**What goes wrong:** If a compressed range contains PINNED commits at positions 3 and 7 (out of 10), the output must be: [summary_before_3, PINNED_3, summary_3_to_7, PINNED_7, summary_after_7]. Getting the interleaving wrong breaks the semantic ordering.

**Why it happens:** The LLM produces summaries for the NORMAL commits in groups, but the summaries must be interleaved with the preserved PINNED commits in their original positions.

**How to avoid:** Before calling the LLM: (1) Walk the commit range, (2) Identify PINNED commits and their positions, (3) Partition NORMAL commits into groups (between PINNED commits), (4) Summarize each group independently, (5) Reconstruct the chain: [summary_group_1, PINNED_1, summary_group_2, PINNED_2, ...]. Each segment feeds the LLM separately.

**Warning signs:** Tests where PINNED commits appear at the end of the compiled output rather than in their original position relative to summaries.

### Pitfall 4: Reachability Calculation in GC

**What goes wrong:** GC removes "unreachable" commits, but the definition of reachable must include ALL branches, not just the current branch.

**Why it happens:** A commit might be unreachable from `main` but still reachable from a feature branch.

**How to avoid:** Use `RefRepository.list_branches()` to get all branch tips, then `dag.get_all_ancestors()` for each tip. Union all ancestor sets. Any commit not in the union AND not referenced by CompressionRecord (unless archive_retention exceeded) is eligible for GC.

**Warning signs:** GC removes commits that are still reachable from other branches.

### Pitfall 5: Cache Invalidation After Compression

**What goes wrong:** The compile cache holds snapshots keyed by head_hash. After compression, the HEAD changes to a new commit with a new hash, but old cache entries for commits in the compressed range become stale (they reference commits that are now unreachable).

**Why it happens:** The LRU cache doesn't know about compression. It still holds snapshots for old HEAD positions.

**How to avoid:** Clear the entire compile cache after compression (same pattern as `Tract.merge()` on line 942: `self._cache.clear()`). The next `compile()` call rebuilds from the new chain.

**Warning signs:** Stale messages appearing in compiled output after compression.

### Pitfall 6: Summary Commit Content Type

**What goes wrong:** Summary commits need a content_type that maps correctly to an LLM role in compilation.

**Why it happens:** Summaries are prose digests, not structured dialogue. Using "dialogue" forces choosing a role. Using "freeform" produces JSON-serialized output rather than prose.

**How to avoid:** Use `DialogueContent(role="assistant", text="...")` for summary commits. The compiler maps "dialogue" content_type to the role field from the content itself (see `compiler.py` line 384-385: `if content_type == "dialogue": return content_data.get("role", "user")`). Using role="assistant" with prose text produces the right output. Alternative: use `FreeformContent` with a text payload, but DialogueContent is semantically cleaner.

**Recommendation:** Use role="assistant" for summaries. This positions summaries as the assistant recapping prior context, which is the most natural framing for downstream LLMs.

## Code Examples

### Example 1: Creating CompressionRecord Tables (Schema)

```python
# In storage/schema.py -- 3 new ORM classes

class CompressionRow(Base):
    """Records a single compression operation for provenance tracking."""
    __tablename__ = "compressions"

    compression_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    branch_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    original_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    compressed_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    target_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class CompressionSourceRow(Base):
    """Maps source commits to a compression operation."""
    __tablename__ = "compression_sources"

    compression_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("compressions.compression_id"), primary_key=True
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

class CompressionResultRow(Base):
    """Maps result summary commits to a compression operation."""
    __tablename__ = "compression_results"

    compression_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("compressions.compression_id"), primary_key=True
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
```

### Example 2: Compression Operation Core Flow

```python
# In operations/compression.py -- follows merge.py pattern

def compress_range(
    tract_id, commit_repo, blob_repo, annotation_repo, ref_repo,
    commit_engine, token_counter, compression_repo, parent_repo,
    *, commits=None, from_commit=None, to_commit=None,
    target_tokens=None, preserve=None, auto_commit=True,
    llm_client=None, content=None, instructions=None, system_prompt=None,
):
    # 1. Resolve commit range
    head = ref_repo.get_head(tract_id)
    all_commits = list(reversed(list(commit_repo.get_ancestors(head))))

    # 2. Filter to range
    range_commits = _resolve_range(all_commits, commits, from_commit, to_commit)

    # 3. Classify by priority
    pinned, normal, skip = _classify_by_priority(
        range_commits, annotation_repo, preserve
    )

    # 4. Partition normal commits around pinned positions
    groups = _partition_around_pinned(range_commits, pinned, normal)

    # 5. Generate summaries (manual, autonomous, or collaborative)
    if content is not None:
        # Manual mode: user provides content directly
        summaries = [content]
    elif llm_client is not None:
        # LLM mode: summarize each group
        summaries = _summarize_groups(groups, llm_client, target_tokens,
                                       instructions, system_prompt, token_counter)
    else:
        raise CompressionError("No LLM client and no manual content provided")

    # 6. Create compression record + summary commits (or return pending)
    if auto_commit:
        return _commit_compression(...)
    else:
        return PendingCompression(summaries=summaries, ...)
```

### Example 3: Compile-Time Reordering at Facade Level

```python
# In tract.py -- Tract.compile() extended

def compile(self, *, at_time=None, at_commit=None,
            include_edit_annotations=False, order=None):
    # ... existing logic ...
    result = self._compiler.compile(self._tract_id, current_head, ...)

    # Reorder if requested
    if order is not None:
        result = self._reorder_compiled(result, order)

    return result

def _reorder_compiled(self, result, order):
    """Reorder compiled messages according to user-specified commit hash order."""
    # Build hash -> index mapping
    hash_to_idx = {h: i for i, h in enumerate(result.commit_hashes)}

    # Validate all hashes in order exist in result
    for h in order:
        if h not in hash_to_idx:
            raise CommitNotFoundError(h)

    # Reorder messages, configs, and hashes
    new_indices = [hash_to_idx[h] for h in order]
    # Include any commits not in order at their original position
    remaining = [i for i in range(len(result.messages)) if i not in set(new_indices)]
    final_order = new_indices + remaining

    new_messages = [result.messages[i] for i in final_order]
    new_configs = [result.generation_configs[i] for i in final_order]
    new_hashes = [result.commit_hashes[i] for i in final_order]

    # Recount tokens (message overhead can change with order)
    token_count = self._token_counter.count_messages(...)

    return CompiledContext(
        messages=new_messages, token_count=token_count,
        commit_count=result.commit_count, token_source=result.token_source,
        generation_configs=new_configs, commit_hashes=new_hashes,
    )
```

### Example 4: GC Reachability Analysis

```python
# In operations/compression.py

def gc(
    tract_id, commit_repo, ref_repo, parent_repo, blob_repo,
    compression_repo, *, orphan_retention_days=7,
    archive_retention_days=None, branch=None,
):
    # 1. Find all reachable commits (from all branches or specified branch)
    if branch:
        branches = [branch]
    else:
        branches = ref_repo.list_branches(tract_id)

    reachable = set()
    for b in branches:
        tip = ref_repo.get_branch(tract_id, b)
        if tip:
            reachable |= dag.get_all_ancestors(tip, commit_repo, parent_repo)

    # 2. Find all commits in tract
    all_commits = _get_all_commits(commit_repo, tract_id)

    # 3. Find unreachable
    unreachable = {c for c in all_commits if c.commit_hash not in reachable}

    # 4. Classify: archived vs orphaned
    archived = set()
    orphaned = set()
    for c in unreachable:
        if compression_repo.is_source_of(c.commit_hash):
            archived.add(c)
        else:
            orphaned.add(c)

    # 5. Apply retention policies
    # IMPORTANT: SQLite stores naive datetimes. Use _normalize_dt() to strip
    # tzinfo before comparison, matching the pattern in compiler.py.
    now = _normalize_dt(datetime.now(timezone.utc))
    to_remove = set()

    for c in orphaned:
        age = (now - c.created_at).days
        if age >= orphan_retention_days:
            to_remove.add(c)

    if archive_retention_days is not None:
        for c in archived:
            age = (now - c.created_at).days
            if age >= archive_retention_days:
                to_remove.add(c)

    # 6. Delete commits and orphaned blobs
    tokens_freed = sum(c.token_count for c in to_remove)
    _delete_commits(to_remove, commit_repo, blob_repo, compression_repo)

    return GCResult(
        commits_removed=len(to_remove),
        tokens_freed=tokens_freed,
        archives_removed=len(to_remove & archived),
    )
```

### Example 5: Default Summarization Prompt

```python
# In prompts/summarize.py

DEFAULT_SUMMARIZE_SYSTEM = (
    "You are a context summarizer for an AI assistant's conversation history. "
    "Your task is to compress a sequence of conversation messages into a concise "
    "prose summary that preserves all critical information, key decisions, and "
    "important context. The summary will replace the original messages in the "
    "assistant's context window.\n\n"
    "Guidelines:\n"
    "- Write in third-person narrative prose, not as a conversation replay\n"
    "- Preserve specific details: names, numbers, code snippets, decisions\n"
    "- Prioritize information that would affect future conversation quality\n"
    "- Omit pleasantries, greetings, and redundant acknowledgments\n"
    "- If a target token count is specified, aim for approximately that length\n"
    "- Begin with a brief framing sentence like 'Previously in this conversation:'\n"
)

def build_summarize_prompt(
    messages_text: str,
    *,
    target_tokens: int | None = None,
    instructions: str | None = None,
) -> str:
    """Build the user prompt for summarization."""
    parts = [f"Summarize the following conversation segment:\n\n{messages_text}"]
    if target_tokens is not None:
        parts.append(f"\nTarget approximately {target_tokens} tokens.")
    if instructions:
        parts.append(f"\nAdditional instructions: {instructions}")
    return "\n".join(parts)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full cache invalidation on EDIT | In-memory snapshot patching | Phase 1.4 | EDIT + compress won't need full recompile |
| Single-snapshot cache | LRU cache (maxsize=8) | Phase 1.4 | Branch-switching after compress is O(1) cache hit |
| No LLM integration | OpenAIClient + ResolverCallable | Phase 3 | Summarization reuses proven infrastructure |
| No merge commits | CommitParentRow DAG | Phase 3 | GC reachability analysis handles merge parents |

**Already established patterns that Phase 4 builds on:**
- `operations/` package for composite operations (Phase 2)
- `LLMClient` protocol + `configure_llm()` for LLM operations (Phase 3)
- `MergeResult` + `commit_merge()` for review-then-commit flow (Phase 3)
- Schema version migration in `init_db()` (Phase 3)
- `dag.get_all_ancestors()` for reachability analysis (Phase 3)

## Claude's Discretion Recommendations

### Summary Message Role: `assistant`

**Recommendation:** Use `DialogueContent(role="assistant", text="...")` for summary commits.

**Reasoning:**
- The compiler already handles `dialogue` content_type by using the role field from the content itself (compiler.py line 384)
- "assistant" role positions the summary as the AI recapping prior context, which is semantically correct
- "system" role could work but is typically reserved for instructions, not conversation history
- DialogueContent is the cleanest fit because it has a `text` field and a `role` field, exactly what summaries need

### Default Summarization Prompt: Python Module

**Recommendation:** Store the default prompt in `src/tract/prompts/summarize.py` as Python string constants, not a text file.

**Reasoning:**
- Python module allows `build_summarize_prompt()` helper function alongside the constant
- Easier to import and test than reading a text file at runtime
- Follows the pattern of `OpenAIResolver._default_system_prompt()` being inline code
- No file I/O concerns, no path resolution issues
- The `instructions` append and `system_prompt` override are cleanly handled as function parameters

### Natural Preamble Style

**Recommendation:** Summaries begin with "Previously in this conversation:" as a natural framing sentence.

**Reasoning:**
- Concise, recognizable as a recap (LLMs trained on similar patterns)
- Distinguishes summaries from original content
- Not too formal (avoids "Summary of prior context:")
- Short enough to not waste tokens

### CompressResult / GCResult Field Names

**Recommendation:**

```python
@dataclass(frozen=True)
class CompressResult:
    compression_id: str
    original_tokens: int
    compressed_tokens: int
    source_commits: list[str]       # Commit hashes of compressed originals
    summary_commits: list[str]      # Commit hashes of new summaries
    preserved_commits: list[str]    # Commit hashes of PINNED commits that survived
    compression_ratio: float        # compressed_tokens / original_tokens
    new_head: str                   # The new HEAD after compression

@dataclass(frozen=True)
class PendingCompression:
    """Returned when auto_commit=False. User reviews then calls approve()."""
    summaries: list[str]            # Draft summary texts
    source_commits: list[str]
    preserved_commits: list[str]
    original_tokens: int
    estimated_tokens: int           # Estimated compressed tokens

    def edit_summary(self, index: int, new_text: str) -> None: ...
    def approve(self) -> CompressResult: ...

@dataclass(frozen=True)
class GCResult:
    commits_removed: int
    blobs_removed: int
    tokens_freed: int
    archives_removed: int           # Compression source commits removed
    duration_seconds: float

@dataclass(frozen=True)
class ReorderWarning:
    """Semantic safety warning from compile-time reordering."""
    warning_type: str               # "edit_before_target", "response_chain_break", "semantic_change"
    commit_hash: str
    description: str
    severity: str                   # "structural" (free) or "semantic" (LLM-detected)
```

### Compression Clustering Strategy

**Recommendation:** Single-pass sequential grouping with PINNED commit boundaries as natural cluster breaks.

**Algorithm:**
1. Walk the commit range in order
2. When a PINNED commit is encountered, it ends the current group
3. Each group of consecutive NORMAL commits becomes one summary
4. If a group is large (>20 commits or >8000 tokens), split at natural boundaries (content_type changes)
5. Each sub-group gets one LLM call, producing one summary commit

**Reasoning:**
- PINNED commits are natural semantic boundaries (user explicitly marked them as important)
- One LLM call per group avoids the complexity of multi-round compression
- Content-type changes are cheap structural boundaries for splitting large groups
- Aligns with CONTEXT.md: "LLM clusters content naturally rather than flattening into one blob"
- No hierarchical compression needed (CONTEXT.md: "No automatic hierarchical compression")

## Codebase Integration Points (Detailed)

### Integration Point 1: Tract Facade (tract.py)

New public methods to add:

| Method | Lines Needed | Delegates To |
|--------|-------------|-------------|
| `compress()` | ~50 | `operations.compression.compress_range()` |
| `gc()` | ~30 | `operations.compression.gc()` |
| `compile(order=...)` | ~30 extension | `self._reorder_compiled()` new private method |

The `Tract.__init__()` needs a `compression_repo` parameter (like `parent_repo` was added in Phase 3). The `Tract.open()` factory creates `SqliteCompressionRepository(session)`.

### Integration Point 2: Storage Layer

- `schema.py`: 3 new ORM classes (~60 lines)
- `repositories.py`: 1 new ABC class `CompressionRepository` (~40 lines)
- `sqlite.py`: 1 new implementation `SqliteCompressionRepository` (~80 lines)
- `engine.py`: `init_db()` migration v2->v3 (~10 lines)

### Integration Point 3: Compile Cache

After compression, call `self._cache.clear()` (same as merge, line 942).

### Integration Point 4: LLM Client

Reuse `self._llm_client` (set by `configure_llm()`). For summarization, call `self._llm_client.chat()` directly with the summarization prompt (no need for a Resolver -- that's for conflict resolution. Summarization is simpler: send content, get summary back).

### Integration Point 5: Priority System

The annotation repository's `batch_get_latest()` efficiently fetches priorities for a range of commits. Compression uses this to classify commits as PINNED/NORMAL/SKIP before proceeding.

## Open Questions

1. **PendingCompression mutability**
   - What we know: MergeResult uses `edit_resolution()` for mutation, is a Pydantic model
   - What's unclear: PendingCompression needs `approve()` method that creates commits. This requires access to repos (via closure or stored reference)
   - Recommendation: PendingCompression stores a reference to a `_commit_fn` callable (closure capturing all repo dependencies) that `approve()` invokes. Or make PendingCompression a non-frozen dataclass with mutable summaries list and pass it back to `Tract.approve_compression(pending)` method. The latter follows MergeResult + `commit_merge()` pattern more closely.

2. **Blob cleanup in GC**
   - What we know: Blobs are content-addressed. Multiple commits can reference the same blob.
   - What's unclear: When GC removes a commit, should it also remove the blob if no other commit references it?
   - Recommendation: Yes, but requires a reference count check. After removing commits, query for blobs where no CommitRow references their content_hash. Delete those orphaned blobs. The `GCResult.blobs_removed` field captures this.

3. **Reordering and the compile cache**
   - What we know: Compile cache stores snapshots keyed by head_hash. Reordered compiles would have the same head_hash but different message order.
   - What's unclear: Should reordered compiles bypass the cache?
   - Recommendation: Yes, bypass cache when `order` parameter is provided. Reordered compiles are special views, not the default. Cache serves the common case (default order). This avoids cache key complexity.

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis of all source files in `src/tract/` (23 files read)
- Existing patterns from Phase 3 merge implementation (`operations/merge.py`, `llm/client.py`, `llm/resolver.py`)
- Schema migration pattern from `storage/engine.py` init_db()
- CONTEXT.md locked decisions for Phase 4

### Secondary (MEDIUM confidence)
- Codebase conventions inferred from 3 complete phases and 489 passing tests
- Operations package pattern consistency across `operations/{merge,rebase,dag,branch,navigation,diff,history}.py`

### Tertiary (LOW confidence)
- None. All findings are based on direct codebase analysis.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new libraries needed; all existing dependencies suffice
- Architecture: HIGH - Patterns are directly copied from established Phase 2/3 code
- Pitfalls: HIGH - Identified from direct analysis of existing code behavior (HEAD management, session timing, cache invalidation)

**Research date:** 2026-02-16
**Valid until:** 2026-03-16 (stable -- no external dependencies to go stale)
