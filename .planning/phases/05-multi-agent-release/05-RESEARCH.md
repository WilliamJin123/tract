# Phase 5: Multi-Agent & Release - Research

**Researched:** 2026-02-16
**Domain:** Multi-agent coordination, SQLite concurrency, Python packaging
**Confidence:** MEDIUM (research validated against official docs and codebase analysis)

## Summary

Phase 5 adds multi-agent coordination (spawn/collapse/session) on top of the existing single-agent infrastructure (Phases 1-4, 563 tests) and prepares the library for pip release. The research investigated seven key areas: SQLite concurrent write behavior (flagged as a concern), FTS5 vs LIKE for cross-repo search, Python packaging best practices, session commit type design, spawn pointer table schema, collapse prompt patterns, and clone/copy implementation strategies.

The primary finding is that **single shared DB with WAL mode is viable for Phase 5's multi-agent use case**, provided that each agent uses its own SQLAlchemy Session (one session per thread) and write transactions are kept short. The existing `create_trace_engine()` already sets `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`, which are the recommended settings. For cross-repo search, simple LIKE queries are recommended over FTS5 for v1, deferring FTS5 to a future optimization pass.

A critical finding is that the **PyPI package name "tract" is already taken** by a neural network inference library (sonos/tract, v0.21.13, actively maintained). The project must choose an alternative PyPI distribution name while keeping `tract` as the import name.

**Primary recommendation:** Implement single shared DB with one SQLAlchemy engine + per-agent sessions, add a `spawn_pointers` table at schema v4, reuse compression engine for collapse with a new prompt template, and use LIKE-based search for v1. Address the PyPI naming conflict before release.

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | >=2.0.46,<2.2 | ORM + session management | Already used; provides per-thread session isolation |
| Pydantic | >=2.10,<3.0 | Domain models | Already used for content types, CommitInfo, etc. |
| tiktoken | >=0.12.0 | Token counting | Already used for token budgets |
| httpx | >=0.27,<1.0 | LLM HTTP client | Already used for OpenAI-compatible APIs |
| tenacity | >=8.2,<10 | Retry logic | Already used for LLM retries |
| hatchling | (build) | Build backend | Already configured in pyproject.toml |

### Supporting (No New Dependencies Needed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| uuid (stdlib) | - | Tract ID generation | Already used for tract_id auto-generation |
| threading (stdlib) | - | Thread-local session management | For Session wrapper thread safety |
| contextlib (stdlib) | - | Context managers | For Session.open() context management |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| LIKE search | SQLite FTS5 | FTS5 is faster for large datasets but requires virtual table management, trigger sync, and has no native SQLAlchemy ORM support. LIKE is sufficient for v1 scale. |
| Single shared DB | Separate DB per tract | Separate DBs eliminate write contention entirely but make cross-repo queries require ATTACH (max 10 in SQLite) or multiple connections. Single DB with JOINs is simpler and matches the CONTEXT.md preference. |
| hatchling | setuptools/flit | Hatchling is already configured. No reason to switch. PEP 621 compliant. |

## Architecture Patterns

### Recommended File Structure (New/Modified Files)
```
src/tract/
  session.py              # NEW: Session class (thin wrapper over shared DB)
  models/
    session.py            # NEW: SessionContent, SpawnInfo models
  operations/
    spawn.py              # NEW: spawn, collapse, clone operations
    session_ops.py        # NEW: cross-repo queries, timeline, search
  prompts/
    summarize.py          # MODIFIED: add collapse-specific prompt
  storage/
    schema.py             # MODIFIED: add SpawnPointerRow, schema v4
    repositories.py       # MODIFIED: add SpawnPointerRepository ABC
    sqlite.py             # MODIFIED: add SqliteSpawnPointerRepository
    engine.py             # MODIFIED: v3->v4 migration, engine config for multi-agent
  tract.py                # MODIFIED: add parent(), children(), spawn() helpers
  __init__.py             # MODIFIED: export Session and new models
```

### Pattern 1: Single Engine, Multiple Sessions (Multi-Agent Concurrency)
**What:** Each agent (Tract) gets its own SQLAlchemy Session from a shared engine. The Session wrapper manages the engine lifecycle.
**When to use:** Always in multi-agent scenarios.
**Confidence:** HIGH (validated against SQLAlchemy docs and existing codebase pattern)

```python
# Session creates the shared engine and provides per-agent sessions
class Session:
    def __init__(self, engine: Engine, session_factory: sessionmaker):
        self._engine = engine
        self._session_factory = session_factory
        self._tracts: dict[str, Tract] = {}

    @classmethod
    def open(cls, path: str, *, autonomy: str = "collaborative") -> Session:
        """Open a multi-agent session backed by a single SQLite file."""
        engine = create_trace_engine(path)
        init_db(engine)
        session_factory = create_session_factory(engine)
        return cls(engine, session_factory)

    def create_tract(self, *, display_name: str | None = None) -> Tract:
        """Create a new tract with its own session."""
        session = self._session_factory()
        tract_id = uuid.uuid4().hex
        # Build Tract with the shared engine but private session
        tract = Tract(engine=self._engine, session=session, ...)
        self._tracts[tract_id] = tract
        return tract
```

### Pattern 2: Spawn Pointer Table (Cross-Tract Linkage)
**What:** A lightweight table linking parent and child tracts with provenance metadata. Status is derived from commits, not stored mutably.
**When to use:** Every spawn/collapse operation.
**Confidence:** HIGH (consistent with existing commit-as-truth pattern)

```python
class SpawnPointerRow(Base):
    __tablename__ = "spawn_pointers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    parent_commit_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash", ondelete="SET NULL"),
        nullable=True,
    )
    child_tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    inheritance_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    # "full_clone", "head_snapshot", "selective"
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_spawn_parent_tract", "parent_tract_id"),
        Index("ix_spawn_child_tract", "child_tract_id"),
    )
```

### Pattern 3: Collapse Reuses Compression Pipeline
**What:** Collapse calls the same `compress_range()` function from Phase 4 but with a collapse-specific prompt. The summary commit is created in the parent tract, not the child.
**When to use:** When collapsing a subagent trace back into the parent.
**Confidence:** HIGH (reuse of existing tested infrastructure)

```python
def collapse_subagent(
    parent_tract: Tract,
    child_tract: Tract,
    spawn_pointer: SpawnPointerRow,
    *,
    content: str | None = None,     # Manual mode
    instructions: str | None = None, # LLM instructions
    auto_commit: bool = True,        # Autonomy control
) -> CollapseResult:
    """Collapse child tract history into a summary commit in the parent."""
    # 1. Compile the child's full context
    child_context = child_tract.compile()

    # 2. Generate summary (reuse compression prompt with collapse variant)
    if content is None:
        summary = _summarize_for_collapse(
            child_context, spawn_pointer.purpose, llm_client, ...
        )
    else:
        summary = content

    # 3. Create APPEND commit in parent tract with metadata
    parent_tract.commit(
        DialogueContent(role="assistant", text=summary),
        message=f"Collapsed subagent: {spawn_pointer.purpose}",
        metadata={
            "collapse_source_tract_id": child_tract.tract_id,
            "collapse_source_head": child_tract.head,
            "spawn_pointer_id": spawn_pointer.id,
        },
    )
```

### Pattern 4: Session Commit Type as New Content Type
**What:** A new `SessionContent` content type (not a new CommitOperation) for session boundary commits.
**When to use:** Session transitions, handoffs, resume points.
**Confidence:** MEDIUM (design choice; new content type is less invasive than new operation enum)

```python
class SessionContent(BaseModel):
    """Session boundary marker with context for handoff."""
    content_type: Literal["session"] = "session"
    session_type: Literal["start", "end", "handoff", "checkpoint"]
    summary: str
    decisions: list[str] = []
    failed_approaches: list[str] = []
    next_steps: list[str] = []
```

**Rationale:** Adding a new `CommitOperation` enum value (e.g., `SESSION`) would require updating every switch/match on CommitOperation across the codebase (compiler, compressor, diff, etc.). A new content type is additive and requires no changes to existing logic -- the commit is just an APPEND with content_type="session".

### Pattern 5: Clone Implementation via SQL Copy
**What:** Full clone copies commits and blobs between tracts in the same DB using SQL INSERT...SELECT for efficiency.
**When to use:** "full_clone" inheritance mode at spawn time.
**Confidence:** MEDIUM (efficient but requires careful handling of parent_hash, refs, annotations)

```python
def _full_clone(
    source_tract_id: str,
    target_tract_id: str,
    session: Session,
) -> str:
    """Clone all commits from source tract to target tract.

    Uses SQL INSERT...SELECT for bulk copy within the same DB.
    Blobs are shared (content-addressable, same hash = same content).
    Commits get new tract_id but same commit_hash (same content+parents).
    """
    # Commits reference blobs by content_hash (shared, no copy needed)
    # Copy CommitRows with updated tract_id
    # Copy RefRows with updated tract_id
    # Copy AnnotationRows with updated tract_id
    # Return new HEAD hash
```

**Important:** Since blobs are content-addressable and shared across all tracts (keyed by content_hash, not tract_id), blob data is naturally deduplicated. Only CommitRow, RefRow, and AnnotationRow need tract_id-scoped copies.

### Anti-Patterns to Avoid
- **Sharing Session objects between threads:** SQLAlchemy Session is NOT thread-safe. Each agent thread must have its own Session instance. This is the #1 cause of concurrency bugs.
- **Long write transactions:** Keep SQLite write transactions to single statements. The existing `session.commit()` after each operation is correct. Do NOT wrap multi-commit sequences in a single transaction for multi-agent scenarios.
- **Mutable status columns on spawn pointers:** The CONTEXT.md explicitly states "derive status from commits, no mutable status column." Do not add a status field to SpawnPointerRow.
- **Separate DB per tract:** While this eliminates write contention, it makes cross-repo queries impossible with simple JOINs and requires SQLite ATTACH (limited to 10 databases). Single shared DB is the correct choice.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe session management | Custom locking around sessions | SQLAlchemy `sessionmaker` + one session per thread | SQLAlchemy handles connection pool thread safety internally |
| Summarization for collapse | New LLM call pipeline | Existing `compress_range()` with different prompt | Phase 4 pipeline already handles 3 autonomy modes, PINNED preservation, provenance tracking |
| Content type for session boundaries | New CommitOperation enum value | New Pydantic content type ("session") | Adding to ContentPayload union is additive; new enum value requires updating every consumer |
| Clone commits between tracts | Manual row-by-row copy | SQL INSERT...SELECT within same session | Single SQL statement is atomic and orders of magnitude faster |
| Cross-tract timeline | Custom sorting + merging | SQL ORDER BY created_at across all tracts | SQLite handles cross-tract queries naturally when all tracts share one DB |
| PyPI version management | Manual version string | hatch-vcs or `_version.py` pattern | Already using `_version.py`; keep it simple |

**Key insight:** Phase 5's multi-agent layer is primarily an orchestration layer on top of existing primitives. The spawn pointer table and Session wrapper are new infrastructure, but collapse, clone, and queries are compositions of existing storage/engine/compilation operations.

## Common Pitfalls

### Pitfall 1: SQLAlchemy Session Thread Safety
**What goes wrong:** Sharing a single Session object between multiple agent threads causes race conditions, stale reads, and "database is locked" errors.
**Why it happens:** Session is designed to be used by one thread at a time. It caches ORM objects in its identity map, and concurrent access corrupts this state.
**How to avoid:** Create one Session per agent thread via `session_factory()`. The underlying Engine handles connection pooling safely.
**Warning signs:** `OperationalError: database is locked` errors during multi-agent tests, stale data reads where one agent doesn't see another's commits.

### Pitfall 2: SQLite Write Contention Under Load
**What goes wrong:** Multiple agents writing simultaneously cause `SQLITE_BUSY` timeouts despite `busy_timeout=5000`.
**Why it happens:** WAL mode allows concurrent reads but serializes writes. If write transactions are long (e.g., compression involving multiple commits), other writers block.
**How to avoid:** Keep write transactions minimal (existing per-commit session.commit() pattern is correct). For batch operations (compress, merge), use savepoints and commit after each discrete operation, not at the end.
**Warning signs:** Tests passing individually but failing under concurrent load. `busy_timeout` exceeded errors in CI.
**Recommendation:** Increase `busy_timeout` to 10000ms for multi-agent scenarios. Add retry logic around `session.commit()` for `OperationalError` with `database is locked`.

### Pitfall 3: PyPI Package Name Collision
**What goes wrong:** Publishing to PyPI as "tract" will fail or shadow the existing `tract` package (sonos/tract, neural network inference library, actively maintained as of Jan 2026).
**Why it happens:** PyPI package names are globally unique. "tract" is taken by an unrelated, active project.
**How to avoid:** Choose an alternative distribution name. Options:
  1. `trace-context` (matches current pyproject.toml `name` field) -- available on PyPI
  2. `tract-ai` -- descriptive, likely available
  3. `context-trace` -- inverted, might be available
**Warning signs:** `pip install tract` installs sonos's neural network library, not ours.
**Recommendation:** Keep the import name as `tract` (no stdlib conflict on Python 3.14) but use a different PyPI distribution name. The current pyproject.toml already uses `trace-context` as the project name -- this may be the path of least resistance.

### Pitfall 4: Clone Commit Hash Uniqueness
**What goes wrong:** Cloning commits from parent to child tract with the same commit_hash creates FK/uniqueness conflicts since commit_hash is the primary key and is not scoped by tract_id.
**Why it happens:** The commit_hash is computed from content_hash + parent_hash + timestamp + operation. If a "full clone" preserves the original commit_hash, two tracts would have commits with the same PK in the same table.
**How to avoid:** Full clone must re-create commits with new commit_hashes (new timestamps produce new hashes). The parent chain within the cloned tract must be self-consistent.
**Warning signs:** `IntegrityError: UNIQUE constraint failed: commits.commit_hash` during spawn with full_clone mode.
**Recommendation:** "Full clone" replays all commits through `CommitEngine.create_commit()` which naturally produces new hashes. HEAD snapshot avoids this entirely (single compiled commit). This is slower but correct.

### Pitfall 5: Circular Spawn References
**What goes wrong:** A tract spawns a child which spawns a grandchild which somehow references back to the original tract, creating infinite loops in ancestor/descendant queries.
**Why it happens:** No validation on spawn pointer creation prevents cycles.
**How to avoid:** Before creating a spawn pointer, walk the spawn graph from the proposed parent to root and verify the proposed child is not an ancestor.
**Warning signs:** Infinite recursion in `get_ancestor_tracts()` or similar graph traversal functions.

### Pitfall 6: Session.resume() Ambiguity
**What goes wrong:** Multiple tracts in the same DB without clear "done" markers make resume() pick the wrong tract.
**Why it happens:** Without session boundary commits, there's no reliable signal for "this tract is still active."
**How to avoid:** `resume()` should use a heuristic: most recent tract by latest commit timestamp, excluding tracts that have a `session_type="end"` content commit. The SessionContent type makes this queryable.
**Warning signs:** resume() returns an already-completed tract or a subagent instead of the main agent.

## Code Examples

### Example 1: Full Multi-Agent Workflow
```python
from tract import Session, DialogueContent, InstructionContent

# Open a multi-agent session
session = Session.open("project.db")

# Create the main agent's tract
main = session.create_tract(display_name="main-agent")
main.commit(InstructionContent(text="You are a coding assistant."))
main.commit(DialogueContent(role="user", text="Build a web app"))

# Spawn a subagent for research
research = session.spawn(
    parent=main,
    purpose="Research web framework options",
    inheritance="head_snapshot",  # Gets compiled context only
)
research.commit(DialogueContent(role="assistant", text="Researching frameworks..."))
research.commit(DialogueContent(role="assistant", text="Recommend FastAPI for backend"))

# Collapse research findings back into main
session.collapse(child=research, into=main)
# Creates summary commit in main: "Subagent researched web frameworks, recommends FastAPI"

# Cross-repo query: unified timeline
timeline = session.timeline()
# Returns all commits from all tracts, sorted by created_at
```

### Example 2: Session Boundary Commit
```python
from tract.models.session import SessionContent

# End-of-session boundary commit
main.commit(
    SessionContent(
        session_type="end",
        summary="Built FastAPI backend with user auth",
        decisions=["Chose FastAPI over Django", "SQLite for dev, Postgres for prod"],
        failed_approaches=["Tried Flask but routing was too manual"],
        next_steps=["Add frontend", "Set up CI/CD"],
    ),
    message="Session complete",
)

# Resume in next session
session2 = Session.open("project.db")
resumed = session2.resume()  # Finds the latest active tract
```

### Example 3: Crash Recovery
```python
# Crash recovery is implicit -- just reopen the DB
session = Session.open("project.db")

# List all tracts in the session
tracts = session.list_tracts()
for t in tracts:
    print(f"{t.display_name}: {t.head} ({t.commit_count} commits)")

# Resume the most recent active tract
active = session.resume()
# Continues from last committed state
```

### Example 4: Cross-Repo Point-in-Time Query
```python
# "What did research-agent know when main-agent made commit X?"
main_commit = main.get_commit("abc123...")
research_context = session.compile_at(
    tract_id=research.tract_id,
    at_time=main_commit.created_at,
)
# Returns CompiledContext for research tract as of that timestamp
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| setup.py + setuptools | pyproject.toml + hatchling | PEP 621 (2021), mature 2024+ | Already using hatchling; no change needed |
| Single Session per DB | One Session per thread, shared engine | SQLAlchemy 2.0 | Must enforce per-agent sessions |
| SQLite DELETE journal | WAL journal mode | Already configured | Already set in create_trace_engine() |
| Manual version strings | hatch-vcs or _version.py | Ongoing | Already using _version.py; keep it |

**Deprecated/outdated:**
- `setup.py` / `setup.cfg`: superseded by `pyproject.toml` per PEP 621
- SQLite `DELETE` journal mode: WAL is strictly better for concurrent read/write

## Research Findings by Topic

### 1. SQLite Concurrent Write Behavior (CRITICAL)

**Confidence:** HIGH (verified via official SQLite docs and SQLAlchemy docs)

**Finding:** WAL mode supports multiple concurrent readers with a single writer. The existing `create_trace_engine()` already configures:
- `PRAGMA journal_mode=WAL` -- enables concurrent reads
- `PRAGMA busy_timeout=5000` -- 5-second wait before SQLITE_BUSY
- `PRAGMA synchronous=NORMAL` -- good performance without full sync
- `PRAGMA foreign_keys=ON` -- referential integrity

**Concurrency limits:**
- Readers: unlimited concurrent readers (snapshot isolation)
- Writers: ONE writer at a time; other writers wait up to busy_timeout
- Write throughput: ~70k-100k transactions/second for typical records
- Degradation: significant above ~100 concurrent writers

**Recommendation: Single shared DB is safe for Phase 5.**
Multi-agent in this project means 2-20 agents, not 100+. With short write transactions (existing per-commit `session.commit()` pattern), write contention will be negligible. Each agent must have its own SQLAlchemy Session. Consider increasing busy_timeout to 10000ms.

**Key risk:** The `batch()` context manager defers commits, creating longer write transactions. In multi-agent scenarios, this could increase contention. Consider documenting that `batch()` should be used sparingly in multi-agent mode.

**Sources:**
- [SQLite WAL mode documentation](https://sqlite.org/wal.html)
- [SQLite concurrent writes analysis](https://tenthousandmeters.com/blog/sqlite-concurrent-writes-and-database-is-locked-errors/)
- [SQLAlchemy SQLite dialect docs](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html)

### 2. FTS5 vs LIKE for Cross-Repo Search

**Confidence:** MEDIUM (verified via SQLAlchemy discussion + SQLite docs)

**FTS5 Pros:**
- Full-text search with ranking (BM25)
- Much faster than LIKE for large text datasets
- Built into SQLite (no external dependency)

**FTS5 Cons:**
- No native SQLAlchemy ORM support (requires raw SQL or DDL hacks)
- Requires virtual table + trigger synchronization with main tables
- Adds schema complexity (virtual tables, triggers, content sync)
- All queries must use MATCH operator (different API from normal SQL)

**LIKE Pros:**
- Works with existing SQLAlchemy queries (`column.like('%term%')`)
- No additional schema or triggers
- Zero setup cost
- Sufficient for v1 scale (hundreds to low thousands of commits)

**Recommendation: Use LIKE for v1.** The cross-repo search feature is primarily for developer debugging ("which agent mentioned X?"), not production text search. LIKE queries on blob content with a few thousand rows will execute in milliseconds. FTS5 can be added later as a performance optimization if search becomes a bottleneck.

**Implementation:** Query `BlobRow.payload_json.like(f'%{search_term}%')` joined with `CommitRow` for tract_id filtering.

**Sources:**
- [SQLAlchemy FTS5 discussion](https://github.com/sqlalchemy/sqlalchemy/discussions/9466)
- [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html)

### 3. Python Packaging

**Confidence:** HIGH (current pyproject.toml already follows best practices)

**Current state:** The project already uses hatchling as the build backend with a well-structured pyproject.toml. The key packaging issue is the **PyPI name conflict**.

**PyPI name "tract" is TAKEN:**
- Package: `tract` (neural network inference engine by Sonos)
- Version: 0.21.13 (released May 2025, patch in Jan 2026)
- License: Apache 2.0 / MIT
- Actively maintained with regular releases

**Current pyproject.toml name:** `trace-context` (this is the distribution name)
**Import name:** `tract` (this is what users `import`)

**Options for resolution:**
1. **Keep `trace-context` as distribution name** -- already set, `pip install trace-context` and `import tract` work
2. **Use `tract-ai`** -- more descriptive, `pip install tract-ai` and `import tract`
3. **Use `llm-trace`** -- domain-specific, `pip install llm-trace` and `import tract`

**Recommendation:** Keep `trace-context` as the distribution name (already configured). The import name `tract` is not conflicted because Python import names are independent of PyPI distribution names. Users do `pip install trace-context` then `import tract`. Document this clearly.

**Other packaging items:**
- `__all__` exports: already comprehensive in `__init__.py` (87 exports)
- Version management: using `_version.py` pattern (fine for now)
- Build backend: hatchling is correct choice per PEP 621
- Optional deps: `[cli]` extra already configured

**Sources:**
- [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [PyPI tract package](https://pypi.org/project/tract/)
- [PEP 423 naming conventions](https://peps.python.org/pep-0423/)

### 4. Session Commit Type Design

**Confidence:** MEDIUM (design recommendation based on codebase analysis)

**Options analyzed:**
1. **New CommitOperation enum value** (e.g., `SESSION`)
2. **New content type** (e.g., `SessionContent` with content_type="session")

**Analysis:**
- Adding a new CommitOperation requires updating: compiler.py (compile logic), compression.py (classification), diff.py (diff computation), CLI commands, and potentially cache logic. The operation enum is a fundamental axis of behavior.
- Adding a new content type requires: adding the Pydantic model to content.py, adding it to ContentPayload union, optionally adding hints to BUILTIN_TYPE_HINTS. No changes to existing logic.

**Recommendation: New content type (`SessionContent`).** A session boundary commit is just an APPEND with special content. The existing compiler, compressor, and cache all handle it correctly without modification. The content type's `session_type` field (start/end/handoff/checkpoint) makes session transitions queryable via `commit_repo.get_by_type("session", tract_id)`.

### 5. Spawn Pointer Table Schema

**Confidence:** HIGH (consistent with existing schema patterns)

**Schema v4 additions:**

```sql
CREATE TABLE spawn_pointers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_tract_id VARCHAR(64) NOT NULL,
    parent_commit_hash VARCHAR(64) REFERENCES commits(commit_hash) ON DELETE SET NULL,
    child_tract_id VARCHAR(64) NOT NULL,
    purpose TEXT NOT NULL,
    inheritance_mode VARCHAR(20) NOT NULL,  -- 'full_clone', 'head_snapshot', 'selective'
    display_name VARCHAR(255),              -- optional human-readable name
    created_at DATETIME NOT NULL
);
CREATE INDEX ix_spawn_parent_tract ON spawn_pointers(parent_tract_id);
CREATE INDEX ix_spawn_child_tract ON spawn_pointers(child_tract_id);
```

**Key design decisions:**
- `id` as integer PK (not composite) for simplicity and auto-increment
- `parent_commit_hash` has FK to commits with SET NULL on delete (consistent with existing FK pattern)
- `inheritance_mode` stored as string enum (consistent with CommitOperation pattern)
- `display_name` is optional (tract_id is always the canonical identifier)
- No `status` column (derived from commits per CONTEXT.md decision)
- No `collapse_commit_hash` column (collapse creates a regular commit in the parent with metadata linking back)

**Migration:** v3 -> v4 adds `spawn_pointers` table. Follows existing pattern in `init_db()`.

### 6. Collapse Prompt Patterns

**Confidence:** MEDIUM (based on multi-agent research and existing compression prompts)

**Existing summarize prompt** (in `prompts/summarize.py`):
```
"You are a context summarizer for an AI assistant's conversation history.
Your job is to produce a concise summary that preserves the information
most relevant to future conversation quality."
```

**Collapse prompt should differ** because the goal is "what did this subagent accomplish" rather than "compress conversation for continuity."

**Recommended collapse prompt:**
```python
DEFAULT_COLLAPSE_SYSTEM: str = (
    "You are summarizing the work of a subagent that was delegated a specific task. "
    "Your job is to produce a concise report for the parent agent.\n\n"
    "Guidelines:\n"
    "- Focus on OUTCOMES: what was accomplished, decided, or produced.\n"
    "- Include key findings, decisions made, and artifacts created.\n"
    "- Note any failures, blockers, or unresolved issues.\n"
    "- Preserve specific technical details: code snippets, configurations, "
    "exact values, error messages.\n"
    "- Omit the subagent's internal reasoning process unless it contains "
    "important caveats or trade-off analysis.\n"
    "- If a target token count is specified, aim for approximately that length.\n"
    'Begin with: "Subagent completed: [task summary]"'
)

def build_collapse_prompt(
    messages_text: str,
    purpose: str,
    *,
    target_tokens: int | None = None,
    instructions: str | None = None,
) -> str:
    prompt = (
        f"The subagent was delegated the following task:\n"
        f"  Purpose: {purpose}\n\n"
        f"Summarize the subagent's work:\n\n{messages_text}"
    )
    if target_tokens is not None:
        prompt += f"\nTarget approximately {target_tokens} tokens."
    if instructions is not None:
        prompt += f"\nAdditional instructions: {instructions}"
    return prompt
```

**Key differences from compression prompt:**
1. Outcome-focused (not continuity-focused)
2. Includes the spawn purpose for context
3. Explicitly requests failure/blocker reporting
4. Begins with "Subagent completed:" rather than "Previously in this conversation:"

**Sources:**
- [Sub-Agent Spawning patterns](https://agentic-patterns.com/patterns/sub-agent-spawning/)
- [Google ADK multi-agent patterns](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/)
- [AWS scatter-gather patterns](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/parallelization-and-scatter-gather-patterns.html)

### 7. Full Clone vs HEAD Snapshot Implementation

**Confidence:** MEDIUM (design analysis based on codebase)

**Full Clone:**
- Must re-create all commits with new timestamps (to get unique commit_hashes, since commit_hash is the global PK)
- Blobs are naturally shared (content-addressable, same content_hash)
- Must re-create RefRow entries with new tract_id
- Must re-create AnnotationRow entries with new tract_id
- Performance: O(n) where n = number of commits in parent tract
- Risk: Slow for large tracts with many commits

**HEAD Snapshot:**
- Compile parent's context at current HEAD
- Create a single APPEND commit in child tract with the compiled text
- Performance: O(1) regardless of parent tract size
- Trade-off: Loses individual commit granularity (child sees one big message, not separate commits)

**Selective:**
- Caller specifies which commits or content types to include
- Implementation: filter parent's commit chain, re-create matching commits
- Most flexible but most complex

**Recommendation for implementation order:**
1. HEAD snapshot first (simplest, most common use case)
2. Full clone second (needed for "subagent needs full history" scenarios)
3. Selective third (power user feature)

### 8. Session.resume() Detection Heuristics

**Confidence:** MEDIUM (design recommendation)

**Heuristic for finding the "current" tract to resume:**
1. List all tracts in the DB (distinct `tract_id` from commits table)
2. For each tract, find the latest commit timestamp
3. Exclude tracts that have a `content_type="session"` commit with `session_type="end"`
4. Among remaining, pick the one with the most recent commit
5. If ties (unlikely but possible), prefer tracts with no parent in spawn_pointers (i.e., root tracts)

**Edge cases:**
- Empty DB: return None (no tracts to resume)
- All tracts ended: return None (suggest creating new)
- Multiple active tracts: return the root tract with the latest activity

## Open Questions

Things that couldn't be fully resolved:

1. **Exactly how to handle the `batch()` context manager in multi-agent mode**
   - What we know: batch() defers session.commit(), creating longer write transactions
   - What's unclear: Should batch() be disabled/warned in multi-agent mode, or should it use savepoints instead?
   - Recommendation: Document as a caveat for v1. Consider adding a `multi_agent=True` flag to batch() that uses shorter transactions.

2. **Thread safety of Tract.open() vs Session.create_tract()**
   - What we know: Tract.open() creates its own engine + session. Session.create_tract() uses the shared engine.
   - What's unclear: If a user creates Tract.open() instances pointing to the same file, they'll each have their own engine but share the same SQLite file. This is valid but less controlled than Session.
   - Recommendation: Document that Session.open() is the recommended multi-agent entry point. Tract.open() remains for single-agent backward compatibility.

3. **Autonomy spectrum implementation details for spawn/collapse**
   - What we know: 3 modes (manual/collaborative/autonomous) should apply to all operations
   - What's unclear: Exact triggering heuristics for autonomous spawn (detecting "delegatable work")
   - Recommendation: For v1, implement manual and collaborative. Autonomous spawn/collapse detection is an advanced feature that can be deferred.

4. **PyPI distribution name final decision**
   - What we know: "tract" is taken. Current config uses "trace-context".
   - What's unclear: Whether the user wants to keep "trace-context" or choose another name.
   - Recommendation: Flag for user decision during planning. Default to "trace-context" if no preference stated.

## Sources

### Primary (HIGH confidence)
- [SQLite WAL documentation](https://sqlite.org/wal.html) - concurrency model, reader/writer semantics
- [SQLAlchemy SQLite dialect](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html) - thread safety, pool configuration
- [Python Packaging Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) - pyproject.toml standards
- [PyPI tract package](https://pypi.org/project/tract/) - confirmed name collision
- Codebase analysis: `storage/engine.py`, `storage/schema.py`, `storage/sqlite.py`, `tract.py`, `operations/compression.py`

### Secondary (MEDIUM confidence)
- [SQLite concurrent writes deep dive](https://tenthousandmeters.com/blog/sqlite-concurrent-writes-and-database-is-locked-errors/) - busy_timeout patterns, benchmarks
- [SQLAlchemy FTS5 discussion](https://github.com/sqlalchemy/sqlalchemy/discussions/9466) - FTS5 ORM integration limitations
- [Sub-Agent Spawning patterns](https://agentic-patterns.com/patterns/sub-agent-spawning/) - delegation patterns
- [Google ADK multi-agent docs](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/) - supervisor/delegate patterns
- [Python Packaging Best Practices 2026](https://dasroot.net/posts/2026/01/python-packaging-best-practices-setuptools-poetry-hatch/) - hatchling recommendation

### Tertiary (LOW confidence)
- [SkyPilot SQLite concurrency](https://blog.skypilot.co/abusing-sqlite-to-handle-concurrency/) - app-level locking patterns (may be relevant if contention is higher than expected)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies needed; all existing libraries sufficient
- Architecture: MEDIUM - spawn pointer table and Session wrapper are new designs validated against existing patterns
- SQLite concurrency: HIGH - verified against official docs; existing engine config is already correct
- Pitfalls: HIGH - catalogued from verified sources and codebase analysis
- Packaging: HIGH for mechanics, MEDIUM for PyPI name decision (requires user input)
- Collapse prompts: MEDIUM - design based on multi-agent research, not empirically validated

**Research date:** 2026-02-16
**Valid until:** 2026-03-16 (30 days - stable domain, no fast-moving dependencies)
