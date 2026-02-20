# Phase 13: Unified Operation Events & Compile Records - Research

**Researched:** 2026-02-20
**Domain:** Internal codebase refactoring -- SQLAlchemy schema redesign, operation event unification, compile record persistence
**Confidence:** HIGH (source is the codebase itself)

## Summary

This phase replaces the compression-specific 3-table model (CompressionRow + CompressionSourceRow + CompressionResultRow) with a unified 2-table model (OperationEventRow + OperationCommitRow) that can record ANY structural transformation (compress, reorganize, import). It adds compile record persistence (CompileRecordRow + CompileEffectiveRow) so the exact context sent to an LLM is always recoverable. Rebase is rewritten as a "reorganize" event with source/result commit mappings. Cherry-pick is dissolved into a convenience method that creates a normal commit plus a "reorganize" event. GC is updated to respect OperationCommitRow FKs. This is a clean break -- zero backward compatibility artifacts.

The scope is entirely internal to the existing codebase. No new external libraries are needed. The standard stack is already in place (SQLAlchemy 2.0, Pydantic, frozen dataclasses). The key challenge is the breadth of changes: 3 schema tables removed, 4 new tables added, repository interfaces rewritten, 7+ source files modified, 3+ test files completely rewritten, and all references to the old tables eliminated.

**Primary recommendation:** Execute in 3 plans: (1) new schema + repository layer, (2) operation rewrites (compression, rebase, cherry-pick dissolution, GC), (3) compile records + chat/generate wiring + old table removal + test migration.

## Standard Stack

No new libraries needed. This phase uses the existing project stack:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.0+ | ORM for all tables | Already used throughout; mapped_column, DeclarativeBase |
| Pydantic | 2.x | Domain models (CommitInfo, merge models) | Already used for all models |
| dataclasses | stdlib | Frozen result types (CompressResult, GCResult) | Already used for frozen output types |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 8.x | Tests | All test files |
| tiktoken | latest | Token counting for compile records | Already used via TiktokenCounter |

**Installation:** No new packages needed.

## Architecture Patterns

### Current Schema (v5) -- Tables Being Replaced

```
compressions            (PK: compression_id)
compression_sources     (PK: compression_id + commit_hash)
compression_results     (PK: compression_id + commit_hash)
```

### New Schema (v6) -- Replacement Tables

```
operation_events        (PK: event_id)
  - event_id: str(64)
  - tract_id: str(64), indexed
  - event_type: str(30)         # "compress", "reorganize", "import"
  - branch_name: str(255), nullable
  - created_at: datetime
  - original_tokens: int, indexed  # SC-6: indexed for benchmarking
  - compressed_tokens: int, indexed  # SC-6: indexed for benchmarking
  - params_json: JSON, nullable    # target_tokens, instructions, etc.

operation_commits       (PK: event_id + commit_hash + role)
  - event_id: str(64), FK -> operation_events
  - commit_hash: str(64), FK -> commits
  - role: str(10)               # "source" or "result"
  - position: int

compile_records         (PK: record_id)
  - record_id: str(64)
  - tract_id: str(64), indexed
  - head_hash: str(64)          # HEAD at compile time
  - token_count: int
  - commit_count: int
  - token_source: str
  - params_json: JSON, nullable  # at_time, at_commit, include_edit_annotations, order
  - created_at: datetime

compile_effectives      (PK: record_id + position)
  - record_id: str(64), FK -> compile_records
  - commit_hash: str(64), FK -> commits
  - position: int               # Order in compiled output
```

### Pattern 1: Unified Event Type Enum

**What:** Use a string column for event_type rather than a Python enum, since event types may be extended in the future (plugins, user-defined operations).
**When to use:** Always -- event_type is stored as a string like "compress", "reorganize", "import".
**Example:**
```python
# In schema.py
class OperationEventRow(Base):
    __tablename__ = "operation_events"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # ...indexed token columns for benchmarking (SC-6)
    original_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    compressed_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_op_events_tract_type", "tract_id", "event_type"),
        Index("ix_op_events_original_tokens", "original_tokens"),
        Index("ix_op_events_compressed_tokens", "compressed_tokens"),
    )
```

### Pattern 2: Role-Based Commit Association

**What:** Instead of separate source/result tables, use a single association table with a `role` column.
**When to use:** For all operation-commit linkages.
**Example:**
```python
class OperationCommitRow(Base):
    __tablename__ = "operation_commits"
    event_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("operation_events.event_id"), primary_key=True
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(10), primary_key=True)  # "source" or "result"
    position: Mapped[int] = mapped_column(Integer, nullable=False)
```

### Pattern 3: Rebase as Reorganize Event

**What:** Rebase currently produces RebaseResult with replayed_commits and original_commits but records NO provenance. Phase 13 makes rebase create an OperationEvent of type "reorganize" linking old commit hashes (sources) to new commit hashes (results).
**When to use:** Every rebase operation.
**Example:**
```python
# After replay loop in rebase()
event_id = uuid.uuid4().hex
event_repo.save_event(
    event_id=event_id, tract_id=tract_id,
    event_type="reorganize", branch_name=current_branch,
    original_tokens=0, compressed_tokens=0,
    params_json={"target_branch": target_branch},
)
for pos, original in enumerate(commits_to_replay):
    event_repo.add_commit(event_id, original.commit_hash, "source", pos)
for pos, replayed in enumerate(replayed_infos):
    event_repo.add_commit(event_id, replayed.commit_hash, "result", pos)
```

### Pattern 4: Cherry-Pick Dissolved to Import

**What:** Cherry-pick currently creates a new commit via replay_commit. Phase 13 dissolves this into an `import_commit` convenience method that: (1) creates a normal commit, (2) creates an OperationEvent of type "import" linking original to new.
**When to use:** Replaces all cherry_pick() calls.
**Key insight:** The cherry-pick operation is semantically "import a commit from elsewhere." The new name is more accurate.

### Pattern 5: Compile Record Persistence

**What:** After compile() produces a CompiledContext, persist a record of what was compiled. chat()/generate() call compile() internally, so they automatically create compile records.
**When to use:** In chat() and generate(), after compile() is called.
**Example:**
```python
# In generate() after step 1 (compile)
compiled = self.compile()
# Persist compile record
record_id = uuid.uuid4().hex
compile_record_repo.save_record(
    record_id=record_id, tract_id=self._tract_id,
    head_hash=current_head, token_count=compiled.token_count,
    commit_count=compiled.commit_count, token_source=compiled.token_source,
)
for pos, commit_hash in enumerate(compiled.commit_hashes):
    compile_record_repo.add_effective(record_id, commit_hash, pos)
```

### Recommended Project Structure Changes

```
src/tract/
  storage/
    schema.py           # +OperationEventRow, +OperationCommitRow, +CompileRecordRow, +CompileEffectiveRow
                        # -CompressionRow, -CompressionSourceRow, -CompressionResultRow
    repositories.py     # +OperationEventRepository, +CompileRecordRepository
                        # -CompressionRepository (replaced)
    sqlite.py           # +SqliteOperationEventRepository, +SqliteCompileRecordRepository
                        # -SqliteCompressionRepository (replaced)
    engine.py           # Schema version 5->6, migration logic
  operations/
    compression.py      # Updated: uses OperationEventRepository instead of CompressionRepository
    rebase.py           # Updated: creates "reorganize" events, cherry_pick -> import_commit
  models/
    compression.py      # GCResult: archives_removed -> events_checked or similar
  tract.py              # Updated: _compression_repo -> _event_repo, compile record wiring
  session.py            # Updated: uses OperationEventRepository
  __init__.py           # Updated exports: CherryPickResult kept (backward compat) or removed
```

### Anti-Patterns to Avoid
- **Keeping backward compat shims:** SC-8 explicitly states "no migration shims, no compat layers." Remove ALL old references.
- **Half-migrating tests:** Don't leave any test importing CompressionRow or using compression_repo directly with old APIs.
- **Adding compile records to every compile() call:** Only chat()/generate() create compile records per SC-3. Manual compile() calls do NOT.

## Don't Hand-Roll

This is an internal refactoring. No "don't hand-roll" items -- everything is custom codebase work.

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| UUID generation | Custom ID scheme | `uuid.uuid4().hex` | Already used everywhere in codebase |
| JSON params storage | Custom serialization | SQLAlchemy JSON column | Already used for metadata_json, generation_config_json |
| Schema migration | Manual ALTER TABLE | `Base.metadata.create_all(checkfirst=True)` + version bump | Existing pattern in init_db() |

## Common Pitfalls

### Pitfall 1: FK Cascade on Old Table Removal
**What goes wrong:** Dropping CompressionRow/Source/Result tables while existing databases have data in them causes FK violations or data loss.
**Why it happens:** The clean break mandate (SC-8) means removing tables, but schema migration must handle existing data.
**How to avoid:** The v5->v6 migration should: (1) Create new tables, (2) Migrate data from old tables to new, (3) Drop old tables. Or since SC-8 says "clean break," the migration can just create new tables and drop old ones (losing old compression provenance, which is acceptable for v6).
**Warning signs:** Tests that create v5 databases and then run init_db() should verify the migration path.

### Pitfall 2: GC Not Checking OperationCommitRow
**What goes wrong:** GC deletes commits that are referenced as "source" in OperationCommitRow, breaking provenance chains.
**Why it happens:** Current GC checks `compression_repo.is_source_of()`. Phase 13 must replace this with `event_repo.is_source_of()`.
**How to avoid:** Update GC to query OperationCommitRow where role="source" for the commit hash. SC-7 explicitly requires this.
**Warning signs:** GC tests that verify archive preservation must be updated.

### Pitfall 3: Cherry-Pick Public API Break
**What goes wrong:** `t.cherry_pick()` is a public API method exported in `__init__.py`. Dissolving it without a replacement breaks user code.
**Why it happens:** SC-5 says "dissolved into a convenience method (import_commit or similar)."
**How to avoid:** The Tract facade should expose `t.import_commit()` (or keep `t.cherry_pick()` as an alias that delegates to import logic + event creation). The CherryPickResult/CherryPickError types in `__init__.py` need attention.
**Warning signs:** `from tract import CherryPickResult, CherryPickError` must still work or be intentionally removed.

### Pitfall 4: Compile Record Created Too Early
**What goes wrong:** Creating compile records inside compile() means every manual compile() call creates a record, not just chat()/generate().
**Why it happens:** SC-3 says "chat()/generate() automatically create compile records" -- this means compile records are created at the chat/generate level, not inside compile() itself.
**How to avoid:** Add compile record creation logic in generate() (which chat() delegates to), AFTER calling self.compile().
**Warning signs:** If compile() directly creates records, manual compile() calls would pollute the compile_records table.

### Pitfall 5: Missing Compression Test Migration
**What goes wrong:** 622 lines in test_compression_storage.py directly reference CompressionRow, CompressionSourceRow, CompressionResultRow. These must be completely rewritten.
**Why it happens:** SC-2 says "zero references in source or tests."
**How to avoid:** Rewrite test_compression_storage.py to test OperationEventRow/OperationCommitRow. Or rename the file entirely.
**Warning signs:** grep for "CompressionRow" should return 0 results across entire repo.

### Pitfall 6: Session.create_tract() Uses SqliteCompressionRepository
**What goes wrong:** session.py line 154 creates `SqliteCompressionRepository(session)` and passes it to Tract. This must be updated.
**Why it happens:** The compression_repo parameter is threaded through Tract constructor, session.py, and spawn.py.
**How to avoid:** Replace all `compression_repo=` with `event_repo=` (or similar). Update Tract.__init__, Tract.open(), Session.create_tract(), and spawn_tract().
**Warning signs:** Any file that imports SqliteCompressionRepository must be checked.

## Code Examples

### Example 1: OperationEventRow Schema Definition
```python
# In storage/schema.py
class OperationEventRow(Base):
    """Unified record for any structural transformation operation.

    Replaces CompressionRow. Supports compress, reorganize, and import events.
    """
    __tablename__ = "operation_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    branch_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    original_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    compressed_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    params_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_op_events_tract_type", "tract_id", "event_type"),
        Index("ix_op_events_original_tokens", "original_tokens"),
        Index("ix_op_events_compressed_tokens", "compressed_tokens"),
    )
```

### Example 2: OperationCommitRow Schema Definition
```python
class OperationCommitRow(Base):
    """Association between an operation event and its source/result commits."""
    __tablename__ = "operation_commits"

    event_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("operation_events.event_id"), primary_key=True
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(10), primary_key=True)  # "source" or "result"
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_op_commits_event", "event_id"),
        Index("ix_op_commits_commit_role", "commit_hash", "role"),
    )
```

### Example 3: CompileRecordRow Schema Definition
```python
class CompileRecordRow(Base):
    """Record of a context compilation -- persists what was sent to the LLM."""
    __tablename__ = "compile_records"

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tract_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    head_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    commit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_source: Mapped[str] = mapped_column(String(50), nullable=False)
    params_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_compile_records_tract_time", "tract_id", "created_at"),
    )


class CompileEffectiveRow(Base):
    """Association between a compile record and the commits that were effective."""
    __tablename__ = "compile_effectives"

    record_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("compile_records.record_id"), primary_key=True
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
```

### Example 4: OperationEventRepository Interface
```python
class OperationEventRepository(ABC):
    """Abstract interface for unified operation event storage."""

    @abstractmethod
    def save_event(
        self, event_id: str, tract_id: str, event_type: str,
        branch_name: str | None, created_at: datetime,
        original_tokens: int, compressed_tokens: int,
        params_json: dict | None,
    ) -> None: ...

    @abstractmethod
    def add_commit(
        self, event_id: str, commit_hash: str, role: str, position: int
    ) -> None: ...

    @abstractmethod
    def get_event(self, event_id: str) -> OperationEventRow | None: ...

    @abstractmethod
    def get_commits(
        self, event_id: str, role: str | None = None
    ) -> list[OperationCommitRow]: ...

    @abstractmethod
    def is_source_of(self, commit_hash: str) -> bool: ...

    @abstractmethod
    def get_all_ids(self, tract_id: str) -> list[str]: ...

    @abstractmethod
    def delete_commit(self, commit_hash: str) -> None: ...

    @abstractmethod
    def delete_event(self, event_id: str) -> None: ...
```

### Example 5: GC Updated for OperationCommitRow
```python
# In operations/compression.py gc() function
for commit in unreachable:
    is_archive = event_repo.is_source_of(commit.commit_hash)  # Changed from compression_repo
    # ... rest of retention logic unchanged

# Cleanup orphaned events (no sources AND no results left)
all_event_ids = event_repo.get_all_ids(tract_id)
for eid in all_event_ids:
    if not event_repo.get_commits(eid, "source") and not event_repo.get_commits(eid, "result"):
        event_repo.delete_event(eid)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CompressionRow + Sources + Results (3 tables) | OperationEventRow + OperationCommitRow (2 tables) | Phase 13 | Single schema for compress, reorganize, import |
| Rebase creates NO provenance | Rebase creates "reorganize" event with commit mappings | Phase 13 | Old commits permanently linked to new |
| Cherry-pick as separate operation | Dissolved into import_commit + "import" event | Phase 13 | Simpler, unified |
| No compile record persistence | CompileRecordRow + CompileEffectiveRow | Phase 13 | Exact LLM context always recoverable |
| Schema v5 | Schema v6 | Phase 13 | Migration from v5 required |

## Full Surface Area of Changes

### Files to Modify

| File | Changes | Scope |
|------|---------|-------|
| `src/tract/storage/schema.py` | Remove 3 old tables, add 4 new tables | HIGH |
| `src/tract/storage/repositories.py` | Remove CompressionRepository, add OperationEventRepository + CompileRecordRepository | HIGH |
| `src/tract/storage/sqlite.py` | Remove SqliteCompressionRepository, add SqliteOperationEventRepository + SqliteCompileRecordRepository | HIGH |
| `src/tract/storage/engine.py` | Add v5->v6 migration in init_db() | MEDIUM |
| `src/tract/operations/compression.py` | Replace compression_repo usage with event_repo, update GC | HIGH |
| `src/tract/operations/rebase.py` | Add event creation in rebase(), dissolve cherry_pick into import_commit | HIGH |
| `src/tract/tract.py` | Replace _compression_repo with _event_repo, add _compile_record_repo, wire compile records in generate(), update cherry_pick->import_commit | HIGH |
| `src/tract/session.py` | Replace SqliteCompressionRepository with SqliteOperationEventRepository | LOW |
| `src/tract/operations/spawn.py` | Replace compression_repo references | LOW |
| `src/tract/models/compression.py` | Update GCResult if needed (archives_removed field semantics) | LOW |
| `src/tract/models/merge.py` | CherryPickResult/CherryPickIssue -- keep or remove depending on dissolution approach | MEDIUM |
| `src/tract/__init__.py` | Update imports/exports (CherryPickResult etc.) | MEDIUM |
| `src/tract/exceptions.py` | CherryPickError -- keep as alias or remove | LOW |
| `src/tract/toolkit/definitions.py` | compress handler unchanged (uses Tract.compress which is updated internally) | NONE |
| `src/tract/policy/evaluator.py` | compress dispatch unchanged (uses Tract.compress) | NONE |
| `src/tract/policy/builtin/compress.py` | Unchanged (uses Tract.compress) | NONE |

### Test Files to Modify

| File | Lines | Changes |
|------|-------|---------|
| `tests/test_compression_storage.py` | 622 | REWRITE: Replace all CompressionRow/Source/Result references with OperationEventRow/OperationCommitRow |
| `tests/test_compression.py` | 682 | UPDATE: Replace compression_repo references, update compress internals tests |
| `tests/test_gc.py` | 348 | UPDATE: Replace compression archive terminology with event source terminology |
| `tests/test_rebase.py` | 651 | UPDATE: Add tests for reorganize events, dissolve cherry-pick tests into import tests |
| NEW: `tests/test_compile_records.py` | ~200 | NEW: Test compile record persistence, chat/generate auto-creation |

### Grep Results: All CompressionRow/Source/Result References

**Source files (7 files):**
1. `src/tract/storage/schema.py` -- Table definitions (REMOVE)
2. `src/tract/storage/repositories.py` -- ABC interface (REPLACE)
3. `src/tract/storage/sqlite.py` -- SQLite implementation (REPLACE)
4. `src/tract/tract.py` -- Tract constructor + compress/gc methods (UPDATE)
5. `src/tract/operations/compression.py` -- compress_range + gc (UPDATE)
6. `src/tract/session.py` -- Session.create_tract (UPDATE)
7. `src/tract/operations/spawn.py` -- spawn_tract (UPDATE)

**Test files (3 files):**
1. `tests/test_compression_storage.py` -- Direct schema tests (REWRITE)
2. `tests/test_compression.py` -- Integration tests (UPDATE)
3. `tests/test_gc.py` -- GC tests (UPDATE)

## Open Questions

1. **Cherry-pick public API preservation**
   - What we know: SC-5 says "dissolved into a convenience method (import_commit or similar)"
   - What's unclear: Should `t.cherry_pick()` remain as a backward-compat alias, or be fully removed per SC-8 ("no renamed-but-unused code")?
   - Recommendation: Rename to `t.import_commit()` and remove `t.cherry_pick()` entirely. Remove CherryPickResult/CherryPickError/CherryPickIssue from __init__.py exports. This is consistent with the "clean break" philosophy. The planner should decide but leaning toward full removal.

2. **Data migration for existing v5 databases**
   - What we know: init_db() currently handles v1->v2->v3->v4->v5 migrations. Phase 13 adds v5->v6.
   - What's unclear: Should old compression data be migrated to the new event tables, or just dropped?
   - Recommendation: Migrate old compression records to new event tables (event_type="compress"). This preserves provenance. The migration is straightforward since the schemas are similar. But SC-8 says "no migration shims" -- this refers to runtime compat layers, not schema migration scripts. The v5->v6 migration in init_db() is a one-time operation, not a shim.

3. **Compile record for manual compile() calls**
   - What we know: SC-3 says "chat()/generate() automatically create compile records." Manual compile() is NOT mentioned.
   - What's unclear: Should there be an opt-in parameter on compile() like `record=True`?
   - Recommendation: Only create compile records in chat()/generate(). Provide a separate `t.record_compile()` or `t.compile(record=True)` for users who want manual recording. But start with just chat()/generate() per SC-3.

4. **OperationEvent for merge operations**
   - What we know: SC-1 says the tables "can record compress, reorganize, and import operations." Merge is not mentioned.
   - What's unclear: Should merge also create an OperationEvent?
   - Recommendation: Not in this phase. The schema supports it (event_type="merge" would work), but it's not in the requirements. Future phases can add it.

5. **Rebase event_repo parameter threading**
   - What we know: Current rebase() takes no compression_repo. It will need event_repo to create reorganize events.
   - What's unclear: How to thread event_repo through the existing rebase function signature.
   - Recommendation: Add event_repo as a parameter to rebase() in operations/rebase.py, same pattern as compression_repo in compress_range(). The Tract facade passes it.

## Sources

### Primary (HIGH confidence)
- **Codebase direct inspection** -- All findings are from reading the actual source code
  - `src/tract/storage/schema.py` -- Current table definitions (CompressionRow lines 160-219)
  - `src/tract/storage/repositories.py` -- CompressionRepository ABC (lines 279-357)
  - `src/tract/storage/sqlite.py` -- SqliteCompressionRepository (lines 589-727)
  - `src/tract/operations/compression.py` -- compress_range + gc (full file, 986 lines)
  - `src/tract/operations/rebase.py` -- rebase + cherry_pick (full file, 413 lines)
  - `src/tract/tract.py` -- Tract facade compress/gc/cherry_pick/rebase/compile/generate/chat
  - `src/tract/protocols.py` -- CompiledContext, CompileSnapshot (full file)
  - `src/tract/session.py` -- Session.create_tract compression_repo wiring
  - `src/tract/operations/spawn.py` -- spawn_tract compression_repo wiring
  - `src/tract/__init__.py` -- Public API exports

### Secondary (HIGH confidence)
- `.planning/ROADMAP.md` -- Phase 13 requirements and success criteria

## Metadata

**Confidence breakdown:**
- Schema design: HIGH -- Derived from analyzing current schema and success criteria
- Architecture patterns: HIGH -- Following established codebase conventions
- Surface area mapping: HIGH -- Comprehensive grep across all files
- Pitfalls: HIGH -- Derived from actual code analysis (not speculation)
- Open questions: MEDIUM -- Some design decisions need planner judgment

**Research date:** 2026-02-20
**Valid until:** 2026-03-20 (stable -- internal refactoring, no external dependencies)
