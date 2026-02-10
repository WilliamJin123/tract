# Architecture Patterns

**Domain:** Git-like version control for LLM context windows (Python library)
**Researched:** 2026-02-10

---

## Part 1: Git Internals -- What Translates to Context Management

### Git's Object Model (Verified -- HIGH confidence)

Source: [Git Internals - Git Objects (official)](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)

Git has exactly four object types, all stored in a content-addressable object store:

| Object | What It Stores | How It's Addressed |
|--------|---------------|-------------------|
| **Blob** | Raw file content (no filename, no metadata) | SHA-1 of `"blob {bytesize}\0{content}"` |
| **Tree** | Directory listing: entries of `(mode, type, hash, filename)` pointing to blobs or subtrees | SHA-1 of the tree structure |
| **Commit** | Pointer to a tree + parent pointer(s) + author/committer + message | SHA-1 of commit content |
| **Tag** | Named pointer to a commit with optional annotation | SHA-1 of tag content |

The storage scheme is:
1. Compute `header + content`
2. SHA-1 hash the result to get the key
3. zlib-compress the content
4. Store at `.git/objects/{hash[0:2]}/{hash[2:]}`

Key architectural properties:
- **Immutable objects**: Once written, never changed. A change in content changes the hash.
- **Deduplication**: Identical content produces identical hash -- stored only once.
- **DAG structure**: Commits point to parent commits, forming a directed acyclic graph.
- **Refs are mutable pointers**: Branch names, HEAD, tags are just files containing a commit hash. Moving a branch is just rewriting a 40-byte file.

### What Maps to Trace (and What Does Not)

| Git Concept | Trace Analog | Mapping Quality | Notes |
|-------------|-------------|-----------------|-------|
| **Blob** | Context block content (the actual text/tokens) | Direct | Blobs in git store raw content. In Trace, this is the context text of a commit. |
| **Tree** | Not needed | No analog | Git trees represent directory hierarchy. Trace has no directory structure -- context is a linear sequence of commits. |
| **Commit** | Context commit (content + metadata + parent pointer) | Direct | The core primitive. But Trace commits are richer: they carry `type` (append/edit/pin), `token_count`, and semantic meaning. |
| **Refs (branches, HEAD)** | Branch and HEAD references | Direct | Mutable pointers into the commit DAG. Identical purpose. |
| **Packfiles** | Not needed for v1 | Defer | Git's delta compression (packfiles) optimizes storage by storing deltas between similar objects. Trace's "compression" is semantic (LLM summarization), not binary delta. Packfile-style optimization is premature for v1. |
| **Index (staging area)** | Not needed | No analog | Git's staging area is for composing a commit from partial file changes. Trace commits are atomic -- you commit a whole context block. |
| **Working tree** | Materialized context window | Conceptual | The "checked out" state in git = the materialized context in Trace (what the LLM actually sees). |
| **Merge commit** | Merge commit with multiple parents | Direct | A commit with 2+ parent pointers. Same DAG structure. |

### Critical Difference: Content Is Order-Sensitive

In git, the tree object captures directory structure, and file content (blobs) has no inherent ordering relationship to other blobs. In Trace, **the order of commits IS the content**. The materialized context window is the concatenation (or transformation) of commits in sequence. This means:

- **Reordering commits changes semantics** (unlike git where file A and file B are independent).
- **Merge is semantic, not line-based** -- you cannot do 3-way text merge on context blocks. Merging requires understanding meaning.
- **The "tree" equivalent is implicit** -- the ordered sequence of commits reachable from HEAD defines the "tree" of the context.

**Recommendation:** Do NOT implement a tree object. The commit chain itself (ordered parent traversal) defines the context structure. A commit's position in the chain is its "path."

---

## Part 2: Recommended Architecture

### Overall System Structure

```
+------------------------------------------------------------------+
|                         PUBLIC API LAYER                          |
|  (Python SDK: Repo, Commit, Branch, Context objects)             |
|  - Context managers for session lifecycle                        |
|  - Builder patterns for commit construction                      |
|  - Sync-first API (async wrapper optional in v2)                 |
+------------------------------------------------------------------+
        |                    |                      |
        v                    v                      v
+----------------+  +------------------+  +-------------------+
|   CORE ENGINE  |  |  MATERIALIZATION |  |  TOKEN ACCOUNTING |
|  (DAG ops,     |  |  (Read path:     |  |  (tiktoken,       |
|   branching,   |  |   how commits    |  |   API response    |
|   merging,     |  |   become prompt) |  |   extraction)     |
|   reset, log)  |  |                  |  |                   |
+----------------+  +------------------+  +-------------------+
        |                    |                      |
        v                    v                      v
+------------------------------------------------------------------+
|                       STORAGE LAYER                               |
|  (SQLAlchemy ORM models, repository abstraction)                 |
|  - CommitStore: CRUD for commit objects                          |
|  - RefStore: branch/HEAD management                              |
|  - ContentStore: content-addressable blob storage                |
+------------------------------------------------------------------+
        |
        v
+------------------------------------------------------------------+
|                        SQLite DATABASE                            |
|  (Single file, via SQLAlchemy engine)                            |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                    LLM OPERATIONS LAYER                           |
|  (Compression, semantic merge, reordering safety checks)         |
|  - Isolated from core engine                                     |
|  - Accepts callables (user-provided or built-in)                 |
|  - Every operation reports token cost                            |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                         CLI LAYER                                 |
|  (Thin wrapper over Public API)                                  |
|  - Click/Typer commands                                          |
|  - Human-readable output formatting                              |
|  - No business logic -- delegates everything to API              |
+------------------------------------------------------------------+
```

### Component Boundaries

| Component | Responsibility | Communicates With | Depends On |
|-----------|---------------|-------------------|------------|
| **Public API** (`trace.api`) | User-facing Python SDK. Repo, Branch, Commit classes. Context managers for lifecycle. | Core Engine, Materialization, Token Accounting | Core Engine |
| **Core Engine** (`trace.engine`) | DAG operations: commit, branch, merge, reset, checkout, log, diff. Pure logic -- no LLM calls, no storage details. | Storage Layer | Storage Layer |
| **Materialization** (`trace.materialize`) | Converts commit DAG into actual context window content. Pluggable strategies: simple concat (default), template-based, custom. | Core Engine (reads commits), Token Accounting | Core Engine |
| **Token Accounting** (`trace.tokens`) | Token counting via tiktoken. Budget tracking. Cost reporting. | None (utility) | tiktoken |
| **Storage Layer** (`trace.storage`) | SQLAlchemy models and repository pattern. Abstracts database operations. Content-addressable blob store. | SQLite (via SQLAlchemy) | SQLAlchemy |
| **LLM Operations** (`trace.llm`) | Compression, semantic merge, reorder safety checks. Isolated behind a callable interface. | Core Engine (operates on commits), Token Accounting | User-provided LLM callable OR built-in client |
| **CLI** (`trace.cli`) | Human interface. Thin wrapper. Formats output. | Public API only | Public API |
| **Multi-Agent** (`trace.agents`) | Spawn pointers, collapse, expand. Manages repo-of-repos. | Core Engine, Storage Layer | Core Engine |

### Data Flow

**Write Path (committing context):**
```
User code / Agent framework
    |
    v
Public API: repo.commit(content="...", type=CommitType.APPEND, message="...")
    |
    v
Token Accounting: count tokens in content
    |
    v
Core Engine:
    1. Hash content -> content_hash (SHA-256)
    2. Create commit object (content_hash, parent=HEAD, type, token_count, timestamp, message)
    3. Hash commit -> commit_hash
    |
    v
Storage Layer:
    1. Store content blob (if not already exists -- deduplication)
    2. Store commit record
    3. Update HEAD ref to new commit hash
    |
    v
SQLite: INSERT INTO content_blobs, INSERT INTO commits, UPDATE refs
```

**Read Path (materializing context):**
```
User code / Agent framework
    |
    v
Public API: repo.materialize() or repo.context()
    |
    v
Core Engine: walk DAG from HEAD backwards, collect ordered commits
    |
    v
Materialization:
    1. Apply materializer strategy (default: ordered concatenation)
    2. Respect commit types (Pin commits always included verbatim)
    3. Apply any active compression summaries
    |
    v
Token Accounting: report total token count of materialized context
    |
    v
Return: MaterializedContext(content=str, token_count=int, commits=[...])
```

**Branch/Merge Flow:**
```
Branch: Create new ref pointing to current HEAD commit
Switch: Update active branch pointer
Merge:
    1. Core Engine: identify common ancestor, collect commits from both branches
    2. If fast-forward possible: just move ref pointer (no merge commit)
    3. If diverged: delegate to merge strategy
        a. Default: concatenate (append branch B commits after branch A)
        b. LLM semantic merge: call LLM Operations layer to reconcile
    4. Create merge commit with two parent pointers
    5. Update HEAD
```

---

## Part 3: Storage Layer Design -- Resolution of Open Question

### Decision: Structured Rows with Content-Addressable Blob Table

**Recommendation:** Hybrid approach -- structured rows for the commit DAG and refs, with a separate content-addressable blob table for the actual context content.

**Rationale:**

The open question in PROJECT.md asks: "structured rows vs blobs." The answer is both, for different purposes:

1. **Commit metadata needs structured columns** because you query it: "find all commits on branch X," "get commits between A and B," "find commits by type," "order by timestamp." These are relational queries that need indexes.

2. **Context content should be stored as content-addressable blobs** because: (a) the same content committed twice should be stored once (deduplication), (b) content can be large (up to tens of thousands of tokens), and (c) you rarely query content by substring -- you query by commit metadata and then fetch content.

3. **SQLite handles blobs under 100KB efficiently** (per SQLite documentation), and context blocks will almost always be under this threshold. A 30K-token context block is roughly 120KB of text at worst, and individual commits will typically be 1-20KB.

### Recommended SQLAlchemy Models

```python
# trace/storage/models.py

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Enum, LargeBinary, Index
from sqlalchemy.orm import relationship, DeclarativeBase
import enum

class CommitType(enum.Enum):
    APPEND = "append"
    EDIT = "edit"
    PIN = "pin"
    MERGE = "merge"       # System-generated merge commit
    COMPRESS = "compress"  # System-generated compression commit

class Base(DeclarativeBase):
    pass

class ContentBlob(Base):
    """Content-addressable storage for context blocks.

    Keyed by SHA-256 hash of content. Deduplicates identical content.
    """
    __tablename__ = "content_blobs"

    content_hash = Column(String(64), primary_key=True)  # SHA-256 hex
    content = Column(Text, nullable=False)
    byte_size = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)


class Commit(Base):
    """A commit in the context DAG.

    Analogous to a git commit: points to content, has parent(s),
    carries metadata. The commit_hash is SHA-256 of the commit's
    canonical representation (content_hash + parent_hashes + metadata).
    """
    __tablename__ = "commits"

    commit_hash = Column(String(64), primary_key=True)  # SHA-256 hex
    content_hash = Column(String(64), ForeignKey("content_blobs.content_hash"), nullable=False)
    commit_type = Column(Enum(CommitType), nullable=False)
    message = Column(Text, nullable=True)
    token_count = Column(Integer, nullable=False)  # Denormalized from content blob
    timestamp = Column(DateTime, nullable=False)

    # Repo scoping (supports multiple repos in one DB)
    repo_id = Column(String(64), nullable=False, index=True)

    # Relationships
    content = relationship("ContentBlob")
    parents = relationship(
        "CommitParent",
        foreign_keys="CommitParent.child_hash",
        back_populates="child",
        order_by="CommitParent.parent_order"
    )

    __table_args__ = (
        Index("ix_commits_repo_timestamp", "repo_id", "timestamp"),
    )


class CommitParent(Base):
    """Association table for commit parent relationships.

    Supports multiple parents (merge commits) with explicit ordering.
    This is the edge table of the DAG.
    """
    __tablename__ = "commit_parents"

    child_hash = Column(String(64), ForeignKey("commits.commit_hash"), primary_key=True)
    parent_hash = Column(String(64), ForeignKey("commits.commit_hash"), primary_key=True)
    parent_order = Column(Integer, nullable=False, default=0)  # 0 = first parent

    child = relationship("Commit", foreign_keys=[child_hash], back_populates="parents")
    parent = relationship("Commit", foreign_keys=[parent_hash])


class Ref(Base):
    """Mutable named pointer to a commit (branch, HEAD, tag).

    Analogous to git refs. A branch is just a ref that moves forward
    with each commit. HEAD is a special ref pointing to the active branch.
    """
    __tablename__ = "refs"

    repo_id = Column(String(64), primary_key=True)
    ref_name = Column(String(255), primary_key=True)  # e.g., "heads/main", "HEAD"
    commit_hash = Column(String(64), ForeignKey("commits.commit_hash"), nullable=True)
    # For symbolic refs (HEAD -> refs/heads/main)
    symbolic_target = Column(String(255), nullable=True)

    commit = relationship("Commit")


class SpawnPointer(Base):
    """Links a parent commit to a child agent's repo.

    When a parent agent spawns a subagent, the parent commit records
    a spawn pointer to the child repo's root commit.
    """
    __tablename__ = "spawn_pointers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_repo_id = Column(String(64), nullable=False, index=True)
    parent_commit_hash = Column(String(64), ForeignKey("commits.commit_hash"), nullable=False)
    child_repo_id = Column(String(64), nullable=False, index=True)
    child_root_hash = Column(String(64), nullable=True)  # Root commit of child's trace
    agent_name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="active")  # active, collapsed, pruned

    parent_commit = relationship("Commit")
```

### Why This Design (Confidence: HIGH)

**DAG via association table (CommitParent):** This follows SQLAlchemy's recommended directed graph pattern (source: [SQLAlchemy ORM Examples - Directed Graph](https://docs.sqlalchemy.org/en/20/orm/examples.html)). Using a separate edge table rather than self-referential FK supports:
- Multiple parents per commit (merge commits)
- Ordered parents (first parent = "mainline" in merge)
- Efficient ancestor queries

**Content-addressable blob table:** Separating content from commits gives us:
- Deduplication: identical context blocks stored once
- Lazy loading: can traverse DAG metadata without loading content
- Clear separation of "what happened" (commit graph) from "what was said" (content)

**Repo scoping via repo_id:** Rather than separate databases per repo, one SQLite file can hold multiple repos (parent + child agents). This simplifies multi-agent scenarios -- all in one transaction space. The repo_id is a UUID or hash assigned at `trace init`.

**SHA-256 over SHA-1:** Git uses SHA-1 for historical reasons (and is migrating to SHA-256). For a new project, use SHA-256 from the start. The performance difference is negligible for our use case. (Confidence: HIGH -- this is a well-established recommendation.)

---

## Part 4: Edit Commit Semantics -- Resolution of Open Question

### Decision: Override Commit (not in-place replacement)

**Recommendation:** An Edit commit should be an **override commit** -- a new commit that supersedes a specific prior commit's content, NOT an in-place mutation of the original.

**Rationale:**

1. **Immutability is load-bearing.** Git's core invariant is that objects are immutable. Content-addressable storage breaks if you mutate objects because the hash changes, invalidating all references. An edit that mutates a commit would require rewriting every descendant commit's hash -- the equivalent of `git rebase` on every edit. This is the most expensive operation in git for a reason.

2. **Override preserves history.** An Edit commit that says "this supersedes commit X" gives you: audit trail (what was the old content?), rollback capability (revert the edit), and clear semantics for the materializer (when building context, use the edit's content instead of the original's).

3. **Materializer handles resolution.** When materializing context, the materializer walks the commit chain and, for each Edit commit, replaces the target commit's content with the edit's content in the materialized output. The DAG itself is never mutated.

### Edit Commit Structure

```python
class Commit:
    # ... existing fields ...

    # For Edit commits: which commit does this edit replace?
    edit_target_hash = Column(String(64), ForeignKey("commits.commit_hash"), nullable=True)
    # Only populated when commit_type == EDIT
```

**Materialization rule:** When walking commits to build context:
1. Collect all commits from root to HEAD in order.
2. Build an edit map: `{target_hash: edit_commit}` from all EDIT commits.
3. For each commit in the sequence, if it's the target of an edit, use the edit's content instead.
4. Skip the edit commits themselves (they're "applied" by replacing their targets).

This gives "surgical correction of a historical commit" (Case Study 6 from PREPLAN.md) without mutating the DAG.

---

## Part 5: API Design -- Python SDK Patterns

### Design Principles

Based on research into modern Python SDK patterns (sources: [Python SDK design patterns](https://vineeth.io/posts/sdk-development), [Python context managers](https://docs.python.org/3/library/contextlib.html), [Sync/async dual interface patterns](https://sethmlarson.dev/designing-libraries-for-async-and-sync-io)):

1. **Sync-first, async-optional.** Trace's I/O is local SQLite writes (fast) and LLM calls (slow but user-provided). The core SDK should be synchronous. Async wrappers can be added in v2 for the LLM operations layer. Trying to go async-first adds complexity with no benefit for v1.

2. **Context managers for lifecycle.** Opening/closing a Trace repo is a resource management concern (database connections, file locks). Use `with` statements.

3. **Method chaining where natural, but don't force builder pattern.** Commits are simple enough to construct in a single call. Branches and merges are single operations. Builder pattern is overkill here.

4. **Explicit over magical.** No implicit commits, no auto-branching, no hidden state. Every mutation is an explicit method call. (The policy engine in Phase 5 adds automation later -- but the core API is explicit.)

### Recommended API Surface

```python
import trace

# Lifecycle: context manager
with trace.open("./my_project", create=True) as repo:

    # --- Committing ---
    repo.commit("User asked about authentication", type="append", message="user query")
    repo.commit("OAuth2 with PKCE recommended...", type="append", message="assistant response")
    repo.commit("Use session-based auth, not JWT", type="pin", message="architecture decision")

    # Edit: override a previous commit's content
    repo.edit(target="abc123", content="Use cookie-based sessions", message="corrected auth approach")

    # --- Reading ---
    context = repo.materialize()        # -> MaterializedContext
    context.text                         # The full context string
    context.token_count                  # Total tokens
    context.commits                      # List of commits included

    # --- History ---
    repo.log()                           # -> List[CommitInfo]
    repo.log(limit=10)
    repo.diff("abc123", "def456")        # -> ContextDiff
    repo.status()                        # -> RepoStatus (HEAD, branch, token count)

    # --- Branching ---
    repo.branch("exploration")           # Create branch
    repo.switch("exploration")           # Switch to branch
    repo.merge("exploration")            # Merge into current branch
    repo.rebase("main")                 # Rebase current onto main

    # --- Reset ---
    repo.reset("abc123", mode="soft")    # Move HEAD, keep commits accessible
    repo.reset("abc123", mode="hard")    # Move HEAD, mark forward commits for GC

    # --- Compression (requires LLM) ---
    repo.compress(range=("abc123", "def456"), strategy="summarize")

    # --- Multi-agent ---
    child = repo.spawn("researcher")     # -> Repo (child agent's trace)
    # ... child does work ...
    repo.collapse("researcher")          # Collapse child trace into parent
    repo.expand("ghi789")               # Expand a collapse commit for debugging

# Context manager handles:
# - Database connection lifecycle
# - Session persistence
# - Crash recovery (write-ahead log)
```

### Key API Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| **Naming convention** | Git-inspired where concepts overlap (`commit`, `branch`, `merge`, `log`, `status`); domain-specific where they diverge (`materialize`, `compress`, `spawn`, `collapse`) | Familiarity for target audience (developers). Diverge only when git terms would be misleading. |
| **Return types** | Named dataclasses/Pydantic models, not raw dicts | Type safety, IDE autocomplete, self-documenting API. |
| **Error handling** | Custom exception hierarchy (`TraceError` base, `MergeConflictError`, `DetachedHeadError`, etc.) | Clear, catchable errors. Git's error messages are notoriously bad -- Trace should do better. |
| **Commit identification** | Full hash or unambiguous prefix (like git's short hash) | Convenience without ambiguity. Implement prefix resolution in storage layer. |
| **Immutable return objects** | `CommitInfo`, `MaterializedContext`, `RepoStatus` are frozen dataclasses | Prevents accidental mutation. Clear that these are snapshots, not live references. |

---

## Part 6: LLM Operations -- Architectural Isolation

### Why Isolation Matters

LLM operations are fundamentally different from core engine operations:
- **Non-deterministic**: Same input can produce different output.
- **Expensive**: Each call costs money and time.
- **Optional**: Core version control works without LLM calls.
- **User-configurable**: Users may want their own LLM, their own prompts, their own logic.

### Isolation Pattern: Strategy + Callable Interface

```python
# trace/llm/protocol.py
from typing import Protocol, Callable

class LLMCallable(Protocol):
    """User-provided LLM callable.
    Accepts a prompt string, returns a response string."""
    def __call__(self, prompt: str) -> str: ...

class MergeStrategy(Protocol):
    """Strategy for resolving merge conflicts."""
    def merge(self, base: str, ours: str, theirs: str, llm: LLMCallable) -> str: ...

class CompressionStrategy(Protocol):
    """Strategy for compressing commit ranges."""
    def compress(self, commits: list[CommitInfo], llm: LLMCallable) -> str: ...
```

**Built-in convenience client:** Trace ships with a simple LLM client (httpx-based, targeting OpenAI-compatible APIs) that users can use out of the box. But the architecture never depends on it -- everything goes through the callable interface.

**Configuration:**
```python
# User provides their own LLM
repo = trace.open("./project", llm=my_custom_llm_function)

# Or use built-in client
from trace.llm import openai_client
repo = trace.open("./project", llm=openai_client(model="gpt-4o", api_key="..."))

# Or no LLM (compression/merge disabled)
repo = trace.open("./project")  # llm=None by default
```

**Key boundary:** The Core Engine NEVER imports from `trace.llm`. The LLM Operations layer is a peer that operates on Core Engine objects but is not a dependency of the Core Engine. The Public API layer orchestrates their interaction.

```
Public API
    |         \
    v          v
Core Engine   LLM Operations
    |              |
    v              v
Storage       (external LLM)
```

---

## Part 7: Multi-Agent Architecture

### Design: OpenTelemetry-Inspired Trace Hierarchy

Based on research into OpenTelemetry's trace/span model (source: [OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/)) and hierarchical multi-agent patterns.

**Key insight from OpenTelemetry:** A trace is a DAG of spans, where each span has a trace_id (shared across the whole tree) and a span_id (unique to the span). Parent-child relationships are expressed via parent_span_id. This is exactly the right model for multi-agent Trace.

### Mapping to Trace

| OpenTelemetry Concept | Trace Analog |
|----------------------|-------------|
| Trace (collection of spans) | Session (entire multi-agent execution) |
| Root Span | Head agent's repo |
| Child Span | Subagent's repo |
| span_id | repo_id |
| trace_id | session_id (shared across all agent repos) |
| parent_span_id | parent_repo_id (via SpawnPointer) |
| SpanLinks | SpawnPointers (link parent commit to child repo) |

### Hierarchical Commit Graph Structure

```
HEAD AGENT (repo_id: "head-001", session_id: "session-xyz")
  |
  c1 -- c2 -- c3 -- [SPAWN: researcher] -- c4 -- [COLLAPSE: researcher] -- c5
                          |
                          v
              RESEARCHER AGENT (repo_id: "research-001", session_id: "session-xyz")
                c1' -- c2' -- c3'
                        |
                        [SPAWN: web-search]
                        |
                        v
                    WEB SEARCH AGENT (repo_id: "search-001", session_id: "session-xyz")
                      c1'' -- c2''
```

### Spawn Mechanics

When a parent agent spawns a subagent:

1. **Parent side:** A spawn pointer is created, linking the parent's current commit to the new child repo. The parent can continue committing independently.

2. **Child side:** A new Repo is initialized with its own commit chain. The child's root commit can optionally inherit content from the parent's spawn commit (selected context inheritance).

3. **All in one database:** All repos share the same SQLite database (scoped by repo_id). This enables:
   - Cross-repo queries ("what did agent X know when agent Y decided Z?")
   - Atomic multi-repo operations
   - Single file for the entire session

### Collapse Mechanics

When a subagent completes:

1. **Generate collapse summary:** Use LLM Operations to summarize the child's entire commit chain into a single block.
2. **Create collapse commit in parent:** A special commit (type=COMPRESS or a new COLLAPSE type) that contains:
   - The summary content
   - Provenance metadata: child repo_id, child's HEAD hash at collapse time
   - Token count of the summary vs. the full child trace
3. **Retain child data:** The child's commits remain in the database but can be marked for garbage collection after a retention period.
4. **Expand for debugging:** `repo.expand(collapse_commit_hash)` retrieves the full child commit chain by following the provenance pointer.

### Session Table

```python
class Session(Base):
    """Groups all repos in a multi-agent execution."""
    __tablename__ = "sessions"

    session_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, nullable=False)
    root_repo_id = Column(String(64), nullable=False)  # Head agent's repo
    metadata = Column(Text, nullable=True)  # JSON metadata
```

---

## Part 8: Patterns to Follow

### Pattern 1: Repository Pattern for Storage

**What:** Abstract all database access behind a repository interface. The Core Engine works with `CommitRepository`, `RefRepository`, `ContentRepository` -- never with SQLAlchemy sessions directly.

**Why:** Testability (mock the repository for unit tests), swappability (could replace SQLite with something else), and clean boundaries (engine doesn't know about SQL).

**Example:**
```python
# trace/storage/repositories.py
from abc import ABC, abstractmethod

class CommitRepository(ABC):
    @abstractmethod
    def get(self, commit_hash: str) -> Commit: ...

    @abstractmethod
    def save(self, commit: Commit) -> None: ...

    @abstractmethod
    def get_ancestors(self, commit_hash: str, limit: int = None) -> list[Commit]: ...

    @abstractmethod
    def get_between(self, from_hash: str, to_hash: str) -> list[Commit]: ...

class SQLiteCommitRepository(CommitRepository):
    """SQLAlchemy-backed implementation."""
    def __init__(self, session: Session):
        self._session = session
    # ... implementations ...
```

### Pattern 2: Strategy Pattern for Pluggable Operations

**What:** Materialization, compression, and merge are all pluggable strategies. Ship defaults, let users customize.

**Why:** Different agent frameworks have different needs. A LangGraph agent might materialize context differently than a Claude Code agent. The strategy pattern lets them plug in their own logic without forking the library.

### Pattern 3: Event Sourcing Mindset

**What:** The commit chain IS an event log. The materialized context IS a projection of that event log. Never store "current state" separately -- always derive it from the commit history.

**Why:** This gives you time-travel (checkout any point), audit trails (how did we get here?), and correctness (one source of truth). The materialized context is always a function of the commits, never a cached value that can go stale.

**Caveat:** The materialized context CAN be cached for performance (keyed by HEAD hash), but the cache is always derivable from the commits.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Mutable Object Store

**What:** Allowing commits or content blobs to be modified after creation.
**Why bad:** Breaks content-addressable integrity. Hash no longer matches content. DAG invariants violated. Cascading corruption.
**Instead:** All objects are immutable. Edit commits create NEW objects that supersede old ones. The materializer resolves the override at read time.

### Anti-Pattern 2: Tight Coupling to LLM Provider

**What:** Hard-coding OpenAI/Anthropic API calls into the core engine.
**Why bad:** Users have different LLM providers, some want local models, some want custom wrappers. Coupling to a provider means the library can't be used without that provider's API key.
**Instead:** Accept a callable. Ship convenience clients as optional extras. Core engine never touches LLM.

### Anti-Pattern 3: Storing Materialized Context as Source of Truth

**What:** Caching the materialized context window and treating it as authoritative rather than re-deriving from commits.
**Why bad:** The cached context can drift from the commit chain (after edits, compression, branch switching). Two sources of truth = bugs.
**Instead:** Always derive from commits. Cache for performance, invalidate on any DAG mutation.

### Anti-Pattern 4: Global Singleton Repo

**What:** A global `trace.current_repo` that all code shares.
**Why bad:** Multi-agent requires multiple repos. Global state makes testing hard. Thread safety nightmare.
**Instead:** Each repo is an explicit object. Passed by reference. No global state.

### Anti-Pattern 5: Reimplementing Git's Binary Delta Compression

**What:** Building packfile-style delta compression for v1.
**Why bad:** Premature optimization. Context blocks are text (highly compressible with zlib), not binary. SQLite already handles page-level compression. The "compression" that matters for Trace is semantic (LLM summarization), not binary.
**Instead:** Use SQLite's built-in compression. Focus engineering effort on semantic compression quality. Consider binary optimization only if storage becomes a proven bottleneck (very unlikely for v1).

---

## Scalability Considerations

| Concern | Single Agent (Phase 1) | Multi-Agent (Phase 3) | Production Scale |
|---------|----------------------|---------------------|-----------------|
| **Storage size** | One SQLite file, few hundred commits. Trivial. | One SQLite file per session, thousands of commits across repos. Still trivial for SQLite (handles millions of rows). | Consider WAL mode for concurrent reads. SQLite handles TB-scale databases. |
| **DAG traversal speed** | Linear walk from HEAD. O(n) where n = commits on branch. | Cross-repo queries may require joins. Index on repo_id + timestamp. | For extremely deep histories (10K+ commits), consider materialized ancestry table or recursive CTE optimization. |
| **Token counting** | Per-commit at write time. O(1) lookup at read time. | Same -- denormalized on commit record. | If tiktoken becomes a bottleneck, batch count or use approximate counting. |
| **LLM operation cost** | Compression/merge are rare, user-initiated. | Collapse operations may trigger compression. | Budget tracking per-session. Warn before expensive operations. |
| **Concurrent access** | Single agent -- no concurrency. | Parent and child agents may write simultaneously. SQLite WAL mode handles this. | For true parallel writes (many agents), consider connection pooling or write serialization. |

---

## Suggested Build Order

Based on component dependencies:

```
Phase 0: Foundation
    1. Storage Layer (models, SQLite setup, repository interfaces)
    2. Content-addressable blob store
    3. Core Engine: commit, read commit chain
    4. Token Accounting (tiktoken integration)
    5. Basic Materialization (simple concatenation)
    6. Public API: Repo.open(), Repo.commit(), Repo.materialize()

    WHY FIRST: Everything else depends on the storage layer and
    basic commit/read cycle. Get the data model right early.

Phase 1: Single Agent Linear History
    7. Core Engine: log, status, diff, reset (soft/hard), checkout
    8. Full Public API for linear history
    9. CLI wrapper (thin, over API)

    WHY SECOND: Proves the core model works for the simplest case.
    Validates the data model before adding branching complexity.

Phase 2: Branching, Merging, Compression
    10. Core Engine: branch, switch, merge (fast-forward)
    11. Ref management (branch pointers, symbolic HEAD)
    12. LLM Operations layer (callable interface, built-in client)
    13. Semantic merge strategy
    14. Compression operations
    15. Commit reordering

    WHY THIRD: Branching requires a stable linear history model.
    LLM operations are isolated and can be developed in parallel
    once the callable interface is defined.

Phase 3: Multi-Agent
    16. SpawnPointer model and storage
    17. Session management
    18. spawn/collapse/expand operations
    19. Cross-repo queries
    20. Garbage collection

    WHY FOURTH: Multi-agent is repo-of-repos -- requires stable
    single-repo operations first.
```

---

## Related Work (Confidence: MEDIUM)

The following projects operate in adjacent or overlapping space:

| Project | Relationship to Trace | Key Takeaway |
|---------|----------------------|--------------|
| **Git-Context-Controller (GCC)** ([arXiv 2508.00031](https://arxiv.org/abs/2508.00031)) | Most directly comparable. Same git-inspired metaphor for LLM context. Achieved SOTA on SWE-Bench. | GCC uses a file-system approach (.GCC/ directory with markdown files). Trace's SQLite-backed approach is more structured and queryable. GCC validates the core concept. |
| **DiffMem** ([GitHub](https://github.com/Growth-Kinetics/DiffMem)) | Git-backed memory for conversational AI. Markdown files + actual git repo. | Uses real git as persistence. Simpler but less flexible than Trace's custom DAG. Useful validation that version-controlled context improves agent quality. |
| **Beads** ([GitHub](https://github.com/steveyegge/beads)) | Git-backed memory + issue tracker for coding agents. SQLite + JSONL + git. | Three-layer persistence (SQLite cache + JSONL + git) is interesting but complex. Trace should avoid this complexity -- SQLite alone is sufficient. |
| **AgentFold** ([arXiv 2510.24699](https://arxiv.org/abs/2510.24699)) | Context compression for long-horizon agents. Sublinear context growth. | "Folding" operations (granular + deep consolidation) map well to Trace's compression. Key insight: multi-scale compression (not just one level). |
| **OpenTelemetry** ([opentelemetry.io](https://opentelemetry.io/docs/concepts/signals/traces/)) | Not a context manager, but the trace/span hierarchy model is directly applicable to multi-agent architecture. | trace_id/span_id/parent_span_id pattern is the right model for session/repo/parent-repo relationships. |

---

## Sources

### HIGH Confidence (Official documentation, verified)
- [Git Internals - Git Objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects) -- Official git documentation on object model
- [Git Internals - Packfiles](https://git-scm.com/book/en/v2/Git-Internals-Packfiles) -- Official git documentation on delta compression
- [SQLAlchemy Adjacency List Relationships](https://docs.sqlalchemy.org/en/21/orm/self_referential.html) -- Official SQLAlchemy documentation
- [SQLAlchemy Directed Graph Example](https://docs.sqlalchemy.org/en/20/orm/examples.html) -- Official SQLAlchemy ORM examples
- [SQLite Internal vs External BLOBs](https://sqlite.org/intern-v-extern-blob.html) -- Official SQLite documentation on blob performance
- [Python contextlib](https://docs.python.org/3/library/contextlib.html) -- Official Python documentation
- [OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/) -- Official OpenTelemetry documentation

### MEDIUM Confidence (Multiple sources agree, cross-verified)
- [Git for Computer Scientists](https://eagain.net/articles/git-for-computer-scientists/) -- Well-known technical reference on git DAG model
- [Designing Libraries for Async and Sync I/O](https://sethmlarson.dev/designing-libraries-for-async-and-sync-io) -- Respected Python community resource
- [Comprehensive Analysis of Design Patterns for REST API SDKs](https://vineeth.io/posts/sdk-development) -- SDK design patterns analysis
- [Content Addressable Storage concepts](https://en.wikipedia.org/wiki/Content-addressable_storage) -- Well-established computer science concept

### LOW Confidence (Single source, needs validation)
- [Git-Context-Controller paper](https://arxiv.org/abs/2508.00031) -- Peer-reviewed but very recent (2025). Architecture details from search summaries, not direct reading of full paper.
- [AgentFold paper](https://arxiv.org/abs/2510.24699) -- Same caveat as GCC.
- [Anthropic Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) -- Referenced in multiple sources but could not fetch directly to verify specific claims.
