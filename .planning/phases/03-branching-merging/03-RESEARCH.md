# Phase 3: Branching & Merging - Research

**Researched:** 2026-02-14
**Domain:** Git-like branching/merging for structured LLM context, LLM client infrastructure, conflict resolution
**Confidence:** HIGH

## Summary

Phase 3 adds four major capabilities to Tract: (1) branch management with pointer-based refs, (2) a built-in httpx-based LLM client for OpenAI-compatible APIs, (3) merge strategies including fast-forward, divergent merge commits with multiple parents, and LLM-mediated semantic merge, and (4) rebase and cherry-pick with semantic safety checks.

The existing codebase already has strong foundations for branching: the `RefRepository` supports `get_branch`, `set_branch`, `list_branches`, `attach_head`, and `detach_head`. The `checkout` operation already handles branch switching. What is missing is: (a) a `create_branch` SDK method, (b) schema support for merge commits with multiple parents, (c) the LLM client, (d) conflict detection and resolution, and (e) rebase/cherry-pick commit replay.

The critical schema change is that `CommitRow.parent_hash` is currently a single `String(64)` column. Merge commits require two parents. The recommended approach is to add a `commit_parents` association table rather than modifying the existing `parent_hash` column, preserving backward compatibility for linear history while enabling multi-parent DAG traversal.

**Primary recommendation:** Add a `commit_parents` association table for multi-parent support. Keep `parent_hash` as-is for the common single-parent case (backward compat + performance). Use httpx directly (no openai library dependency) with tenacity for retry logic. Implement conflict detection as structural graph analysis before any LLM involvement.

## Standard Stack

### Core (New Dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| httpx | >=0.27,<1.0 | Sync HTTP client for OpenAI-compatible APIs | Already powers the official openai library; mature, well-typed, sync+async support |
| tenacity | >=8.2,<10 | Retry with backoff for LLM API calls | De facto standard for Python retry logic; composable wait/stop/retry strategies |

### Existing (Already in pyproject.toml)

| Library | Version | Purpose | Phase 3 Use |
|---------|---------|---------|-------------|
| sqlalchemy | >=2.0.46,<2.2 | ORM/schema | New `commit_parents` table, DAG queries |
| pydantic | >=2.10,<3.0 | Models | MergeResult, ConflictInfo, BranchInfo models |
| tiktoken | >=0.12.0 | Token counting | Token counting for merge commit content |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| httpx (direct) | openai library | openai lib adds ~15MB dependency, locks to OpenAI semantics; httpx is lighter, more generic for OpenAI-compatible endpoints |
| tenacity | backoff | backoff is simpler but less composable; tenacity has better retry condition composition |
| tenacity | httpx built-in retries | httpx retries only handle ConnectError/ConnectTimeout, not HTTP 429/500/503 |

**Installation:**
```bash
pip install httpx>=0.27 tenacity>=8.2
```

These should be added as required dependencies in `pyproject.toml` (not optional), per the CONTEXT.md decision that the LLM client is a required dependency.

## Architecture Patterns

### Recommended Project Structure

```
src/tract/
  llm/                     # NEW: LLM client package (Plan 03-02)
    __init__.py             # Exports: OpenAIClient, LLMResolver, LLMClientProtocol
    client.py               # Built-in httpx OpenAI-compatible client
    protocols.py            # LLMClient protocol, ResolverCallable protocol
    resolver.py             # Built-in OpenAIResolver (uses client)
    errors.py               # LLMClientError, RateLimitError, etc.
  operations/
    __init__.py
    navigation.py           # existing
    history.py              # existing
    diff.py                 # existing
    branch.py               # NEW: create, delete, list, switch (Plan 03-01)
    merge.py                # NEW: merge strategies, conflict detection (Plan 03-03)
    rebase.py               # NEW: rebase, cherry-pick (Plan 03-04)
    dag.py                  # NEW: DAG utilities (merge_base, ancestor queries) (Plan 03-01)
  models/
    branch.py               # NEW: BranchInfo model
    merge.py                # NEW: MergeResult, ConflictInfo, Resolution models
  storage/
    schema.py               # MODIFIED: add commit_parents table
    repositories.py         # MODIFIED: add CommitParentRepository ABC
    sqlite.py               # MODIFIED: add SqliteCommitParentRepository
```

### Pattern 1: Multi-Parent Commit Storage (Association Table)

**What:** Store merge commit parents in a separate `commit_parents` table rather than modifying the existing `parent_hash` column.

**Why:** The current `parent_hash` column on `CommitRow` stores a single parent. Merge commits need two parents. Rather than changing this column to a JSON array or similar, use a proper relational association table. This preserves backward compatibility -- `parent_hash` remains the "first parent" (equivalent to git's first parent), and additional parents are looked up from the association table.

**Schema:**

```python
class CommitParentRow(Base):
    """Association table for multi-parent commits (merge commits).

    For non-merge commits, only CommitRow.parent_hash is used (single parent).
    For merge commits, this table stores ALL parents (including the first).
    The 'position' column preserves parent ordering (important for merge semantics).
    """
    __tablename__ = "commit_parents"

    commit_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        primary_key=True,
    )
    parent_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commits.commit_hash"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    # position 0 = first parent (current branch tip), 1 = merged branch tip

    __table_args__ = (
        Index("ix_commit_parents_commit", "commit_hash"),
    )
```

**Rationale:**
- Git stores parent pointers as an ordered list on each commit object. We mirror this with positional ordering.
- `position=0` is the "first parent" (the branch being merged into), matching git's `--first-parent` semantics.
- `position=1` is the "second parent" (the branch being merged from).
- The existing `parent_hash` on `CommitRow` remains as-is for backward compat with linear history traversal (get_ancestors still works for first-parent walks).
- For merge commits, `CommitRow.parent_hash` should equal the first parent (position=0) for consistency.

**Confidence:** HIGH -- this is the standard relational pattern for DAG parent tracking.

### Pattern 2: Merge Base (Lowest Common Ancestor)

**What:** Find the most recent common ancestor of two branch tips for three-way merge.

**Algorithm:** Walk both ancestor chains from each branch tip to root. The merge base is the first commit that appears in both chains. For the existing codebase where history is stored as a linked list via `parent_hash`, this is straightforward:

```python
def find_merge_base(
    commit_repo: CommitRepository,
    hash_a: str,
    hash_b: str,
) -> str | None:
    """Find the best common ancestor (merge base) of two commits.

    Walks both ancestor chains and returns the first intersection.
    For linear history, this is the point where branches diverged.
    For merge commits, must follow ALL parents.
    """
    # Build set of all ancestors of A (including A itself)
    ancestors_a: set[str] = set()
    queue_a = [hash_a]
    while queue_a:
        current = queue_a.pop(0)
        if current in ancestors_a:
            continue
        ancestors_a.add(current)
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue_a.append(commit.parent_hash)
        # Also check commit_parents table for merge commits
        extra_parents = parent_repo.get_parents(current)
        for p in extra_parents:
            if p not in ancestors_a:
                queue_a.append(p)

    # Walk ancestors of B; first one in ancestors_a is the merge base
    queue_b = [hash_b]
    visited_b: set[str] = set()
    while queue_b:
        current = queue_b.pop(0)
        if current in visited_b:
            continue
        visited_b.add(current)
        if current in ancestors_a:
            return current
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue_b.append(commit.parent_hash)
        extra_parents = parent_repo.get_parents(current)
        for p in extra_parents:
            if p not in visited_b:
                queue_b.append(p)

    return None  # No common ancestor (shouldn't happen in same tract)
```

**Confidence:** HIGH -- standard graph algorithm; git's `merge-base` does exactly this (find best common ancestor using BFS/DFS on DAG).

### Pattern 3: Structural Conflict Detection

**What:** Detect merge conflicts by analyzing the commit graph between merge base and each branch tip.

**Algorithm:**
1. Find merge base (common ancestor)
2. Collect commits on branch A (merge_base..tip_a)
3. Collect commits on branch B (merge_base..tip_b)
4. Classify each commit by operation type (APPEND or EDIT)
5. Detect conflicts:
   - **Both EDIT same target:** Two EDIT commits with the same `response_to` -- structural conflict
   - **SKIP vs EDIT:** One branch has SKIP annotation on a commit, the other has EDIT targeting it
   - **EDIT + other branch has appends:** One branch has ANY EDIT, the other branch has appends -- per CONTEXT.md, EDITs are high-stakes and warrant conflict resolution

```python
@dataclass(frozen=True)
class ConflictInfo:
    """Rich context for a conflict, passed to resolver."""
    conflict_type: str  # "both_edit", "skip_vs_edit", "edit_plus_append"
    commit_a: CommitInfo  # Conflicting commit from branch A
    commit_b: CommitInfo  # Conflicting commit from branch B
    ancestor: CommitInfo | None  # Common ancestor commit (if applicable)
    branch_a_history: list[CommitInfo]  # Full branch A history
    branch_b_history: list[CommitInfo]  # Full branch B history
    compiled_context: CompiledContext | None  # Context up to conflict point
```

**Confidence:** HIGH -- the conflict types are locked in CONTEXT.md.

### Pattern 4: OpenAI-Compatible httpx Client

**What:** Minimal sync httpx client for OpenAI chat completions API.

**Request format (OpenAI API):**
```
POST /v1/chat/completions
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Response format:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 18,
    "total_tokens": 60
  }
}
```

**Client implementation pattern:**
```python
class OpenAIClient:
    """Sync httpx client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self._api_key = api_key or os.environ.get("TRACT_OPENAI_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        self._max_retries = max_retries

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.HTTPStatusError,)),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=30)
            + tenacity.wait_random(0, 2),
        stop=tenacity.stop_after_attempt(3),
        before_sleep=_log_retry,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send chat completion request, return parsed response dict."""
        payload: dict = {
            "model": model or self._default_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
```

**Confidence:** HIGH -- httpx sync client patterns are well-established; OpenAI API format is stable.

### Pattern 5: Resolver Protocol

**What:** Pluggable resolver callable that receives conflict info and returns resolution.

```python
@runtime_checkable
class ResolverCallable(Protocol):
    """Protocol for conflict resolution callables.

    Can be a function, lambda, or class with __call__.
    """
    def __call__(self, issue: ConflictInfo) -> Resolution: ...


@dataclass
class Resolution:
    """Result of conflict resolution."""
    action: Literal["resolved", "abort", "skip"]
    content: BaseModel | None = None  # Resolved content for "resolved"
    reasoning: str | None = None  # LLM's explanation (if applicable)
    generation_config: dict | None = None  # Model/params used
```

**Confidence:** HIGH -- Python Protocol pattern matches existing codebase style (see TokenCounter, ContextCompiler protocols).

### Pattern 6: Commit Replay (Cherry-pick / Rebase)

**What:** Re-apply a commit's changes with a new parent, producing a new commit with a new hash.

Cherry-pick and rebase both fundamentally do the same thing: take a commit's content and create a new commit with that content but different parentage. In git, this means computing a diff against the original parent and applying it to the new parent. In Tract, since commits store complete content (not diffs), the process is simpler:

1. **APPEND commits:** Copy the content blob, create new commit with new parent_hash. The content is self-contained, so no re-diffing is needed.
2. **EDIT commits:** More complex. The EDIT's `response_to` target may not exist on the target branch. This is a "cherry-pick issue" that should be flagged to the resolver.

```python
def replay_commit(
    original: CommitInfo,
    new_parent_hash: str | None,
    commit_engine: CommitEngine,
    blob_repo: BlobRepository,
) -> CommitInfo:
    """Replay a commit with a new parent, producing a new commit."""
    # Load original content
    blob = blob_repo.get(original.content_hash)
    content_dict = json.loads(blob.payload_json)
    content = validate_content(content_dict)

    # Create new commit with different parent
    return commit_engine.create_commit(
        content=content,
        operation=original.operation,
        message=original.message,
        response_to=original.response_to,  # May need remapping for EDITs
        metadata=original.metadata,
        generation_config=original.generation_config,
    )
```

**Confidence:** MEDIUM -- the general pattern is clear, but EDIT commit replay with response_to remapping needs careful design during planning.

### Anti-Patterns to Avoid

- **Storing parent list as JSON in CommitRow:** Breaks referential integrity, makes JOIN queries impossible, defeats the purpose of a relational schema.
- **Modifying existing parent_hash to be nullable/multi-value:** Breaks all existing get_ancestors() calls and linear history traversal.
- **Making the openai library a dependency:** Adds heavy dependency for a minimal use case; httpx is sufficient.
- **Async LLM calls in a sync library:** Per CONTEXT.md, everything is synchronous. Do not introduce async.
- **Auto-importing LLM modules:** Keep `llm/` package lazy-loaded so users who don't use LLM features don't pay for httpx import.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP retry with backoff | Custom retry loop with sleep() | tenacity decorator | Handles jitter, max attempts, exception filtering, logging; battle-tested |
| HTTP client | urllib3/requests wrapper | httpx.Client | Modern, well-typed, supports sync+async, proper resource cleanup |
| Merge base algorithm | Custom recursive graph walk | BFS on ancestor sets (implemented above) | Simple, correct, O(n) where n = total commits. No library needed -- the algorithm is 15 lines |
| Three-way text merge | Custom line-by-line differ | This domain doesn't need text merge | Tract merges structured content (Pydantic models), not text files. Conflicts are at the commit level, not the line level |
| Branch name validation | Custom regex | Simple validation function | Git naming rules: no `.lock`, no `..`, no whitespace, no `~^:?*[\`, no leading/trailing `.`, must not be empty |

**Key insight:** Tract's merge is fundamentally different from git's text merge. Git merges file contents line-by-line. Tract merges context commit chains at the structural level (which commits conflict, which can auto-merge). The "content merge" for conflicts is either (a) human-provided resolution or (b) LLM-synthesized resolution -- never an algorithmic text merge.

## Common Pitfalls

### Pitfall 1: get_ancestors() Only Follows parent_hash (First Parent)

**What goes wrong:** The existing `get_ancestors()` walks `parent_hash` pointers, which is the first-parent chain. After merge commits exist, this misses the second parent's history entirely.
**Why it happens:** Linear history didn't need multi-parent support.
**How to avoid:** Add a `get_all_ancestors()` method that also follows `commit_parents` table entries. Keep `get_ancestors()` as first-parent-only (useful for `log --first-parent` equivalent). Use `get_all_ancestors()` for merge base computation and reachability checks.
**Warning signs:** Merge base returns None for branches that clearly share history.

### Pitfall 2: Compile Cache Invalidation on Branch Switch

**What goes wrong:** Switching branches doesn't invalidate the compile cache, so `compile()` returns stale results from the previous branch.
**Why it happens:** The LRU cache is keyed by head_hash, which changes on branch switch. This should actually work correctly -- different branch tips have different head_hashes. But merge commits introduce multi-path compilation that the current compiler doesn't handle.
**How to avoid:** The compiler's `_walk_chain()` follows `parent_hash` (first-parent only). For merge commits, the compiler must follow ALL parents to build the complete effective commit list. This is a required change to `DefaultContextCompiler`.
**Warning signs:** Compiled context after merge is missing commits from the merged branch.

### Pitfall 3: CommitEngine.create_commit() Always Sets parent_hash from HEAD

**What goes wrong:** `create_commit()` auto-reads HEAD and sets `parent_hash`. For merge commits, we need TWO parents, but the engine only supports one.
**Why it happens:** The engine was designed for linear history.
**How to avoid:** Either (a) add a `create_merge_commit()` method that accepts multiple parents, or (b) extend `create_commit()` with an optional `extra_parents: list[str]` parameter. Option (a) is cleaner -- merge commits are structurally different and deserve their own method.
**Warning signs:** Merge commits only record one parent; second parent's history is lost.

### Pitfall 4: Commit Hash Doesn't Include All Parents

**What goes wrong:** `commit_hash()` in `hashing.py` only includes `parent_hash` (single parent). A merge commit's identity should include both parents.
**Why it happens:** Hashing was designed for linear history.
**How to avoid:** For merge commits, include a sorted list of all parent hashes in the hash computation. This ensures different merge orders produce different hashes.
**Warning signs:** Two different merge commits (same content, different second parents) produce the same hash -- collision.

### Pitfall 5: Branch Delete While Commits Are Only Reachable Through That Branch

**What goes wrong:** Deleting a branch ref makes commits unreachable if they haven't been merged.
**Why it happens:** Branch refs are the only named pointers to commit chains.
**How to avoid:** Warn (or block) when deleting a branch with unmerged commits. Check if the branch tip is reachable from the current branch before allowing deletion. CONTEXT.md says delete is blocked on current branch; also block on unmerged.
**Warning signs:** Commits become orphaned after branch delete.

### Pitfall 6: LLM Client Errors During Merge Leave State Inconsistent

**What goes wrong:** If the LLM call fails mid-merge (rate limit, timeout, network error), the merge operation has partially executed but the merge commit isn't created.
**Why it happens:** The merge flow has multiple steps: detect conflicts, call resolver, create merge commit.
**How to avoid:** Per CONTEXT.md, the flow is: detect -> block -> resolve -> commit. If resolution fails, nothing is committed. The `MergeResult` pattern ensures atomicity: either we have a fully resolved merge ready to commit, or we don't commit at all. Use `session.rollback()` on failure.
**Warning signs:** Partially merged state with no merge commit.

### Pitfall 7: Tenacity Retries on Non-Retryable Errors

**What goes wrong:** Retrying on 401 (bad API key) or 400 (bad request) wastes time and confuses users.
**Why it happens:** Default retry catches all HTTPStatusError.
**How to avoid:** Only retry on 429 (rate limit), 500, 502, 503, 504 (server errors). Immediately fail on 401, 403, 400, 404.
**Warning signs:** 3x delay on auth errors; users waiting 30+ seconds for a bad API key to fail.

## Code Examples

### Branch Creation and Switching

```python
# In operations/branch.py
def create_branch(
    name: str,
    tract_id: str,
    ref_repo: RefRepository,
    commit_repo: CommitRepository,
    *,
    source: str | None = None,  # Defaults to HEAD
    switch: bool = True,
) -> str:
    """Create a new branch pointing at source commit.

    Args:
        name: Branch name (validated against git naming rules).
        source: Commit hash to branch from. Defaults to HEAD.
        switch: If True, switch HEAD to new branch.

    Returns:
        The commit hash the new branch points to.

    Raises:
        TraceError: If branch name already exists or is invalid.
    """
    validate_branch_name(name)

    # Check branch doesn't already exist
    existing = ref_repo.get_branch(tract_id, name)
    if existing is not None:
        raise BranchExistsError(name)

    # Resolve source
    if source is None:
        source = ref_repo.get_head(tract_id)
        if source is None:
            raise TraceError("Cannot create branch: no commits exist")

    # Create branch ref
    ref_repo.set_branch(tract_id, name, source)

    # Switch to new branch if requested
    if switch:
        ref_repo.attach_head(tract_id, name)

    return source
```

### Merge Commit Creation

```python
# In operations/merge.py
def create_merge_commit(
    commit_engine: CommitEngine,
    parent_repo: CommitParentRepository,
    content: BaseModel,
    parent_hashes: list[str],
    *,
    message: str | None = None,
    metadata: dict | None = None,
    generation_config: dict | None = None,
) -> CommitInfo:
    """Create a merge commit with multiple parents.

    The first parent (parent_hashes[0]) is the current branch tip.
    The second parent (parent_hashes[1]) is the merged branch tip.

    Uses CommitEngine.create_commit for the first parent, then records
    additional parents in the commit_parents table.
    """
    # Create the commit (first parent handled by engine)
    info = commit_engine.create_commit(
        content=content,
        operation=CommitOperation.APPEND,
        message=message or f"Merge commit",
        metadata=metadata,
        generation_config=generation_config,
    )

    # Record all parents in association table
    for position, parent_hash in enumerate(parent_hashes):
        parent_repo.add_parent(info.commit_hash, parent_hash, position)

    return info
```

### httpx Client with Tenacity Retry

```python
# In llm/client.py
import os
import logging
import httpx
import tenacity

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def _is_retryable(exc: BaseException) -> bool:
    """Check if an HTTP error is retryable."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


class OpenAIClient:
    """Sync httpx client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self._api_key = api_key or os.environ.get("TRACT_OPENAI_API_KEY", "")
        self._base_url = (
            base_url
            or os.environ.get("TRACT_OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> dict:
        """Send chat completion request with retry.

        Returns the full response dict (caller extracts choices/usage).
        """
        # Use tenacity programmatically (not decorator) for configurable max_retries
        retryer = tenacity.Retrying(
            retry=tenacity.retry_if_exception(_is_retryable),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=30)
                + tenacity.wait_random(0, 2),
            stop=tenacity.stop_after_attempt(self._max_retries),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        return retryer(self._do_chat, messages, model=model,
                       temperature=temperature, max_tokens=max_tokens, **kwargs)

    def _do_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> dict:
        payload: dict = {
            "model": model or self._default_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        response = self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()
```

### LLM Client Protocol

```python
# In llm/protocols.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMClient(Protocol):
    """Protocol for pluggable LLM clients.

    Any object with a chat() method matching this signature works.
    """
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send messages, return response dict with 'choices' and 'usage'."""
        ...
```

### Merge Base Computation

```python
# In operations/dag.py
def find_merge_base(
    commit_repo: CommitRepository,
    parent_repo: CommitParentRepository,
    hash_a: str,
    hash_b: str,
) -> str | None:
    """Find best common ancestor of two commits using BFS."""
    # Collect all ancestors of A
    ancestors_a: set[str] = set()
    queue: list[str] = [hash_a]
    while queue:
        current = queue.pop(0)
        if current in ancestors_a:
            continue
        ancestors_a.add(current)
        # Follow first parent
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue.append(commit.parent_hash)
        # Follow extra parents (merge commits)
        for extra in parent_repo.get_parents(current):
            queue.append(extra)

    # BFS from B; first hit in ancestors_a is merge base
    visited: set[str] = set()
    queue = [hash_b]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if current in ancestors_a:
            return current
        commit = commit_repo.get(current)
        if commit and commit.parent_hash:
            queue.append(commit.parent_hash)
        for extra in parent_repo.get_parents(current):
            queue.append(extra)

    return None
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| openai library for API calls | Direct httpx (lighter, OpenAI-compat generic) | 2024+ | Avoids vendor lock-in, smaller dependency |
| requests + manual retry | httpx + tenacity | 2023+ | Modern async-ready client, composable retry |
| Single parent_hash column | Association table for multi-parent | Phase 3 | Enables merge commits, DAG traversal |

**Current OpenAI API state (as of 2025):**
- Chat Completions endpoint: `POST /v1/chat/completions`
- Response includes `usage.prompt_tokens` and `usage.completion_tokens`
- Base URL configurable for OpenAI, Ollama, Together, vLLM, Azure, etc.
- All OpenAI-compatible providers use the same request/response format

## Critical Schema Migration

### Adding commit_parents Table

This is a new table (no existing data to migrate). The `init_db()` function already calls `Base.metadata.create_all(engine)` which will create new tables automatically. No migration script needed for in-memory databases. For file-backed databases, the table simply won't exist until the database is re-initialized or upgraded.

**Schema version:** Should bump from 1 to 2 in `_trace_meta`. Add a migration check: if schema_version=1 and commit_parents table doesn't exist, create it.

### CommitRow Changes

The existing `parent_hash` column stays as-is. No schema change to CommitRow. The `commit_parents` table is additive.

### Compiler Changes

The `DefaultContextCompiler._walk_chain()` must be updated to handle merge commits:
- When walking the chain and encountering a merge commit, follow ALL parents (via commit_parents table)
- Produce a topologically sorted effective commit list
- Handle the "branch blocks" ordering from CONTEXT.md: all commits from one branch, then all from the other

## Interaction with Existing Systems

### Compile Cache Impact

- Branch switching: Already works (LRU keyed by head_hash)
- Merge commits: New head_hash, cache miss, full recompile -- correct behavior
- After merge: The compiled context includes commits from both branches, so the snapshot will be larger
- No cache invalidation issues specific to branching

### CommitEngine Impact

- `create_commit()`: No change for regular commits
- New `create_merge_commit()`: Creates commit + writes to commit_parents table
- `update_head()`: Already handles attached/detached correctly
- Token budget: Merge commit content counts toward budget normally

### Operations Impact

- `log()`: Should support `--first-parent` mode (follow parent_hash only) and `--all` mode (follow all parents)
- `diff()`: No change -- still compares two compiled contexts
- `status()`: Should show merge/rebase in-progress state
- `reset()`/`checkout()`: No change -- still move HEAD pointer

## Plan Decomposition Guidance

Based on CONTEXT.md's 4-plan structure:

### Plan 03-01: Branch and Switch Operations
- `commit_parents` association table + repository
- `operations/branch.py`: create, delete, list, switch
- `operations/dag.py`: merge_base, ancestor walking with multi-parent support
- Branch name validation
- SDK methods on Tract: `branch()`, `switch()`, `list_branches()`, `delete_branch()`
- Update `DefaultContextCompiler` to handle multi-parent chains (topological sort)
- Tests: branch create/switch/delete, merge_base, multi-parent walk

### Plan 03-02: LLM Client Infrastructure
- `llm/` package: client.py, protocols.py, resolver.py, errors.py
- OpenAIClient (httpx + tenacity)
- LLMClient protocol
- ResolverCallable protocol
- Built-in OpenAIResolver
- Tract-level config: `configure_llm()`, default model, env vars
- Tests: client (mocked httpx), protocol conformance, retry behavior

### Plan 03-03: Merge Strategies
- Conflict detection (structural analysis)
- Fast-forward merge (pointer move)
- Divergent merge commit (two parents, branch blocks ordering)
- LLM-mediated semantic merge (via resolver)
- MergeResult model for conflict review
- `Tract.merge()`, `Tract.commit_merge()`
- Tests: fast-forward, clean merge, conflict detection, resolution flow

### Plan 03-04: Rebase, Cherry-pick, and Semantic Safety Checks
- Commit replay (cherry-pick single commit)
- Rebase (replay chain of commits)
- Semantic safety checks (reorder detection)
- CherryPickIssue, RebaseWarning typed issues
- `Tract.rebase()`, `Tract.cherry_pick()`
- Tests: cherry-pick, rebase, safety check triggers

## Open Questions

1. **Topological sort for merged contexts:**
   - What we know: After merge, the compiled context must include commits from both branches in a sensible order.
   - What's unclear: The "branch blocks" ordering (all of A, then all of B) is clear for the simple case, but what about nested merges (merge of merges)?
   - Recommendation: For Phase 3, implement simple branch-blocks ordering. Nested merges can be handled by recursive flattening in a later phase if needed.

2. **Merge commit content model:**
   - What we know: Merge commit has "content = resolved/merged content" per CONTEXT.md.
   - What's unclear: What content type should a merge commit use? A new `MergeContent` type? Or the same type as the resolved conflict?
   - Recommendation: Use `FreeformContent` or a new `MergeContent` type with a `resolutions` field. The merge commit should be a single message summarizing the merge. For clean merges (no conflicts), the content can be a simple "Merged branch X into Y" message.

3. **Token tracking for LLM calls during merge:**
   - What we know: Per CONTEXT.md, merge/rebase LLM calls should be labeled `infrastructure:merge`.
   - What's unclear: How exactly to record these -- as metadata on the merge commit? As separate usage records?
   - Recommendation: Record in the merge commit's `generation_config` with a `source: "infrastructure:merge"` field. Also call `record_usage()` if the TokenUsageExtractor is available.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/tract/storage/schema.py` -- CommitRow, RefRow, current schema
- Codebase analysis: `src/tract/storage/repositories.py` -- RefRepository ABC with branch support
- Codebase analysis: `src/tract/storage/sqlite.py` -- SqliteRefRepository with symbolic HEAD
- Codebase analysis: `src/tract/engine/commit.py` -- CommitEngine.create_commit() pattern
- Codebase analysis: `src/tract/engine/compiler.py` -- DefaultContextCompiler._walk_chain()
- Codebase analysis: `src/tract/engine/hashing.py` -- commit_hash() inputs
- Codebase analysis: `src/tract/protocols.py` -- TokenCounter, ContextCompiler protocols
- CONTEXT.md: All 10 locked decisions (autonomy modes, resolver pattern, merge semantics, etc.)

### Secondary (MEDIUM confidence)
- [HTTPX Transports documentation](https://www.python-httpx.org/advanced/transports/) -- retry behavior, transport customization
- [Tenacity documentation](https://tenacity.readthedocs.io/) -- retry strategies, wait/stop patterns
- [SQLAlchemy Adjacency List docs](https://docs.sqlalchemy.org/en/20/orm/self_referential.html) -- self-referential patterns
- [Git merge-base documentation](https://git-scm.com/docs/git-merge-base) -- merge base algorithm description
- [Git cherry-pick documentation](https://git-scm.com/docs/git-cherry-pick) -- cherry-pick semantics

### Tertiary (LOW confidence)
- [Jujutsu VCS FAQ](https://jj-vcs.github.io/jj/latest/FAQ/) -- alternative merge storage model (reference only)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- httpx and tenacity are well-established; codebase analysis is direct
- Architecture: HIGH -- patterns derived from existing codebase structure and locked CONTEXT.md decisions
- Schema changes: HIGH -- association table for multi-parent is standard relational pattern
- Merge algorithms: HIGH -- merge base BFS is textbook; conflict types are locked in CONTEXT.md
- LLM client: HIGH -- OpenAI API format is stable; httpx usage patterns are well-documented
- Rebase/cherry-pick: MEDIUM -- general pattern is clear but EDIT commit replay has edge cases
- Compiler updates: MEDIUM -- multi-parent chain walking adds complexity; topological sort for merged histories needs careful implementation

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (stable domain; httpx/tenacity APIs unlikely to change)
