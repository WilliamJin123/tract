# Phase 3: Branching & Merging — CONTEXT.md

> Decisions locked during discussion. Guides research and planning.

---

## 1. Design Philosophy

**Three autonomy modes** supported everywhere:
- **Manual:** No LLM involved. Human reviews conflicts, edits content directly.
- **Collaborative (DEFAULT):** LLM resolves, human reviews before commit.
- **Autonomous:** LLM resolves and commits, no human in the loop.

**Universal pattern** for all operations that transform context (merge, rebase, cherry-pick, and eventually compression):
```
detect issue → block → call resolver → commit only after resolution
```

**Config pattern:** Default on Tract, overridable per-operation (applies to: LLM client, autonomy mode, resolver, model).

**No staging concept.** Commits are atomic. `batch()` handles groups. MergeResult handles conflict review. Branch switching is always safe.

---

## 2. LLM Client

**Built-in OpenAI-compatible client:**
- httpx-based, sync only, **required dependency**
- Configurable: `api_key`, `base_url` (covers OpenAI, Ollama, Together, vLLM, Azure, etc.)
- Env var support: `TRACT_OPENAI_API_KEY`, `TRACT_OPENAI_BASE_URL`

**Callable protocol escape hatch:**
- Any object conforming to a minimal protocol works for non-OpenAI-compatible providers
- Set on Tract: `tract.configure_llm(client)`
- Override per-operation: `tract.merge("branch", model="gpt-4o-mini")`

**Model selection:**
- Default model set on client, overridable per-operation
- Model used for semantic operations recorded in merge commit's `generation_config`

**Retry behavior:**
- Backoff + max attempts
- Per-error-type policies
- Callbacks on retry

**Token tracking:**
- Merge/rebase LLM calls labeled `infrastructure:merge` (filterable)
- Separate from conversation token tracking

---

## 3. Merge Conflict Semantics

### Structural Conflicts (auto-detected, free)

Three types, all detected from commit graph structure:

1. **Both EDIT same commit** — two branches modified the same original commit differently
2. **SKIP vs EDIT** — one branch removed a commit, the other changed it
3. **Either branch has EDIT + other has appends** — EDITs are always high-stakes operations that change critical context; downstream appends may be invalidated

> Design rationale: EDITs are never trivial (not used for typos). They change critical context that affects future runs. Any EDIT on a divergent branch warrants conflict resolution.

### Non-Conflicts (auto-merge, no LLM)

- **Fast-forward** — one branch is direct ancestor of other. Just move pointer.
- **Both branches only APPEND** — no overlap. Auto-merge with **branch blocks** ordering (all of branch A's commits, then all of branch B's). Preserves each branch's narrative coherence for LLM comprehension.

### Semantic Review (opt-in)

- `strategy="semantic"` — LLM reviews entire merged context for contradictions (e.g., "dark mode" vs "light mode" in separate appends)
- Catches issues structural detection misses
- Also configurable as Tract-level policy: `merge_review="always"`
- Cost/quality tradeoff controlled by the user

### Merge Strategies

| Scenario | Default behavior |
|---|---|
| Fast-forward possible | Auto FF (pointer move, no merge commit) |
| Diverged, no conflicts | Auto-merge (branch blocks ordering), merge commit with two parents |
| Structural conflict | Block → resolver called → MergeResult returned for review |
| Semantic review requested | LLM reviews full merged context |

`no_ff=True` option forces merge commit even when fast-forward is possible.

---

## 4. Resolver Pattern

**Single resolver callable** (not hooks/pipeline). Takes a typed issue, returns resolution.

### Typed Issue Subtypes

| Type | Trigger | Resolver receives | Resolver returns |
|---|---|---|---|
| `MergeConflict` | Both EDIT same commit, SKIP vs EDIT, EDIT+APPEND | Both versions, ancestor, full context | Resolved content |
| `RebaseWarning` | Reordering changes meaning | Reordered commit, new base, context | Proceed/abort/modified content |
| `CherryPickIssue` | Commit lacks context in target | Out-of-context commit, target state | Proceed/abort/adapted content |

### Rich ConflictInfo

Resolver receives everything: conflicting commits, common ancestor, both branch histories, compiled context up to conflict point. Resolver uses what it needs — the protocol is rich, the implementation controls cost.

### Built-in OpenAIResolver

- Cost-smart: sends conflicting commits + surrounding context, not entire history
- Configurable model (per-call default + override)
- Ships with Tract (uses the configured LLM client)

### Customization Levels

1. **Prompt template** (quick): `tract.merge("branch", resolver_prompt="Always prefer newer version")` — injects custom system prompt into built-in resolver
2. **Callable protocol** (full control): `tract.merge("branch", resolver=my_resolver)` — any callable matching the protocol

---

## 5. Merge Flow (Hybrid)

**Clean merges (no conflicts):** auto-commit. One step, no ceremony.

```python
tract.merge("feature")  # fast-forward or clean merge, committed automatically
```

**Conflict merges:** return MergeResult for review, user finalizes.

```python
result = tract.merge("feature")
# result.conflicts → resolved conflicts
# result.preview → merged context preview
# result.resolution_reasoning → LLM's explanation

result.edit_resolution("commit_abc", "my fixed content")  # tweak if needed
tract.commit_merge(result)  # finalize
```

**Autonomous override:** `auto_commit=True` to skip review even for conflicts.

### Merge Commit Structure (Lean Option 2)

- Merge commit content = resolved/merged content (compile sees it normally)
- Two parents: current branch HEAD + merged branch HEAD
- `generation_config` records model/params used for resolution
- Both original commits reachable via parents (audit trail)
- LLM output: single synthesized message per conflict

---

## 6. Safety Check Behavior

**Block until resolved.** When any safety check (rebase reordering, cherry-pick coherence, merge conflict) detects an issue:

1. **Detect** — structural analysis flags the issue
2. **Block** — operation pauses, nothing committed
3. **Resolve** — call the resolver:
   - No resolver configured → raise error with conflict details
   - Resolver provided → pass ConflictInfo, get resolution back
4. **Commit** — only after resolution

**No warn-and-continue.** Incoherent context never lands silently.

---

## 7. Branch Management UX

### Naming & Organization
- Default branch: `main` (created on Tract init)
- Git-style naming rules with `/` namespacing (e.g., `feature/auth`, `explore/flask`)
- Optional description metadata per branch

### Lifecycle
- Create: auto-switch to new branch, configurable source (defaults to HEAD)
- Delete: blocked on current branch
- Post-merge: keep by default, `delete_branch=True` option
- Soft limit: configurable (default 10), warning only (no hard block)

### Listing
- Minimal by default (just names)
- `-v` / verbose option shows: description, commit count, last activity, ahead/behind main

### CLI Commands (Phase 3)
- `tract branch` — list, create, delete branches
- `tract switch` — switch between branches
- `tract merge` — merge branches

### Branch Switching
- Always safe (no staging concept, commits are atomic)
- Just moves HEAD pointer to target branch

---

## 8. Sync Model

**Everything is synchronous** (like git). Including LLM calls — merge blocks while the LLM resolves. Users in async contexts use `asyncio.to_thread()`.

Rationale: Library is sync throughout (SQLAlchemy sync sessions). Adding async for one subsystem creates inconsistency. LLM call during merge is the one slow operation, but making the whole library async for one use case isn't worth it.

---

## 9. Deferred Ideas

- **Async support** — whole-library async migration if demand warrants it (not Phase 3)
- **Agent SDK toolkit** — tool definitions wrapping Tract SDK for agent frameworks (Phase 5 territory)
- **Staging area** — not needed given atomic commits and batch(), could revisit if use cases emerge
- **Chronological merge ordering** — decided against (branch blocks preserves narrative), but could be an option later

---

## 10. Key Terminology

| Term | Meaning |
|---|---|
| Structural conflict | Auto-detected from commit graph (EDIT overlap, SKIP vs EDIT, EDIT+APPEND) |
| Semantic review | Opt-in LLM review of full merged context for contradictions |
| Resolver | Callable that takes ConflictInfo and returns Resolution |
| Branch blocks | Merge ordering: all of branch A, then all of branch B |
| MergeResult | Return object from conflict merge, reviewed before commit_merge() |
| ConflictInfo | Rich object passed to resolver: conflicts + ancestors + context |
