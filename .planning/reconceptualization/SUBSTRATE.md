# Substrate: Fundamental Graph Primitives

The substrate is the minimal set of operations that cannot be decomposed
further. Everything else is built from these via rules.

## The Test

Can you express it as `scope + condition + action` over the commit DAG?
- If yes → application (built from rules)
- If no → substrate (graph primitive)

## Graph Primitives

### 1. Add Node (APPEND)
Add a content node with an edge from HEAD. The most primitive write operation.

### 2. Add Parallel Node (EDIT)
Add an alternative node linked to the same position. Creates a parallel edge:
```
A → B1 → C
     ↕
     B2
```
B2 doesn't delete B1 — compile resolves which to use. Represents revision
while preserving history.

### 3. Fork (BRANCH)
Split the graph into divergent paths from a shared node.

### 4. Move Pointer (SWITCH)
Change which path HEAD points to. Graph navigation, not mutation.

### 5. Join Paths (MERGE)
Converge two paths into a single node. The structural operation (joining)
is substrate. How to resolve conflicts (pick ours, LLM-resolve, etc.) is
application/rules.

### 6. Collapse Subgraph (COMPRESS)
Replace N nodes with 1 summary node in the compiled view. Could technically
decompose into batch delete + append, but the **atomicity guarantee** matters —
"these N nodes become this 1 node" must happen as one operation. Earns its
place as a primitive for the same reason databases have REPLACE alongside
INSERT + DELETE.

### 7. Transplant Subgraph (REBASE)
Re-parent a subgraph onto a new base. Structural re-parenting of nodes.
When to rebase and what to include/exclude is application/rules.

### 8. Remove Node (DELETE)
Prune a node from the graph. Needed for GC but the GC policy itself
(what to prune, retention period) is application/rules.

### 9. Attach Metadata (ANNOTATE)
Attach key-value metadata to a node. The ability to annotate is substrate.
Specific semantics (PINNED = never compress, SKIP = exclude from compile)
are application-level rules.

### 10. Resolve to Linear (COMPILE)
Resolve the DAG into a linear sequence of messages. The read operation.
Traverses the graph, resolves edits, applies priority filtering, respects
active rules.

Compile accepts a **strategy** that controls detail level:
- `full` — all content (current default)
- `messages` — commit messages only, no content bodies (free, no LLM)
- `adaptive(k)` — last K commits at full detail, rest at messages-only

Strategy is a compile-time parameter, not a mutation. The DAG is unchanged.
The agent can set its compile strategy via rules (see RULES.md).

## Supporting Substrate

These aren't operations but are substrate-level capabilities the rule
system needs:

### Token Counting
Measurement primitive on the graph. Without it, rules about context
management ("compress when > 4000 tokens") are blind. Already implemented
in current tract (tiktoken + API-reported).

### Temporal Ordering
Commits have timestamps and topological order. Rules reference both:
"retain for 24 hours" (wall-clock), "last 5 messages" (topological).
Already exists via commit timestamps and DAG structure.

### Identity / Addressing (Querying)
How rules reference specific nodes or subgraphs.

Current: hashes + annotations/tags + role.

Decision: **Start with simple queries (hash, tag, role) + strong tool
compaction. Add rich querying where real usage reveals pain points.**
Tag-based filtering is substrate from day one since the entire rule
system depends on it for cross-stage data references.

Rich querying (complex multi-field filters) starts as "search + compact"
and graduates to query primitives via the promotion loop if repeatedly
useful.

### Structured Metadata (MetadataContent)
Agent-maintained structured knowledge about its environment: file trees,
dependency graphs, project plans, environment configs. Not conversation
content — workspace state the agent updates as it works.

- `compilable=False` (like RuleContent — never rendered as LLM messages)
- Updated in-place via EDIT operations
- Preserved across compression (engine treats like rules)
- Optional `path` field for filesystem export/sync
- Queryable by `kind` without full compile

Bridges the gap between tract's SQL storage and agents' natural file-based
state management. The source of truth is always tract; files are views.

## Classification Table

| Operation | Substrate (graph primitive) | Application (rules/behavior) |
|---|---|---|
| Append | add node + edge | — |
| Edit | add parallel node | — |
| Branch/Switch | fork graph / move HEAD | — |
| Merge | join two paths into one node | conflict resolution strategy |
| Compress | collapse N nodes → 1 node | when/how to summarize |
| Rebase | re-parent subgraph | when to rebase, what to skip |
| Delete | remove node from graph | GC policy, retention rules |
| Annotate | attach metadata to node | PINNED/SKIP semantics |
| Compile | resolve DAG → linear | — |
| Cherry-pick | — (just append from existing) | — |
