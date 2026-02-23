# Cross-Tract Merge: Reference-Based

## Problem

When a child tract finishes work (e.g., a delegated research subtask), there's
no structural way to bring its results back into the parent tract. Today you'd
manually copy content, losing provenance.

## Goal

A parent tract can **merge from a child tract** by referencing the child's
commits directly in the DAG — no copying, no duplication. The parent's commit
history structurally records "this content came from that agent's work."

## Approach: Reference-Based (Not Copy-Based)

The merge commit in the parent tract has a second parent pointing to a commit
in the child tract. The DAG itself is the provenance trail.

**Why not copy-based:**
- Data duplication compounds with deep delegation chains (agent -> sub-agent -> sub-sub-agent)
- Copies are stale snapshots that diverge from the source
- Provenance becomes metadata annotations instead of structural graph edges
- Would need to be migrated away from later

**Why reference-based works:**
- All tracts share one SQLite database; FKs already work across tract boundaries
- Multi-parent merge machinery (CommitParentRow, DAG traversal, compiler merge
  expansion) already exists for intra-tract merges
- The extension is: allow those parent references to cross tract_id boundaries

## Use Cases

### 1. Agent Delegation Round-Trip

A coordinator agent delegates a subtask, waits for results, merges them back:

```
Coordinator Tract:
  system("You coordinate research tasks")
  user("Write a report on X")
  assistant("I'll delegate the research phase")
  → spawn child tract
  → child does independent work
  → merge_from(child) ← structural link to child's commits
  assistant("Based on the research, here's the report...")
```

The coordinator's compiled context includes the child's content, and the DAG
records exactly where it came from.

### 2. Multi-Agent Recombination

Multiple child agents work in parallel, each producing results that get merged
into a single parent context:

```
Parent Tract:
  ← merge_from(researcher)     position 1
  ← merge_from(fact_checker)   position 1
  ← merge_from(editor)         position 1
```

Each merge commit references a different child tract. The full computation
graph is preserved — you can trace any piece of content back to the agent that
produced it and the context it had.

### 3. Deep Delegation Chains

Agent A delegates to B, B delegates to C. When C finishes, B merges from C,
then A merges from B. With reference-based linking, A's DAG transitively
reaches all the way to C's original commits — no duplication at any level.

### 4. Audit and Replay

For debugging multi-agent workflows: follow the DAG to answer "which agent
produced this content, and what was its full context when it did?" The answer
is structural, not a metadata annotation that could be wrong or missing.

## Design Questions to Resolve During Implementation

1. **Compiler tract boundary crossing**: Does `get_ancestors()` filter by
   `tract_id`? If so, merge-parent expansion needs to relax that filter for
   cross-tract parents.

2. **Annotations on cross-tract commits**: When compiling the parent tract,
   should the parent's annotations (SKIP/PINNED) apply to referenced child
   commits? Probably yes for compile purposes.

3. **Edit targeting across tracts**: Can a parent tract EDIT a child's commit?
   Probably not — edits should stay intra-tract. The parent can SKIP or
   annotate cross-referenced commits instead.

4. **Token budget accounting**: Cross-tract referenced content counts against
   the parent's token budget during compile. Need to ensure the compiler
   accounts for this.

5. **Child tract lifecycle**: What happens if a child tract is deleted or GC'd
   after being referenced? SET NULL FK behavior already exists — need to decide
   if referenced child commits should be protected from GC.

6. **Selection policy**: `merge_from()` pulls in all child commits? Only
   commits after the spawn point? A user-specified range? Likely "everything
   after spawn point" as default with optional filtering.

## Existing Infrastructure That Supports This

- **CommitParentRow**: Multi-parent association table, FK to `commits.commit_hash`
  (no tract_id constraint — already works cross-tract)
- **create_merge_commit()**: Creates merge commits with multiple parents
- **Compiler merge expansion**: Walks second-parent ancestors and inserts them
  as "branch blocks" before the merge commit
- **DAG BFS traversal**: Follows both first-parent and extra parents
- **SpawnPointerRow**: Already records which parent commit the child was born from
  (provides the merge-base equivalent for cross-tract merges)

## Estimated Scope

Single-phase project. Main pieces:
- `Tract.merge_from(child_tract)` method
- Compiler adjustments for cross-tract parent walking
- Annotation/edit policy for cross-tract commits
- GC protection for referenced commits
- Tests and edge cases
