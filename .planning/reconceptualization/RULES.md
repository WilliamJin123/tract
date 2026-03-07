# The Unified Rule System

> **Status: SUPERSEDED by Phase 14 (Config + Directives + Middleware).** Rule engine was implemented (R0-R4, commit 7a86b94) then replaced (commit 23a89eb). Kept as historical reference.

## Core Concept

A rule is: **trigger + condition + action**, scoped by where it lives in
the DAG. This replaces the current scattered system of hooks, triggers,
policies, and orchestrator callbacks with one unified model.

## Rule Schema

```
RuleContent:
  name:      str             # stable identity, human-readable
  trigger:   str             # when the engine evaluates this rule
  condition: dict | None     # {type: str, ...} or None = always fires
  action:    dict            # {type: str, ...}
```

Four fields. Everything else comes from context:
- **Scope** → where the commit lives in the DAG (branch placement)
- **Precedence** → DAG distance from HEAD (closer wins)
- **Provenance** → commit metadata_json (source: developer/llm/promoted)
- **Status** → substrate annotations (SKIP to disable, DELETE to remove)

"Editing" a rule = commit a new RuleContent with the same name. The closer
one wins via precedence. The old version stays in the DAG (history preserved).

## Scope via DAG Placement

A rule's scope is determined by WHERE it's committed, not by explicit fields:
- Committed on **workflow root** → inherited by all branches (global rules)
- Committed on **stage branch** → active only on that branch and children
- Committed on **child branch** → overrides parent rules via DAG distance

This eliminates redundancy. No explicit scope fields needed. A data-scoped
rule like "never compress pricing data" lives on the workflow root (inherited
everywhere) and uses a tag condition to narrow to specific commits.

## Trigger Events

The `trigger` field determines when the engine checks this rule:

```
"active"                  # continuously in effect (configs)
"pre_commit"              # before a commit is persisted (can block)
"commit"                  # after every commit (reactive: redaction, tagging)
"compile"                 # before every compile
"compress"                # when compression is attempted
"merge"                   # when merge is attempted
"gc"                      # when GC is attempted
"transition"              # any transition from this branch
"transition:{target}"     # transition to specific target branch
```

Triggers are a **registry** — custom triggers (e.g., "webhook:shopify")
can be registered without schema changes.

## Condition Types

Conditions are dicts with a `type` key dispatched to registered evaluators:

```
{type: "tag", tag: str, present: bool}
    Tag existence check on the triggering commit or scope.

{type: "pattern", regex: str}
    Regex match on commit content.

{type: "threshold", metric: str, op: str, value: number}
    Numeric comparison. Metrics are a registry (see Extensibility).
    Built-in: token_count, total_tokens, commit_count, age_hours, branch_depth.

{type: "llm", instruction: str}
    LLM-evaluated condition (fuzzy). The instruction is sent to the LLM
    with the evaluation context; LLM returns true/false.

{type: "all", conditions: [Condition...]}    # AND
{type: "any", conditions: [Condition...]}    # OR
{type: "not", condition: Condition}          # negation
```

`None` condition = always fires (used for configs and unconditional actions).

## Action Types

Actions are dicts with a `type` key dispatched to registered handlers:

```
{type: "set_config", key: str, value: any}
    Set a configuration parameter. Override semantics.

{type: "operation", op: str, params: dict}
    Run a substrate operation: compress, edit, branch, merge, delete,
    annotate, cherry_pick. Accumulate semantics.

{type: "block", reason: str?}
    Prevent the triggering operation/transition. Accumulate semantics.

{type: "require", condition: Condition}
    Block until the embedded condition is met. Used for approval gates.
    Accumulate semantics.

{type: "compile_filter", mode: str, ...mode_params}
    Configure transition handoff compilation. Override semantics.
    Modes: selective, summarized, same_context, new_agent (see below).

{type: "llm", instruction: str}
    LLM-evaluated action. Accumulate semantics.

{type: "create_rule", template: dict}
    Commit a new RuleContent. Template is validated against the schema
    at execution time. Used by the promotion loop.
```

## Engine Conventions

### Override vs Accumulate

When multiple rules match the same event, the engine needs to know whether
they compose or compete:

| Action type    | Semantics   | Rationale                              |
|----------------|-------------|----------------------------------------|
| set_config     | Override    | Can't have two temperatures            |
| compile_filter | Override    | One handoff mode per transition        |
| block          | Accumulate  | ANY block stops the operation          |
| require        | Accumulate  | ALL requirements must be met           |
| llm            | Accumulate  | Independent evaluations all run        |
| operation      | Accumulate  | Independent operations all run         |
| create_rule    | Accumulate  | Independent rule creation              |

Override = closest to HEAD wins. Accumulate = all matching rules execute.

### Execution Order on Transitions

When multiple rules fire on the same transition event, the engine processes
them in a fixed order by action category:

1. **Gates** — require, block (must all pass before anything else)
2. **Work** — llm, operation (pre-transition actions: audits, cleanup)
3. **Handoff** — compile_filter (build the transition payload)
4. **Post** — create_rule, custom actions (side effects, notifications)

Within each category, rules execute in reverse DAG order (furthest from
HEAD first → root-level rules before branch-level rules).

### Precedence

When override-semantic rules conflict, closest to HEAD in DAG ancestry wins.
At the same DAG distance, the later commit (by timestamp) wins. One
precedence dimension, no explicit priority field.

```
child branch rule > parent branch rule > workflow root
```

## Transition Mechanics

A transition is NOT a monolithic object. It decomposes into rules on the
source branch that the engine evaluates when `t.transition(target)` is called:

```python
t.transition("lander_pages")

# Engine does:
# 1. Collect rules with trigger="transition" or "transition:lander_pages"
# 2. Gates: evaluate require/block rules — any failure stops everything
# 3. Work: execute llm/operation actions (audits, cleanup)
# 4. Handoff: execute compile_filter to build payload
# 5. Switch to / create target branch
# 6. Commit handoff payload on target
```

Each concern is an independent rule commit. Add an approval gate by
committing one new rule — no modification to existing transition logic.

### Compile Filter Modes

**selective** — tag/type filtering:
```
{mode: "selective", include_tags: [...], exclude_tags: [...], max_tokens: N}
```

**summarized** — compress source into summary:
```
{mode: "summarized", target_tokens: N, preserve_tags: [...], instruction: "..."}
```

**same_context** — no branching, new stage rules appended to current branch:
```
{mode: "same_context"}
```
**Note:** When stages share a branch via same_context, all rules on that branch
apply to both stages. Stage-specific rules need manual condition guards (e.g.,
tag-based) to scope behavior. Use same_context only when full branch isolation
is overkill (e.g., implementation→validation).

**new_agent** — fresh context, handoff payload only:
```
{mode: "new_agent", include_tags: [...], max_tokens: N, instruction: "..."}
```

## Compile Behavior

RuleContent commits are **never compiled to messages**. During compile,
the engine separates rule commits from content commits. Rules configure
the compilation; content becomes the message list. Implemented via a
`compilable=False` hint on the ContentTypeHints for the rule content type.

## Compile Strategy (Agent Self-Management)

The agent controls its own context window through the rule system. Compile
strategy is an active rule that the orchestrator respects when building
the base context:

```python
# Full compile (default — everything on the branch)
RuleContent(name="context_strategy", trigger="active",
    condition=None,
    action={"type": "set_config", "key": "compile_strategy", "value": "full"})

# Minimal base (GCC-style — agent reads more via tools as needed)
RuleContent(name="context_strategy", trigger="active",
    condition=None,
    action={"type": "set_config", "key": "compile_strategy", "value": "minimal"})

# Adaptive K-window (last K commits full, rest messages-only)
RuleContent(name="context_strategy", trigger="active",
    condition=None,
    action={"type": "set_config", "key": "compile_strategy",
            "value": "adaptive", "k": 5})
```

Strategies:
- **full** — current behavior. Entire branch compiled to messages.
- **minimal** — only latest commit summary + active rules in base context.
  Agent uses read tools (log, get_commit, compile with branch/detail params)
  to load more on demand. Best for long-running agents.
- **adaptive(k)** — last K commits at full detail, everything before at
  commit-messages-only. Non-destructive (no compression). Balances
  orientation and context budget.

The agent can change strategy mid-session via reactive rules:

```python
RuleContent(name="auto_slim", trigger="commit",
    condition={"type": "threshold", "metric": "total_tokens", "op": ">", "value": 20000},
    action={"type": "set_config", "key": "compile_strategy",
            "value": "adaptive", "k": 3})
```

compile() is also exposed as a **read tool** with `branch` and `detail`
parameters — for cross-branch exploration, not re-reading your own context.
The agent calls `compile(branch="feature_x", detail="messages")` to peek
at another branch's progress.

## Configs Are Rules

Configs are rules with `trigger="active"` and `condition=None`:

```python
t.rule("temp",        trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.3})
t.rule("compaction",  trigger="active", action={"type": "set_config", "key": "tool_compaction", "value": True})
```

Current scattered config locations unified under rules:
- LLMConfig (temperature, model) → stage-scoped active rules
- OperationConfigs → operation-scoped active rules
- TractConfig → workflow-root active rules
- Hook handlers → rules with conditions
- Orchestrator profiles → stage-scoped active rules

## Deterministic vs LLM-Evaluated

The engine inspects condition.type and action.type to determine cost:
- Both non-LLM → fully deterministic, no LLM call
- One is LLM → partial LLM evaluation
- Both LLM → fully fuzzy (discouraged for production rules)

This enables short-circuiting: deterministic rules skip LLM evaluation.

## The Promotion Loop

Fuzzy decisions crystallize into deterministic rules over time:

1. LLM makes a decision via consult() (e.g., "redact this API key")
2. System observes the pattern repeating (same decision N times)
3. LLM proposes a deterministic rule
4. Rule committed with metadata `{source: "promoted"}` — evaluated
   alongside LLM for validation
5. After M successful applications, metadata updated — skips LLM call

## Rule Authoring

Three paths to the same committed RuleContent:

1. **Developer SDK**: `t.rule(name, trigger, condition, action)` — creates
   a RuleContent commit on the current branch.
2. **LLM tool calls**: `create_rule` tool through the pending/consult
   pattern. Produces the same commit.
3. **Natural language**: LLM parses at authoring time into structured rule.
   Concrete instructions, not vague preferences.

## Rule Engine Runtime

### Rule Index

Rules are collected via DAG ancestry traversal and cached in an index:

```
RuleIndex: dict[(trigger, name) → RuleContent + dag_distance]
```

Maintained incrementally:
- **Commit (non-rule)** → index unchanged
- **Commit (RuleContent)** → add/update entry, distance=0, bump existing +1
- **Switch/branch** → rebuild from new branch ancestry
- **Merge/rebase** → rebuild (ancestry changed)

Same invalidation pattern as the compile cache. Lookup is O(1) by trigger.

### EvalContext

```python
@dataclass(frozen=True)
class EvalContext:
    event: str                  # the trigger that fired
    commit: CommitInfo | None   # the triggering commit (if applicable)
    branch: str                 # current branch name
    head: str                   # current HEAD hash
    tract: Tract                # for operations and queries
    metrics: MetricRegistry     # for threshold conditions
    rule_index: RuleIndex       # for introspection
```

Condition evaluators and action handlers both receive this.

### Two Engine Modes

The engine has two distinct modes behind the same "rule" abstraction:

**Event processing** — triggered by commit, transition, compress, etc.
Collects matching rules, evaluates conditions, executes actions in the
gates → work → handoff → post pipeline.

**Config resolution** — for `trigger="active"` rules. Nothing "fires" them.
Instead, when the system needs a config value (compile_strategy, temperature),
it queries active rules for that key and resolves by DAG precedence. This is
a key-value store with scoped inheritance, not an event handler.

### Execution Flow (Event Processing)

```
event fires ("commit", "transition:ads", etc.)
  → look up matching rules from index by trigger (O(1))
  → group by action category: gates | work | handoff | post
  → for gates:
      sort by cost (deterministic first), then DAG distance
      evaluate conditions — first block/require failure stops everything
  → for work:
      sort by DAG distance (furthest first = root before branch)
      evaluate conditions, execute passing actions (accumulate: all run)
  → for handoff:
      sort by DAG distance (closest wins = override)
      evaluate conditions, execute first passing action only
  → for post:
      sort by DAG distance (furthest first)
      evaluate conditions, execute passing actions
  → return result (blocked | success + action results)
```

### Short-Circuit Optimization

Conditions evaluate in cost order within each category:
1. `condition=None` → always fires, free
2. `type` in (tag, pattern, threshold) → deterministic, cheap
3. `type="llm"` → expensive, evaluate last

If a deterministic gate blocks, all LLM evaluations are skipped.

### Recursion Guard

Actions can trigger new events (e.g., operation action runs compress →
fires "compress" trigger). Engine tracks evaluation depth. At depth > 3,
nested events execute the raw operation without rule evaluation. Same
pattern as the current hook system's `_fire_hook` guard.

### DAG Traversal

compile() and the rule engine both walk commit ancestry, but their needs differ:
- The compiler needs the full chain for edit resolution and priority filtering.
- The rule engine only needs rule commits (content_type="rule").

These are **separate walks** — the compiler keeps its own `_walk_chain` (which
handles merge blocks, edit maps, etc.), while the rule engine uses a simpler
`walk_ancestry(content_type_filter={"rule"})` that only collects rule commits.
The overhead of a second walk is minimal (rule commits are sparse in the DAG).

## Extensibility

Condition types, action types, metrics, and triggers are all **registries**
following the same pattern as tract's content type registry:

```python
# Custom condition
engine.register_condition("embedding_similarity", EmbeddingSimilarityEvaluator)

# Custom action
engine.register_action("notify_slack", SlackNotifyHandler)

# Custom metric (for threshold conditions)
engine.register_metric("campaign_spend", CampaignSpendMetric)

# Custom trigger
engine.register_trigger("webhook:shopify", ShopifyWebhookTrigger)
```

Evaluators and handlers follow protocols:
```python
class ConditionEvaluator(Protocol):
    def evaluate(self, params: dict, ctx: EvalContext) -> bool: ...

class ActionHandler(Protocol):
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult: ...
```

EvalContext provides: triggering commit, current branch HEAD, Tract instance,
event details, metric registry.

## Rule Storage

Rules are **first-class commits** with a RuleContent content type, committed
to the branch they scope to. This gives rules all substrate operations for
free: EDIT to revise, DELETE to remove, BRANCH to experiment with variants.

Rules are typically committed at the start of a branch (the "configuration
preamble") but can be appended anywhere — learned rules from the promotion
loop naturally appear interleaved with work commits. Rule commits survive
compression (implicitly preserved by the engine).

## Persistence and Artifacts

Long-running workflows persist important outputs as external artifacts
(markdown files, code, images) referenced by commits in the DAG. Cross-
iteration context flows through artifact references + selective transition
rules, not by querying deep historical branches.

## Multi-Agent Model

Parallel agents work on separate branches and merge results (fan-out /
fan-in). Urgent cross-agent communication uses cherry-pick broadcast:
a commit tagged `urgent` triggers a rule that appends it to sibling
branches. No messaging infrastructure needed — substrate operations
composed by rules.

## Edge Cases and Mitigations

**Transition non-atomicity**: Transitions mutate two branches (work actions
commit to source in step 3, handoff commits to target in step 6). If step 6
fails, source already has step 3's commits. This is by design — work actions
on source are valid regardless. Retry skips past committed work and retries
the handoff. Rules and action handlers should be idempotent.

**Circular workflows**: Ecommerce loops don't cause infinite transitions
because `t.transition()` is an explicit agent call, not automatic. Stopping
conditions are just more rules (iteration counter thresholds, budget limits).

**Emergency rule override**: DAG precedence is local-first (can't override
from workflow root). Use `t.broadcast_rule()` to commit override rules to
all active descendant branches.

**same_context stage boundaries**: When stages share a branch, use tag
annotations to mark stage boundaries for visualization.
