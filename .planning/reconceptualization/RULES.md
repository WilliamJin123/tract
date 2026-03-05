# The Unified Rule System

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
"commit"                  # after every commit
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

**new_agent** — fresh context, handoff payload only:
```
{mode: "new_agent", include_tags: [...], max_tokens: N, instruction: "..."}
```

## Compile Behavior

RuleContent commits are **never compiled to messages**. During compile,
the engine separates rule commits from content commits. Rules configure
the compilation; content becomes the message list. Implemented via a
`compilable=False` hint on the ContentTypeHints for the rule content type.

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

**Partial transition failure**: If a transition action fails mid-execution,
earlier actions (audits, notifications) have already committed/fired. The
transition can be retried. External side effects need idempotent handlers.

**Circular workflows**: Ecommerce loops don't cause infinite transitions
because `t.transition()` is an explicit agent call, not automatic. Stopping
conditions are just more rules (iteration counter thresholds, budget limits).

**Emergency rule override**: DAG precedence is local-first (can't override
from workflow root). Use `t.broadcast_rule()` to commit override rules to
all active descendant branches.

**same_context stage boundaries**: When stages share a branch, use tag
annotations to mark stage boundaries for visualization.
