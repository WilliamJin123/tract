# Phase R1: Rule Engine Core

> **Status: SUPERSEDED by Phase 14 (Config + Directives + Middleware).** Rule engine was implemented (R0-R4, commit 7a86b94) then replaced (commit 23a89eb). Kept as historical reference.

## Goal

Build the rule engine: rule index, condition evaluators, config resolution mode,
and the `t.rule()` API. After this phase, developers can create rules, the engine
can evaluate deterministic conditions, and configs resolve via DAG precedence.

**Depends on:** R0 (RuleContent type exists, compiler skips non-compilable)

## Architecture

```
src/tract/rules/
    __init__.py          # Public API exports
    models.py            # EvalContext, RuleEntry, RuleIndex, ActionResult
    index.py             # RuleIndex: build, maintain, query
    conditions.py        # Condition evaluators (tag, pattern, threshold, logic)
    config.py            # Config resolution (active trigger key-value store)
    ancestry.py          # walk_ancestry(filter) shared traversal
```

## Task Breakdown

### Task 1.1: Shared Ancestry Walk (`rules/ancestry.py`)

The compiler and rule engine both need to walk commit ancestry. Currently the
compiler does this inline in `_walk_chain`. Extract a shared primitive.

```python
def walk_ancestry(
    commit_repo: CommitRepository,
    blob_repo: BlobRepository,
    head_hash: str,
    *,
    content_type_filter: set[str] | None = None,
    parent_repo: CommitParentRepository | None = None,
) -> list[CommitRow]:
    """Walk DAG ancestry from head, optionally filtering by content type.

    Returns commits in root-to-head order.

    Args:
        content_type_filter: If provided, only include commits whose
            content_type is in this set. None = include all.
    """
```

Used by `RuleIndex.build()` to collect rule commits from DAG ancestry.
The compiler keeps its own `_walk_chain` for compilation (it needs the full
chain for edit resolution) -- this function does NOT replace the compiler.

### Task 1.2: Rule Models (`rules/models.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.rules.index import RuleIndex
    from tract.tract import Tract


@dataclass(frozen=True)
class RuleEntry:
    """A rule with its DAG context."""
    name: str
    trigger: str
    condition: dict | None
    action: dict
    commit_hash: str        # the commit containing this rule
    dag_distance: int       # distance from HEAD (0 = closest)
    provenance: dict | None # commit metadata (source: developer/llm/promoted)


@dataclass(frozen=True)
class EvalContext:
    """Immutable context passed to condition evaluators and action handlers."""
    event: str                           # the trigger that fired
    commit: CommitInfo | None            # triggering commit (if applicable)
    branch: str                          # current branch name
    head: str                            # current HEAD hash
    tract: Tract                         # for operations and queries
    metrics: dict[str, Any] | None = None  # MetricRegistry (deferred to R4)
    rule_index: RuleIndex | None = None  # for introspection


@dataclass(frozen=True)
class ActionResult:
    """Result from executing a single action."""
    action_type: str
    success: bool
    data: dict = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class EvalResult:
    """Result from evaluating rules for an event."""
    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    action_results: list[ActionResult] = field(default_factory=list)
    rules_evaluated: int = 0
    rules_fired: int = 0
```

### Task 1.3: Rule Index (`rules/index.py`)

```python
class RuleIndex:
    """In-memory index of active rules, built from DAG ancestry.

    Index structure: dict[(trigger, name) -> RuleEntry]
    When multiple rules share (trigger, name), closest to HEAD wins.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], RuleEntry] = {}
        self._head_hash: str | None = None

    @classmethod
    def build(
        cls,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        head_hash: str,
        *,
        parent_repo: CommitParentRepository | None = None,
    ) -> RuleIndex:
        """Build index by walking ancestry and collecting RuleContent commits.

        Rules at distance 0 (closest to HEAD) override rules at distance N
        with the same (trigger, name) key.

        Distance computation: walk_ancestry() returns commits in root-to-head
        order. Distance is computed as: distance = len(rule_commits) - 1 - index.
        This gives distance 0 to the commit closest to HEAD (last in the list)
        and the highest distance to the root-most rule commit.
        """

    def get_by_trigger(self, trigger: str) -> list[RuleEntry]:
        """Get all rules matching a trigger, sorted by dag_distance ascending."""

    def get_config(self, key: str) -> Any | None:
        """Resolve a config value from active rules.

        Looks for rules with trigger="active" and
        action={"type": "set_config", "key": key, ...}.
        Returns the value from the closest rule (lowest dag_distance).
        """

    def get_all_configs(self) -> dict[str, Any]:
        """Resolve all active config values."""

    def add_rule(self, entry: RuleEntry) -> None:
        """Add or update a rule entry (used for incremental maintenance)."""

    def invalidate(self) -> None:
        """Mark index as stale (requires rebuild on next access)."""

    @property
    def is_stale(self) -> bool: ...

    def __len__(self) -> int:
        """Number of unique (trigger, name) entries."""

    def __contains__(self, key: tuple[str, str]) -> bool: ...
```

**Incremental maintenance rules (same as compile cache):**
- Non-rule commit: index unchanged
- Rule commit: add/update entry at distance=0, bump all existing distances +1
- Switch/branch: rebuild
- Merge/rebase: rebuild

### Task 1.4: Condition Evaluators (`rules/conditions.py`)

```python
from __future__ import annotations
from typing import Protocol

class ConditionEvaluator(Protocol):
    """Protocol for condition type evaluators."""
    def evaluate(self, params: dict, ctx: EvalContext) -> bool: ...


class TagCondition:
    """Check if a tag exists on the triggering commit or in scope."""
    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        # params: {"tag": str, "present": bool}
        # Check commit's tags_json for the tag
        ...

class PatternCondition:
    """Regex match on commit content."""
    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        # params: {"regex": str}
        # Load commit content, re.search(regex, content)
        ...

class ThresholdCondition:
    """Numeric comparison on a metric."""

    VALID_OPS = {">", "<", "==", ">=", "<=", "!="}

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        # params: {"metric": str, "op": str, "value": number}
        # Built-in metrics: token_count, total_tokens, commit_count,
        #                    age_hours, branch_depth
        op = params.get("op", "")
        if op not in self.VALID_OPS:
            raise ValueError(f"Invalid threshold operator: {op!r}. Valid: {sorted(self.VALID_OPS)}")
        ...

class AllCondition:
    """AND combinator."""
    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        # params: {"conditions": [condition_dict, ...]}
        ...

class AnyCondition:
    """OR combinator."""
    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        # params: {"conditions": [condition_dict, ...]}
        ...

class NotCondition:
    """Negation."""
    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        # params: {"condition": condition_dict}
        ...


# Registry of built-in condition evaluators
BUILTIN_CONDITIONS: dict[str, ConditionEvaluator] = {
    "tag": TagCondition(),
    "pattern": PatternCondition(),
    "threshold": ThresholdCondition(),
    "all": AllCondition(),
    "any": AnyCondition(),
    "not": NotCondition(),
}


def evaluate_condition(
    condition: dict | None,
    ctx: EvalContext,
    *,
    custom_conditions: dict[str, ConditionEvaluator] | None = None,
) -> bool:
    """Evaluate a condition dict. None = always True."""
    if condition is None:
        return True
    ctype = condition.get("type")
    evaluator = (custom_conditions or {}).get(ctype) or BUILTIN_CONDITIONS.get(ctype)
    if evaluator is None:
        raise ValueError(f"Unknown condition type: {ctype!r}")
    return evaluator.evaluate(condition, ctx)
```

**Built-in metrics for ThresholdCondition:**

```python
def _get_metric(name: str, ctx: EvalContext) -> float:
    """Resolve a built-in metric value.

    IMPORTANT: Never call ctx.tract.compile() here -- this function may be
    called during a "compile" event, which would cause infinite recursion.
    All metrics must use pre-computed values from ctx.metrics or lightweight
    queries (log length, commit fields).

    NOTE (cold start): total_tokens defaults to 0 when no compile cache exists
    (first compile in a session). This means token-based threshold rules will
    not fire until after the first compile populates the cache. This is by
    design to avoid expensive traversals on every event fire.
    """
    # Check pre-computed metrics first (set by _fire_rules in tract.py)
    if ctx.metrics and name in ctx.metrics:
        return ctx.metrics[name]
    if name == "token_count" and ctx.commit:
        return ctx.commit.token_count or 0
    if name == "total_tokens":
        # Pre-computed by _fire_rules() from cache or last known value.
        # Falls back to 0 if not available (first compile, no cache).
        return (ctx.metrics or {}).get("total_tokens", 0)
    if name == "commit_count":
        return len(ctx.tract.log())
    if name == "age_hours" and ctx.commit:
        delta = datetime.now(timezone.utc) - ctx.commit.created_at
        return delta.total_seconds() / 3600
    if name == "branch_depth":
        return len(ctx.tract.log())  # approximate
    # Deferred: custom metrics via MetricRegistry (R4)
    raise ValueError(f"Unknown metric: {name!r}")
```

### Task 1.5: Config Resolution (`rules/config.py`)

```python
def resolve_config(
    rule_index: RuleIndex,
    key: str,
    *,
    default: Any = None,
) -> Any:
    """Resolve a config value from active rules.

    Looks through rules with trigger="active" for set_config actions
    matching the given key. Returns the value from the closest rule
    (lowest dag_distance). If no rule matches, returns default.

    This is the "config resolution" mode of the rule engine -- not
    event-driven, just a scoped key-value store.
    """
    return rule_index.get_config(key) or default


def resolve_all_configs(rule_index: RuleIndex) -> dict[str, Any]:
    """Resolve all active config key-value pairs."""
    return rule_index.get_all_configs()
```

### Task 1.6: Wire t.rule() API on Tract

**Add to `tract.py`:**

```python
def rule(
    self,
    name: str,
    *,
    trigger: str,
    condition: dict | None = None,
    action: dict,
    message: str | None = None,
    tags: list[str] | None = None,
) -> CommitInfo:
    """Create a rule by committing a RuleContent to the current branch.

    Args:
        name: Stable identity for the rule (human-readable).
        trigger: When the engine evaluates this rule
            ("active", "commit", "compile", "compress", "merge",
             "gc", "transition", "transition:{target}").
        condition: Condition dict or None (always fires).
        action: Action dict (required).
        message: Optional commit message. Defaults to f"rule: {name}".
        tags: Optional tags for the rule commit.

    Returns:
        CommitInfo for the rule commit.
    """
    from tract.models.content import RuleContent

    content = RuleContent(
        name=name,
        trigger=trigger,
        condition=condition,
        action=action,
    )
    return self.commit(
        content,
        message=message or f"rule: {name}",
        tags=tags,
    )
```

**Add to `tract.py`:**

```python
def metadata(
    self,
    kind: str,
    data: dict | str,
    *,
    path: str | None = None,
    message: str | None = None,
    tags: list[str] | None = None,
) -> CommitInfo:
    """Create or update a metadata entry.

    Args:
        kind: Freeform label ("file_tree", "project_plan", etc.).
        data: Structured or text content.
        path: Optional filesystem path for export/sync.
        message: Optional commit message.
        tags: Optional tags.

    Returns:
        CommitInfo for the metadata commit.
    """
    from tract.models.content import MetadataContent

    content = MetadataContent(kind=kind, data=data, path=path)
    return self.commit(
        content,
        message=message or f"metadata: {kind}",
        tags=tags,
    )
```

**Add rule_index property:**

```python
@property
def rule_index(self) -> RuleIndex:
    """Get the current rule index (built/cached from DAG ancestry)."""
    if self._rule_index is None or self._rule_index.is_stale:
        from tract.rules.index import RuleIndex
        head = self.head
        if head is None:
            return RuleIndex()
        self._rule_index = RuleIndex.build(
            self._commit_repo, self._blob_repo, head,
            parent_repo=self._parent_repo,
        )
    return self._rule_index
```

**Add config resolution:**

```python
def get_config(self, key: str, default: Any = None) -> Any:
    """Resolve a config value from active rules.

    Uses DAG precedence: closest to HEAD wins.
    """
    from tract.rules.config import resolve_config
    return resolve_config(self.rule_index, key, default=default)
```

**Invalidate rule index on state changes:**
- After `commit()` if content is RuleContent: `self._rule_index_stale = True`
- After `switch()`, `checkout()`: rebuild
- After `merge()`, `rebase()`: rebuild

---

## Test Plan

### `tests/test_rule_models.py` (~15 tests)

- `test_rule_entry_creation` -- all fields populate
- `test_rule_entry_frozen` -- immutable
- `test_eval_context_creation` -- all fields
- `test_eval_context_frozen` -- immutable
- `test_action_result_defaults` -- data={}, reason=None
- `test_eval_result_defaults` -- blocked=False, empty lists

### `tests/test_rule_index.py` (~30 tests)

- `test_build_empty` -- no rules = empty index
- `test_build_single_rule` -- one rule commit
- `test_build_multiple_rules` -- several rules, different triggers
- `test_precedence_closer_wins` -- two rules same name, closer to HEAD wins
- `test_precedence_same_distance_later_wins` -- timestamp tiebreak
- `test_different_names_coexist` -- same trigger, different names
- `test_get_by_trigger` -- returns correct subset
- `test_get_by_trigger_empty` -- no matches = empty list
- `test_get_config_active_rule` -- active rule with set_config action
- `test_get_config_override` -- closer rule overrides farther
- `test_get_config_missing` -- returns None/default
- `test_get_all_configs` -- multiple keys
- `test_incremental_add` -- add_rule updates index
- `test_invalidate_and_rebuild` -- stale flag works
- `test_len_and_contains` -- __len__, __contains__
- `test_build_skips_non_rule_commits` -- dialogue commits ignored
- `test_build_across_branches` -- rules inherited from parent branch
- `test_build_with_merge` -- rules from both parents collected
- `test_rule_on_workflow_root` -- inherited by all branches
- `test_rule_disabled_via_skip` -- SKIP annotation disables rule

### `tests/test_conditions.py` (~25 tests)

- `test_none_condition_always_true` -- None = unconditional
- `test_tag_present_true` -- tag exists
- `test_tag_present_false` -- tag doesn't exist
- `test_tag_absent_check` -- present=False
- `test_pattern_match` -- regex matches content
- `test_pattern_no_match` -- regex doesn't match
- `test_pattern_complex_regex` -- multiline, groups
- `test_threshold_gt` -- greater than
- `test_threshold_lt` -- less than
- `test_threshold_eq` -- equal
- `test_threshold_gte` -- >=
- `test_threshold_lte` -- <=
- `test_threshold_token_count` -- built-in metric
- `test_threshold_total_tokens` -- built-in metric
- `test_threshold_commit_count` -- built-in metric
- `test_threshold_unknown_metric` -- raises ValueError
- `test_all_combinator` -- all conditions must pass
- `test_all_short_circuit` -- stops on first failure
- `test_any_combinator` -- any condition passes
- `test_any_short_circuit` -- stops on first success
- `test_not_combinator` -- inverts result
- `test_nested_combinators` -- all(tag, any(pattern, threshold))
- `test_unknown_condition_type` -- raises ValueError
- `test_evaluate_condition_dispatches` -- correct evaluator called

### `tests/test_rule_config.py` (~10 tests)

- `test_resolve_config_basic` -- single active rule
- `test_resolve_config_override` -- closer rule wins
- `test_resolve_config_default` -- no rule, returns default
- `test_resolve_all_configs` -- multiple keys resolved
- `test_config_through_facade` -- t.get_config(key)
- `test_config_after_rule_commit` -- new rule changes config
- `test_config_on_branch` -- branch-scoped config
- `test_config_inherited` -- parent branch config inherited by child

### `tests/test_rule_api.py` (~15 tests)

- `test_t_rule_creates_commit` -- creates RuleContent commit
- `test_t_rule_fields` -- all fields populated correctly
- `test_t_rule_default_message` -- "rule: {name}"
- `test_t_rule_custom_message` -- message= override
- `test_t_rule_with_tags` -- tags applied
- `test_t_rule_condition_none` -- unconditional rule
- `test_t_metadata_creates_commit` -- MetadataContent committed
- `test_t_metadata_dict_data` -- dict data preserved
- `test_t_metadata_str_data` -- string data preserved
- `test_t_metadata_with_path` -- path field set
- `test_rule_index_property` -- t.rule_index returns RuleIndex
- `test_rule_index_invalidated_on_rule_commit` -- stale after rule commit
- `test_rule_index_invalidated_on_switch` -- stale after branch switch
- `test_rule_not_in_compile` -- rule commit excluded from compile()
- `test_metadata_not_in_compile` -- metadata excluded from compile()

---

## Acceptance Criteria

1. `t.rule(name, trigger=..., action=...)` creates a RuleContent commit
2. `t.metadata(kind, data)` creates a MetadataContent commit
3. RuleIndex builds from DAG ancestry, respects precedence
4. `t.get_config(key)` resolves from active rules
5. All 6 condition types work: tag, pattern, threshold, all, any, not
6. Rule/metadata commits are excluded from `t.compile()`
7. All ~95 new tests pass
8. All surviving R0 tests still pass
