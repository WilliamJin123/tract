# Phase R2: Event Processing + Actions

> **Status: SUPERSEDED by Phase 14 (Config + Directives + Middleware).** Rule engine was implemented (R0-R4, commit 7a86b94) then replaced (commit 23a89eb). Kept as historical reference.

## Goal

Build the event processing mode of the rule engine: triggers fire on events
(commit, compile, compress, merge, gc, transition), conditions are evaluated,
actions execute in the gates->work->handoff->post pipeline.

**Depends on:** R1 (rule index, conditions, config resolution all exist)

## Architecture

```
src/tract/rules/
    engine.py           # RuleEngine: event processing orchestrator
    actions.py          # Action handlers (set_config, operation, block, require, etc.)
```

Plus modifications to `tract.py` to wire event processing into commit/compile/
compress/merge/gc/transition operations.

## Task Breakdown

### Task 2.1: Action Handlers (`rules/actions.py`)

```python
from __future__ import annotations
from typing import Protocol


class ActionHandler(Protocol):
    """Protocol for action type handlers."""
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult: ...


class SetConfigAction:
    """Set a configuration parameter. Override semantics.

    Returns config data in ActionResult for the caller to apply. The engine
    deduplicates set_config results by key (override semantics: closest to
    HEAD wins). The caller (_fire_rules in tract.py) extracts the deduped
    set_config results from EvalResult and applies them.

    For persistent config changes, use trigger="active" rules (resolved via
    RuleIndex.get_config). Event-driven set_config is for transient overrides
    within a single event processing cycle.
    """
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        # params: {"key": str, "value": any}
        # Store in EvalResult for the caller to apply
        return ActionResult(
            action_type="set_config",
            success=True,
            data={"key": params["key"], "value": params["value"]},
        )


class OperationAction:
    """Run a substrate operation. Accumulate semantics."""

    # All supported substrate operations and their Tract method mappings
    SUPPORTED_OPS = {
        "compress", "branch", "annotate", "edit", "merge",
        "rebase", "delete", "cherry_pick", "gc",
    }

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        # params: {"op": str, "params": dict}
        # Dispatch to tract substrate operations
        op = params["op"]
        op_params = params.get("params", {})
        t = ctx.tract

        if op not in self.SUPPORTED_OPS:
            raise ValueError(f"Unknown operation: {op!r}. Supported: {sorted(self.SUPPORTED_OPS)}")

        if op == "compress":
            result = t.compress(**op_params)
            return ActionResult("operation", True, {"op": "compress", "result": str(result)})
        elif op == "branch":
            result = t.create_branch(**op_params)
            return ActionResult("operation", True, {"op": "branch", "result": str(result)})
        elif op == "annotate":
            t.annotate(**op_params)
            return ActionResult("operation", True, {"op": "annotate"})
        elif op == "edit":
            result = t.commit(op_params.pop("content"), operation="edit", **op_params)
            return ActionResult("operation", True, {"op": "edit", "result": str(result)})
        elif op == "merge":
            result = t.merge(**op_params)
            return ActionResult("operation", True, {"op": "merge", "result": str(result)})
        elif op == "rebase":
            result = t.rebase(**op_params)
            return ActionResult("operation", True, {"op": "rebase", "result": str(result)})
        elif op == "delete":
            t.delete_commit(**op_params)
            return ActionResult("operation", True, {"op": "delete"})
        elif op == "cherry_pick":
            result = t.import_commit(**op_params)
            return ActionResult("operation", True, {"op": "cherry_pick", "result": str(result)})
        elif op == "gc":
            result = t.gc(**op_params)
            return ActionResult("operation", True, {"op": "gc", "result": str(result)})


class BlockAction:
    """Prevent the triggering operation/transition. Accumulate semantics."""
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        reason = params.get("reason", "Blocked by rule")
        return ActionResult("block", True, {"blocked": True}, reason=reason)


class RequireAction:
    """Block until embedded condition is met. Accumulate semantics."""
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.conditions import evaluate_condition
        inner_condition = params.get("condition")
        met = evaluate_condition(inner_condition, ctx)
        if met:
            return ActionResult("require", True, {"met": True})
        return ActionResult("require", False, {"met": False},
                          reason=f"Requirement not met: {inner_condition}")


class CompileFilterAction:
    """Configure transition handoff compilation. Override semantics."""
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        # params: {"mode": str, ...mode_params}
        # Store filter config in result for transition handler to use
        return ActionResult("compile_filter", True, data=params)


# Placeholder for LLM conditions/actions (mocked in tests, real in R4)
class LLMAction:
    """LLM-evaluated action. Accumulate semantics."""
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        # params: {"instruction": str}
        # For now: return a placeholder. Real impl needs LLM client.
        return ActionResult("llm", True, {"instruction": params["instruction"],
                                          "note": "LLM evaluation deferred"})


class CreateRuleAction:
    """Commit a new RuleContent. Accumulate semantics.

    IMPORTANT: Does NOT commit during event processing. Instead, returns
    the template in ActionResult.data for deferred commitment. The engine
    collects all create_rule results and commits them in the post-processing
    step after the event pipeline completes. This prevents rule index
    invalidation mid-iteration.
    """
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        template = params.get("template", {})
        # Validate template has required fields
        if not all(k in template for k in ("name", "trigger", "action")):
            return ActionResult("create_rule", False,
                              reason="Template missing required fields: name, trigger, action")
        # Return template for deferred commit (do NOT commit here)
        return ActionResult("create_rule", True, {"template": template, "deferred": True})


# Registry
BUILTIN_ACTIONS: dict[str, ActionHandler] = {
    "set_config": SetConfigAction(),
    "operation": OperationAction(),
    "block": BlockAction(),
    "require": RequireAction(),
    "compile_filter": CompileFilterAction(),
    "llm": LLMAction(),
    "create_rule": CreateRuleAction(),
}

# Semantics metadata
ACTION_SEMANTICS: dict[str, str] = {
    "set_config": "override",
    "compile_filter": "override",
    "block": "accumulate",
    "require": "accumulate",
    "llm": "accumulate",
    "operation": "accumulate",
    "create_rule": "accumulate",
}

# Category ordering
ACTION_CATEGORIES: dict[str, str] = {
    "require": "gate",
    "block": "gate",
    "llm": "work",
    "operation": "work",
    "compile_filter": "handoff",
    "set_config": "work",
    "create_rule": "post",
}
```

### Task 2.2: Rule Engine (`rules/engine.py`)

```python
class RuleEngine:
    """Evaluates rules against events.

    Two modes:
    1. Event processing: commit/compile/compress/merge/gc/transition
       Gates -> Work -> Handoff -> Post pipeline.
    2. Config resolution: trigger="active" rules as key-value store.
       (Already implemented in RuleIndex.get_config)
    """

    def __init__(
        self,
        rule_index: RuleIndex,
        *,
        custom_conditions: dict[str, ConditionEvaluator] | None = None,
        custom_actions: dict[str, ActionHandler] | None = None,
        max_depth: int = 3,
    ) -> None:
        self._rule_index = rule_index
        self._custom_conditions = custom_conditions or {}
        self._custom_actions = custom_actions or {}
        self._max_depth = max_depth
        # NOTE: _eval_depth lives on Tract, NOT on RuleEngine.
        # RuleEngine is re-instantiated when rule_index goes stale
        # (e.g., after a CreateRuleAction commits a new rule), so a
        # depth counter on the engine would reset mid-recursion and
        # defeat the guard. Tract is single-threaded (SQLite-backed),
        # so this mutable state on Tract is safe.

    def process_event(
        self,
        event: str,
        ctx: EvalContext,
    ) -> EvalResult:
        """Process an event through the rule pipeline.

        1. Collect rules matching the event trigger
        2. Group by action category: gates | work | handoff | post
        3. Evaluate each category in order
        4. Short-circuit: gates fail -> skip everything
        5. Recursion guard: depth > max_depth -> skip rule eval

        Returns EvalResult with blocked status and action results.
        """
        # Recursion guard -- depth counter lives on Tract to survive
        # engine re-instantiation (e.g., when a CreateRuleAction commits
        # a new rule and the index goes stale).
        depth = getattr(ctx.tract, '_rule_eval_depth', 0)
        if depth >= self._max_depth:
            return EvalResult()

        ctx.tract._rule_eval_depth = depth + 1
        try:
            return self._process_event_inner(event, ctx)
        finally:
            ctx.tract._rule_eval_depth = depth

    def _process_event_inner(self, event: str, ctx: EvalContext) -> EvalResult:
        # 1. Collect matching rules
        rules = self._rule_index.get_by_trigger(event)
        if not rules:
            return EvalResult()

        # 2. Group by category
        categories = self._group_by_category(rules)

        all_results = []
        rules_evaluated = 0
        rules_fired = 0

        # 3. Gates (must all pass)
        for rule in categories.get("gate", []):
            rules_evaluated += 1
            if not self._check_condition(rule.condition, ctx):
                continue
            rules_fired += 1
            result = self._execute_action(rule.action, ctx)
            all_results.append(result)
            if result.action_type == "block":
                return EvalResult(
                    blocked=True,
                    block_reasons=[result.reason or "Blocked"],
                    action_results=all_results,
                    rules_evaluated=rules_evaluated,
                    rules_fired=rules_fired,
                )
            if result.action_type == "require" and not result.success:
                return EvalResult(
                    blocked=True,
                    block_reasons=[result.reason or "Requirement not met"],
                    action_results=all_results,
                    rules_evaluated=rules_evaluated,
                    rules_fired=rules_fired,
                )

        # 4. Work (all matching execute, accumulate)
        #    EXCEPTION: set_config uses override semantics — post-process
        #    deduplication keeps only the closest rule's value per key.
        for rule in categories.get("work", []):
            rules_evaluated += 1
            if not self._check_condition(rule.condition, ctx):
                continue
            rules_fired += 1
            result = self._execute_action(rule.action, ctx)
            all_results.append(result)

        # Post-process: deduplicate set_config results (override semantics)
        # Work rules sort furthest-first, so the LAST set_config for a key
        # is from the closest rule. Keep only that one per key.
        all_results = self._dedup_set_config(all_results)

        # 5. Handoff (override: closest wins, first match only)
        for rule in categories.get("handoff", []):
            rules_evaluated += 1
            if not self._check_condition(rule.condition, ctx):
                continue
            rules_fired += 1
            result = self._execute_action(rule.action, ctx)
            all_results.append(result)
            break  # override: first match wins

        # 6. Post (all matching execute)
        for rule in categories.get("post", []):
            rules_evaluated += 1
            if not self._check_condition(rule.condition, ctx):
                continue
            rules_fired += 1
            result = self._execute_action(rule.action, ctx)
            all_results.append(result)

        return EvalResult(
            action_results=all_results,
            rules_evaluated=rules_evaluated,
            rules_fired=rules_fired,
        )

    def _group_by_category(
        self, rules: list[RuleEntry]
    ) -> dict[str, list[RuleEntry]]:
        """Group rules by action category, sorted appropriately.

        Gates: sort by cost (deterministic first), then DAG distance
        Work: sort by DAG distance descending (furthest first = root before branch)
        Handoff: sort by DAG distance ascending (closest wins)
        Post: sort by DAG distance descending (furthest first)
        """
        groups: dict[str, list[RuleEntry]] = {}
        for rule in rules:
            action_type = rule.action.get("type", "")
            category = ACTION_CATEGORIES.get(action_type, "work")
            groups.setdefault(category, []).append(rule)

        # Sort gates: deterministic first, then by distance
        if "gate" in groups:
            groups["gate"].sort(key=lambda r: (
                0 if self._is_deterministic(r) else 1,
                r.dag_distance,
            ))

        # Sort work: furthest first (root before branch)
        if "work" in groups:
            groups["work"].sort(key=lambda r: -r.dag_distance)

        # Sort handoff: closest first (branch before root)
        if "handoff" in groups:
            groups["handoff"].sort(key=lambda r: r.dag_distance)

        # Sort post: furthest first
        if "post" in groups:
            groups["post"].sort(key=lambda r: -r.dag_distance)

        return groups

    def _is_deterministic(self, rule: RuleEntry) -> bool:
        """Check if a rule's condition is fully deterministic (no LLM)."""
        if rule.condition is None:
            return True
        return rule.condition.get("type") != "llm"

    def _check_condition(self, condition: dict | None, ctx: EvalContext) -> bool:
        from tract.rules.conditions import evaluate_condition
        return evaluate_condition(condition, ctx,
                                custom_conditions=self._custom_conditions)

    def _execute_action(self, action: dict, ctx: EvalContext) -> ActionResult:
        action_type = action.get("type", "")
        handler = (self._custom_actions.get(action_type) or
                   BUILTIN_ACTIONS.get(action_type))
        if handler is None:
            return ActionResult(action_type, False, reason=f"Unknown action: {action_type!r}")
        return handler.execute(action, ctx)

    @staticmethod
    def _dedup_set_config(results: list[ActionResult]) -> list[ActionResult]:
        """Deduplicate set_config results by key (override semantics).

        Work rules sort furthest-first (root before branch), so the LAST
        set_config for a given key is from the closest rule (highest precedence).
        Keep only the last occurrence of each key; preserve all non-set_config results.
        """
        # Walk backwards to find the first (= closest) occurrence of each key
        seen_keys: set[str] = set()
        keep_indices: set[int] = set()
        for i in range(len(results) - 1, -1, -1):
            r = results[i]
            if r.action_type == "set_config":
                key = r.data.get("key")
                if key not in seen_keys:
                    seen_keys.add(key)
                    keep_indices.add(i)
                # else: skip (farther rule, lower precedence)
            else:
                keep_indices.add(i)
        return [results[i] for i in sorted(keep_indices)]
```

### Task 2.3: Wire Event Processing into Tract

**Modify `tract.py`:**

Add `_rule_eval_depth` to Tract `__init__` and `_rule_engine` property:

```python
# In __init__:
self._rule_eval_depth: int = 0  # recursion guard, survives engine rebuild

@property
def _rule_engine(self) -> RuleEngine:
    """Get the rule engine (lazy init, re-created when index is stale)."""
    if self.__rule_engine is None or self._rule_index_stale:
        from tract.rules.engine import RuleEngine
        self.__rule_engine = RuleEngine(self.rule_index)
    return self.__rule_engine

def _fire_rules(self, event: str, commit: CommitInfo | None = None) -> EvalResult:
    """Fire rules for an event. Returns EvalResult.

    Post-processes deferred actions (e.g., create_rule commits) after the
    event pipeline completes. This prevents rule index invalidation mid-iteration.
    """
    from tract.rules.models import EvalContext

    # Pre-compute metrics so condition evaluators never call compile().
    # Use cached token count if available, else 0 (first compile).
    metrics = {}
    if self._cache_manager:
        head = self.head
        if head:
            snapshot = self._cache_manager.get(head)
            if snapshot:
                metrics["total_tokens"] = snapshot.token_count
    metrics.setdefault("total_tokens", 0)

    ctx = EvalContext(
        event=event,
        commit=commit,
        branch=self.current_branch or "",
        head=self.head or "",
        tract=self,
        metrics=metrics,
        rule_index=self.rule_index,
    )
    result = self._rule_engine.process_event(event, ctx)

    # Post-process: commit deferred create_rule actions
    for ar in result.action_results:
        if ar.action_type == "create_rule" and ar.success and ar.data.get("deferred"):
            template = ar.data["template"]
            self.rule(
                name=template["name"],
                trigger=template["trigger"],
                condition=template.get("condition"),
                action=template["action"],
            )

    return result
```

**Wire into operations:**

```python
# In commit():
def commit(self, content, ...):
    # ... existing commit logic ...
    # Fire pre_commit event (gates can block BEFORE persisting)
    eval_result = self._fire_rules("pre_commit")
    if eval_result.blocked:
        raise TraceError(f"Commit blocked: {eval_result.block_reasons}")
    result = self._do_commit(content, ...)
    # Fire post-commit event (reactive: redaction, auto-tagging, auto-compress)
    self._fire_rules("commit", commit=result)
    return result

# In compile():
def compile(self, ...):
    # Fire compile event (gates can block)
    eval_result = self._fire_rules("compile")
    if eval_result.blocked:
        raise TraceError(f"Compile blocked: {eval_result.block_reasons}")
    # ... existing compile logic ...

# In compress():
def compress(self, ...):
    eval_result = self._fire_rules("compress")
    if eval_result.blocked:
        return None  # or raise, TBD
    # ... existing compress logic ...

# In merge():
def merge(self, ...):
    eval_result = self._fire_rules("merge")
    if eval_result.blocked:
        raise MergeError(f"Merge blocked: {eval_result.block_reasons}")
    # ... existing merge logic ...

# In gc():
def gc(self, ...):
    eval_result = self._fire_rules("gc")
    if eval_result.blocked:
        return None
    # ... existing gc logic ...
```

**Transition method (NEW):**

```python
def transition(self, target: str, **kwargs) -> CommitInfo | None:
    """Transition to a target branch/stage using rules.

    Evaluates transition rules on the current branch:
    1. Gates: require/block rules
    2. Work: pre-transition actions (audits, cleanup)
    3. Handoff: compile_filter to build payload
    4. Switch to / create target branch
    5. Commit handoff payload on target

    IMPORTANT: Rules from BOTH "transition" (generic) and "transition:{target}"
    (specific) triggers are collected and processed through a SINGLE
    gates→work→handoff→post pipeline. This ensures gate failures from either
    trigger prevent work actions from executing.

    Args:
        target: Target branch name.

    Returns:
        CommitInfo of the handoff commit on the target, or None if blocked.
    """
    # Collect rules from BOTH generic and specific triggers into one batch
    eval_result = self._fire_transition_rules(target)
    if eval_result.blocked:
        return None

    # Extract compile_filter from handoff results
    compile_filter = None
    for ar in eval_result.action_results:
        if ar.action_type == "compile_filter":
            compile_filter = ar.data
            break

    # Build handoff payload
    source = self.current_branch
    if compile_filter:
        mode = compile_filter.get("mode", "full")
        if mode == "selective":
            include_tags = set(compile_filter.get("include_tags", []))
            compiled = self.compile()
            # Filter to messages from commits with matching tags
            payload_messages = []
            for msg, commit_hash in zip(compiled.messages, compiled.commit_hashes):
                commit = self._commit_repo.get(commit_hash)
                commit_tags = set(json.loads(commit.tags_json or "[]"))
                if commit_tags & include_tags:
                    payload_messages.append(msg)
            payload_text = "\n\n".join(m.content for m in payload_messages if m.content)
        elif mode == "summarized":
            target_tokens = compile_filter.get("target_tokens", 500)
            instruction = compile_filter.get("instruction", "Summarize for handoff.")
            # Use compression to create a summary at target token budget
            compiled = self.compile()
            payload_text = str(compiled.to_dicts())  # fallback; real impl uses LLM summary
        elif mode == "same_context":
            # No branching, no handoff payload -- just append rules on current branch
            payload_text = None
        elif mode == "new_agent":
            include_tags = set(compile_filter.get("include_tags", []))
            compiled = self.compile()
            payload_messages = []
            for msg, commit_hash in zip(compiled.messages, compiled.commit_hashes):
                commit = self._commit_repo.get(commit_hash)
                commit_tags = set(json.loads(commit.tags_json or "[]"))
                if commit_tags & include_tags:
                    payload_messages.append(msg)
            payload_text = "\n\n".join(m.content for m in payload_messages if m.content)
        else:
            payload_text = str(self.compile().to_dicts())
    else:
        payload_text = str(self.compile().to_dicts())

    # For same_context: no branch switch, just continue on current branch
    if compile_filter and compile_filter.get("mode") == "same_context":
        return None  # rules already applied, no handoff needed

    # Switch to target branch (create if needed)
    if target not in [b.name for b in self.branches]:
        self.create_branch(target)
    self.switch(target)

    # Commit handoff payload
    from tract.models.content import DialogueContent
    handoff_content = DialogueContent(
        role="system",
        text=f"Transition handoff from {source}.\n\n{payload_text}",
    )
    return self.commit(handoff_content, message=f"transition handoff from {source} to {target}",
                       tags=["transition_handoff"])


def _fire_transition_rules(self, target: str) -> EvalResult:
    """Collect rules from both 'transition' and 'transition:{target}' triggers
    and process through a SINGLE gates→work→handoff→post pipeline.

    This ensures gate failures from either trigger prevent work actions from
    either trigger from executing, maintaining the ordering guarantee.
    """
    from tract.rules.models import EvalContext

    metrics = {}
    if self._cache_manager:
        head = self.head
        if head:
            snapshot = self._cache_manager.get(head)
            if snapshot:
                metrics["total_tokens"] = snapshot.token_count
    metrics.setdefault("total_tokens", 0)

    ctx = EvalContext(
        event=f"transition:{target}",
        commit=None,
        branch=self.current_branch or "",
        head=self.head or "",
        tract=self,
        metrics=metrics,
        rule_index=self.rule_index,
    )
    return self._rule_engine.process_transition(target, ctx)
```

**Add to `rules/engine.py`:**

```python
def process_transition(self, target: str, ctx: EvalContext) -> EvalResult:
    """Process a transition by collecting rules from both generic and
    specific triggers into a single pipeline.

    Combines rules from 'transition' and 'transition:{target}' triggers,
    then processes them through the standard gates→work→handoff→post pipeline.
    """
    # Recursion guard
    depth = getattr(ctx.tract, '_rule_eval_depth', 0)
    if depth >= self._max_depth:
        return EvalResult()

    ctx.tract._rule_eval_depth = depth + 1
    try:
        # Collect rules from both triggers
        generic_rules = self._rule_index.get_by_trigger("transition")
        specific_rules = self._rule_index.get_by_trigger(f"transition:{target}")
        all_rules = generic_rules + specific_rules

        if not all_rules:
            return EvalResult()

        # Process through single unified pipeline
        categories = self._group_by_category(all_rules)
        return self._execute_pipeline(categories, ctx)
    finally:
        ctx.tract._rule_eval_depth = depth
```

### Task 2.4: LLM Condition (placeholder)

```python
class LLMCondition:
    """LLM-evaluated condition. Expensive, evaluated last."""
    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        # params: {"instruction": str}
        # Needs LLM client on tract. If no client, raise or return True.
        llm = getattr(ctx.tract, '_llm_client', None)
        if llm is None:
            # No LLM available -- pass through (permissive default)
            return True
        # Real implementation: send instruction + context to LLM, parse bool
        # Deferred to R4 for full implementation
        return True
```

Add to `BUILTIN_CONDITIONS`:
```python
BUILTIN_CONDITIONS["llm"] = LLMCondition()
```

---

## Test Plan

### `tests/test_rule_actions.py` (~20 tests)

- `test_set_config_action` -- returns key/value in data
- `test_block_action` -- returns blocked=True
- `test_block_action_with_reason` -- custom reason
- `test_require_action_met` -- condition met, success=True
- `test_require_action_not_met` -- condition not met, success=False
- `test_operation_action_compress` -- dispatches to compress
- `test_operation_action_branch` -- dispatches to branch
- `test_operation_action_unknown` -- raises ValueError
- `test_compile_filter_action` -- stores mode params in data
- `test_llm_action_placeholder` -- returns deferred note
- `test_create_rule_action` -- commits new rule
- `test_action_semantics_override` -- set_config, compile_filter are "override"
- `test_action_semantics_accumulate` -- block, require, operation are "accumulate"

### `tests/test_rule_engine.py` (~35 tests)

- `test_no_rules_no_effect` -- empty index, event fires cleanly
- `test_single_gate_block` -- block action stops everything
- `test_single_gate_require_met` -- requirement met, continues
- `test_single_gate_require_not_met` -- requirement not met, blocked
- `test_work_actions_all_execute` -- accumulate: all work rules fire
- `test_work_order_furthest_first` -- root rules before branch rules
- `test_handoff_closest_wins` -- only first matching handoff executes
- `test_post_actions_after_handoff` -- post runs after handoff
- `test_gate_blocks_skips_work` -- gate failure skips work/handoff/post
- `test_deterministic_before_llm` -- gate sorting: deterministic first
- `test_recursion_guard_depth_3` -- nested events capped at depth 3
- `test_recursion_guard_resets` -- depth resets after event completes
- `test_condition_evaluation` -- conditions dispatched correctly
- `test_unconditional_rule` -- condition=None always fires
- `test_multiple_triggers_same_event` -- all matching rules collected
- `test_mixed_categories` -- gate+work+handoff+post in one event
- `test_override_semantics` -- set_config closest wins
- `test_accumulate_semantics` -- block: any block stops
- `test_commit_event_fires` -- commit triggers commit rules
- `test_compile_event_fires` -- compile triggers compile rules
- `test_compress_event_fires` -- compress triggers compress rules
- `test_merge_event_fires` -- merge triggers merge rules
- `test_gc_event_fires` -- gc triggers gc rules
- `test_transition_event_fires` -- transition triggers both generic and specific
- `test_transition_specific_target` -- "transition:ads" matches
- `test_transition_generic_and_specific` -- both fire, generic first
- `test_eval_result_aggregation` -- results collected correctly
- `test_process_event_returns_eval_result` -- correct type returned

### `tests/test_rule_engine_integration.py` (~20 tests)

- `test_rule_blocks_compress` -- compress blocked by rule
- `test_rule_blocks_merge` -- merge blocked by rule
- `test_rule_auto_compress_on_threshold` -- token threshold triggers compress
- `test_rule_auto_branch` -- commit triggers branch creation
- `test_transition_with_compile_filter` -- selective handoff
- `test_transition_blocked_by_require` -- missing tag blocks transition
- `test_transition_after_approval` -- add tag, transition succeeds
- `test_config_from_rules` -- temperature resolved from active rule
- `test_config_override_on_branch` -- branch rule overrides root
- `test_rule_created_by_rule` -- create_rule action
- `test_multiple_events_in_sequence` -- commit -> compile -> transition
- `test_rule_preserves_data_on_compress` -- tag condition blocks compress for tagged commits
- `test_ecommerce_workflow_skeleton` -- multi-stage with transition rules

---

## Acceptance Criteria

1. `RuleEngine.process_event()` processes gates->work->handoff->post in order
2. Short-circuit: deterministic gates before LLM, gate failure skips everything
3. Recursion guard: depth > 3 skips rule evaluation
4. Override semantics for set_config/compile_filter (closest wins)
5. Accumulate semantics for block/require/operation (all execute)
6. `t.transition(target)` evaluates transition rules and performs handoff
7. Events fire from commit/compile/compress/merge/gc operations
8. All ~75 new tests pass
9. All surviving R0+R1 tests still pass
