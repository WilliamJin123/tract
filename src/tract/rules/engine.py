"""Rule engine: event processing orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tract.rules.actions import ACTION_CATEGORIES, BUILTIN_ACTIONS, ActionHandler
from tract.rules.models import ActionResult, EvalResult

if TYPE_CHECKING:
    from tract.rules.conditions import ConditionEvaluator
    from tract.rules.index import RuleIndex
    from tract.rules.models import EvalContext, RuleEntry


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
        """
        depth = getattr(ctx.tract, "_rule_eval_depth", 0)
        if depth >= self._max_depth:
            return EvalResult()

        ctx.tract._rule_eval_depth = depth + 1
        try:
            return self._process_event_inner(event, ctx)
        finally:
            ctx.tract._rule_eval_depth = depth

    def process_transition(self, target: str, ctx: EvalContext) -> EvalResult:
        """Process a transition by collecting rules from both generic and
        specific triggers into a single pipeline."""
        depth = getattr(ctx.tract, "_rule_eval_depth", 0)
        if depth >= self._max_depth:
            return EvalResult()

        ctx.tract._rule_eval_depth = depth + 1
        try:
            generic_rules = self._rule_index.get_by_trigger("transition")
            specific_rules = self._rule_index.get_by_trigger(f"transition:{target}")
            all_rules = generic_rules + specific_rules

            if not all_rules:
                return EvalResult()

            categories = self._group_by_category(all_rules)
            return self._execute_pipeline(categories, ctx)
        finally:
            ctx.tract._rule_eval_depth = depth

    def _process_event_inner(self, event: str, ctx: EvalContext) -> EvalResult:
        rules = self._rule_index.get_by_trigger(event)
        if not rules:
            return EvalResult()

        categories = self._group_by_category(rules)
        return self._execute_pipeline(categories, ctx)

    def _execute_pipeline(
        self,
        categories: dict[str, list[RuleEntry]],
        ctx: EvalContext,
    ) -> EvalResult:
        """Execute the gates -> work -> handoff -> post pipeline."""
        all_results: list[ActionResult] = []
        rules_evaluated = 0
        rules_fired = 0

        # 1. Gates (must all pass)
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

        # 2. Work (all matching execute, accumulate)
        for rule in categories.get("work", []):
            rules_evaluated += 1
            if not self._check_condition(rule.condition, ctx):
                continue
            rules_fired += 1
            result = self._execute_action(rule.action, ctx)
            all_results.append(result)

        # Post-process: deduplicate set_config results (override semantics)
        all_results = self._dedup_set_config(all_results)

        # 3. Handoff (override: closest wins, first match only)
        for rule in categories.get("handoff", []):
            rules_evaluated += 1
            if not self._check_condition(rule.condition, ctx):
                continue
            rules_fired += 1
            result = self._execute_action(rule.action, ctx)
            all_results.append(result)
            break  # override: first match wins

        # 4. Post (all matching execute)
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
        """Group rules by action category, sorted appropriately."""
        groups: dict[str, list[RuleEntry]] = {}
        for rule in rules:
            action_type = rule.action.get("type", "")
            category = ACTION_CATEGORIES.get(action_type, "work")
            groups.setdefault(category, []).append(rule)

        # Sort gates: deterministic first, then by distance
        if "gate" in groups:
            groups["gate"].sort(
                key=lambda r: (
                    0 if self._is_deterministic(r) else 1,
                    r.dag_distance,
                )
            )

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
        if rule.condition is None:
            return True
        return rule.condition.get("type") != "llm"

    def _check_condition(self, condition: dict | None, ctx: EvalContext) -> bool:
        from tract.rules.conditions import evaluate_condition

        try:
            return evaluate_condition(
                condition, ctx, custom_conditions=self._custom_conditions
            )
        except Exception:
            # Malformed conditions fail closed (don't fire the rule)
            return False

    def _execute_action(self, action: dict, ctx: EvalContext) -> ActionResult:
        action_type = action.get("type", "")
        handler = self._custom_actions.get(action_type) or BUILTIN_ACTIONS.get(
            action_type
        )
        if handler is None:
            return ActionResult(
                action_type, False, reason=f"Unknown action: {action_type!r}"
            )
        try:
            return handler.execute(action, ctx)
        except Exception as exc:
            return ActionResult(
                action_type, False, reason=f"Action failed: {exc}"
            )

    @staticmethod
    def _dedup_set_config(results: list[ActionResult]) -> list[ActionResult]:
        """Deduplicate set_config results by key (override semantics).

        Work rules sort furthest-first (root before branch), so the LAST
        set_config for a given key is from the closest rule. Keep only
        the last occurrence of each key.
        """
        seen_keys: set[str] = set()
        keep_indices: set[int] = set()
        for i in range(len(results) - 1, -1, -1):
            r = results[i]
            if r.action_type == "set_config":
                key = r.data.get("key")
                if key not in seen_keys:
                    seen_keys.add(key)
                    keep_indices.add(i)
            else:
                keep_indices.add(i)
        return [results[i] for i in sorted(keep_indices)]
